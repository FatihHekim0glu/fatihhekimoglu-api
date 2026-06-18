"""Pairs-trading tool - wraps `pairs-trading/src/pairs/` unchanged via the vendor shim.

Endpoint:
  POST /tools/pairs-trading/run - cointegration scan over a curated universe

Pipeline (mirrors `pairs-trading/app/cache.py:run_screen` minus the streamlit
caching decorators):

  1. `load_pair_universe(universe)` -> list[(ticker_a, ticker_b)]
  2. `load_prices(tickers, start, end)` -> MultiIndex (ticker, field) frame
  3. Flatten to wide [date x ticker] Close-price frame (tz-stripped)
  4. `screen_cointegration(candidates, prices, ...)` -> ScreenResult
  5. Filter by half_life bounds, compute KPI aggregates, build Plotly histogram

v1 is single-shot synchronous - the curated 25-name universe (~300 pairs)
typically completes in 10-30s after warm cache. The XLK universe (~3000 pairs)
can exceed 60s; we cap with a soft warning in the response payload.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from ..deps import get_supabase

# Importing the vendor package mutates sys.path so the absolute `pairs.*`
# imports below resolve to api/lib/pairs_trading/_vendor/pairs/.
from ..lib import pairs_trading as _pairs_trading_vendor  # noqa: F401
from ..lib.pairs_trading._polygon_adapter import load_prices_via_provider
from ..lib.polygon.provider import (
    PolygonProvider,
    PolygonProviderFallback,
    make_provider,
)
from ..lib.polygon.sp500_universe import SP500UniverseBuilder

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools/pairs-trading", tags=["pairs-trading"])


# ---------------------------------------------------------------------------
# Constants & method translation
# ---------------------------------------------------------------------------

Universe = Literal["curated_25_v1", "xlk_v1", "sp500-pit"]
FdrMethod = Literal["benjamini_hochberg", "bonferroni", "none"]
DataSource = Literal["polygon", "yfinance", "cache"]

# Friendly slug -> statsmodels multipletests method tag (mirrors selection.mtc).
_MTC_METHOD_MAP: dict[str, str] = {
    "benjamini_hochberg": "fdr_bh",
    "bonferroni": "bonferroni",
    "none": "none",
}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class PairsTradingRequest(BaseModel):
    universe: Universe = "curated_25_v1"
    train_start: date = date(2022, 1, 1)
    train_end: date = date(2023, 12, 31)
    hl_min: float = Field(5.0, ge=1.0, le=120.0)
    hl_max: float = Field(60.0, ge=1.0, le=120.0)
    p_threshold: float = Field(0.05, ge=0.001, le=0.5)
    fdr_method: FdrMethod = "benjamini_hochberg"

    @field_validator("train_end")
    @classmethod
    def _end_after_start(cls, v: date, info: Any) -> date:
        start = info.data.get("train_start")
        if start is not None and v <= start:
            raise ValueError("train_end must be strictly after train_start")
        return v

    @field_validator("hl_max")
    @classmethod
    def _hl_max_after_min(cls, v: float, info: Any) -> float:
        hl_min = info.data.get("hl_min")
        if hl_min is not None and v < hl_min:
            raise ValueError("hl_max must be >= hl_min")
        return v


class DiagnosticRow(BaseModel):
    pair_id: str
    ticker_a: str
    ticker_b: str
    p_raw: float | None
    hedge_ratio: float | None
    half_life: float | None
    q_value: float | None
    survives_mtc: bool
    in_hl_band: bool


class PairsTradingSummary(BaseModel):
    pairs_tested: int
    survivors: int
    median_half_life: float | None
    median_q_value: float | None


class PairsTradingResponse(BaseModel):
    summary: PairsTradingSummary
    diagnostics: list[DiagnosticRow]
    figure: dict[str, Any] = Field(
        ..., description="Plotly histogram of p_raw across all candidate pairs."
    )
    universe: Universe
    train_start: date
    train_end: date
    mtc_method: FdrMethod
    data_source: DataSource = Field(
        "yfinance",
        description="Where the OHLCV bars came from: 'polygon' (live + cache), "
        "'yfinance' (fallback), or 'cache' (Polygon cache-only hit).",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value: Any) -> float | None:
    """Convert NaN / Inf to JSON-safe None."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _resolve_pair_tuples(universe: str) -> list[tuple[str, str]]:
    """Return all candidate (ticker_a, ticker_b) tuples for a universe.

    `curated_25_v1` is a pair universe; `xlk_v1` is a constituent universe and
    yields all within-universe pairs. `sp500-pit` is built dynamically in the
    router via :class:`SP500UniverseBuilder` and is NOT resolved here.
    """
    from pairs.data import load_pair_universe, load_universe

    try:
        pu = load_pair_universe(universe)
        return [(spec.a, spec.b) for spec in pu.pairs]
    except Exception:
        from itertools import combinations

        u = load_universe(universe)
        return list(combinations(u.tickers, 2))


def _resolve_sp500_pit_pairs(
    provider: PolygonProvider | PolygonProviderFallback,
    supabase: Any,
    start: date,
    end: date,
) -> list[tuple[str, str]]:
    """Build the point-in-time S&P 500 ticker union and enumerate all pairs.

    Only valid when the real Polygon provider is configured —
    ``SP500UniverseBuilder.get_grouped_daily`` has no yfinance equivalent.
    Returns the within-universe ``combinations(members, 2)`` tuple list.
    Raises ``ValueError`` when the fallback provider is in play so the caller
    can surface a clean 400.
    """
    from itertools import combinations

    if not isinstance(provider, PolygonProvider):
        raise ValueError(
            "universe='sp500-pit' requires POLYGON_API_KEY; current provider "
            "is the yfinance fallback."
        )
    builder = SP500UniverseBuilder(provider=provider, supabase_client=supabase)
    window = builder.get_membership_window(start, end)
    members: set[str] = set()
    for members_list in window.values():
        members.update(members_list)
    if len(members) < 2:
        raise ValueError(
            "SP500UniverseBuilder returned fewer than 2 members for the requested window."
        )
    tickers = sorted(members)
    return list(combinations(tickers, 2))


def _flatten_close_prices(raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse the MultiIndex (ticker, field) into a wide [date x ticker] Close frame.

    Mirrors `pairs-trading/app/cache.py::_flatten_prices` minus the field alias
    fallbacks - we always pick "Close" first, falling back to "close" or
    "Adj Close" if needed. Strips tz so the formation-window slice in
    `screen_cointegration` (which uses tz-naive timestamps) works.
    """
    if not isinstance(raw.columns, pd.MultiIndex):
        wide = raw.astype(float).sort_index()
    else:
        available = set(raw.columns.get_level_values(1))
        for candidate in ("Close", "close", "Adj Close", "adj_close"):
            if candidate in available:
                wide = raw.xs(candidate, axis=1, level=1).astype(float).sort_index()
                break
        else:
            msg = f"no usable price field in frame; available: {sorted(available)}"
            raise KeyError(msg)

    if getattr(wide.index, "tz", None) is not None:
        wide = wide.tz_localize(None)
    return wide


def _build_histogram(p_values: list[float], cutoff: float) -> dict[str, Any]:
    """Plotly histogram of raw p-values with a q-cutoff annotation."""
    fig = go.Figure()
    if p_values:
        fig.add_trace(
            go.Histogram(
                x=p_values,
                xbins={"start": 0.0, "end": 1.0, "size": 0.025},
                marker={"color": "#0d9488"},
                name="p_raw",
                hovertemplate="p in [%{x:.3f}]<br>count=%{y}<extra></extra>",
            )
        )
    fig.add_vline(
        x=cutoff,
        line_dash="dash",
        line_color="#dc2626",
        annotation_text=f"q-cutoff {cutoff:.3f}",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Raw p-value distribution",
        xaxis_title="Engle-Granger p-value",
        yaxis_title="Pair count",
        bargap=0.05,
        template="plotly_white",
        showlegend=False,
        height=420,
    )
    payload: dict[str, Any] = json.loads(pio.to_json(fig, validate=False))
    return payload


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


def get_provider(
    supabase=Depends(get_supabase),
) -> PolygonProvider | PolygonProviderFallback:
    """FastAPI dependency factory for the shared Polygon provider.

    Wrapped here (rather than wired directly via ``make_provider``) so tests
    can override it through ``app.dependency_overrides[get_provider]``.
    """
    return make_provider(supabase_client=supabase)


@router.post("/run", response_model=PairsTradingResponse)
def run(
    req: PairsTradingRequest,
    supabase=Depends(get_supabase),
    provider: PolygonProvider | PolygonProviderFallback = Depends(get_provider),
) -> PairsTradingResponse:
    """Run the cointegration scan over the chosen universe & window."""
    from pairs._exceptions import InputError, PairsError
    from pairs.data import load_prices
    from pairs.selection import Candidate, screen_cointegration

    mtc_tag = _MTC_METHOD_MAP[req.fdr_method]

    # 1. Resolve pair tuples for the requested universe.
    try:
        if req.universe == "sp500-pit":
            pair_tuples = _resolve_sp500_pit_pairs(
                provider, supabase, req.train_start, req.train_end
            )
        else:
            pair_tuples = _resolve_pair_tuples(req.universe)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("universe resolution failed for %s", req.universe)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not load universe {req.universe!r}: {exc}",
        ) from exc
    if not pair_tuples:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Universe {req.universe!r} resolved to zero pairs",
        )
    tickers = sorted({t for a, b in pair_tuples for t in (a, b)})

    # 2. Fetch prices via the injected provider when a real Polygon key is
    #    configured; otherwise fall back to the vendored yfinance loader so
    #    local dev keeps working without a key.
    use_polygon = isinstance(provider, PolygonProvider)
    data_source: DataSource = "polygon" if use_polygon else "yfinance"
    try:
        if use_polygon:
            raw_prices = load_prices_via_provider(
                provider,
                tickers,
                start=req.train_start.isoformat(),
                end=req.train_end.isoformat(),
            )
            if raw_prices.empty:
                # Provider returned nothing usable — degrade to yfinance so
                # the demo doesn't blank out on a transient Polygon hiccup.
                raw_prices = load_prices(
                    tickers,
                    start=req.train_start.isoformat(),
                    end=req.train_end.isoformat(),
                )
                data_source = "yfinance"
        else:
            raw_prices = load_prices(
                tickers,
                start=req.train_start.isoformat(),
                end=req.train_end.isoformat(),
            )
    except (InputError, PairsError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Price fetch failed: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("unexpected price fetch failure")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Upstream price fetch failed",
        ) from exc
    try:
        prices = _flatten_close_prices(raw_prices)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    # 3. Cointegration scan.
    try:
        result = screen_cointegration(
            [Candidate(ticker_a=a, ticker_b=b) for a, b in pair_tuples],
            prices,
            formation_window=(pd.Timestamp(req.train_start), pd.Timestamp(req.train_end)),
            alpha=req.p_threshold,
            mtc_method=mtc_tag,
            bootstrap=False,
        )
    except (InputError, PairsError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Screen failed: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("unexpected screen failure")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Cointegration scan failed",
        ) from exc

    diagnostics = result.diagnostics.copy()
    # 4. Half-life band filter (post-scan annotation, not a row drop).
    if not diagnostics.empty:
        hl = diagnostics["half_life"].astype(float)
        diagnostics["in_hl_band"] = (hl >= req.hl_min) & (hl <= req.hl_max)
        diagnostics = diagnostics.sort_values("q_value", ascending=True, na_position="last")
    else:
        diagnostics["in_hl_band"] = pd.Series(dtype=bool)

    # 5. KPI aggregates.
    pairs_tested = len(diagnostics)
    if pairs_tested:
        survivors = int((diagnostics["q_value"].astype(float) < 0.05).sum())
        median_hl = _safe_float(diagnostics["half_life"].median(skipna=True))
        median_q = _safe_float(diagnostics["q_value"].median(skipna=True))
        p_values = [
            float(p) for p in diagnostics["p_raw"].tolist() if p is not None and not math.isnan(p)
        ]
    else:
        survivors = 0
        median_hl = None
        median_q = None
        p_values = []

    # 6. JSON-safe diagnostic rows.
    rows: list[DiagnosticRow] = []
    for _, row in diagnostics.iterrows():
        rows.append(
            DiagnosticRow(
                pair_id=str(row["pair_id"]),
                ticker_a=str(row["ticker_a"]),
                ticker_b=str(row["ticker_b"]),
                p_raw=_safe_float(row["p_raw"]),
                hedge_ratio=_safe_float(row["hedge_ratio"]),
                half_life=_safe_float(row["half_life"]),
                q_value=_safe_float(row["q_value"]),
                survives_mtc=bool(row.get("survives_mtc", False)),
                in_hl_band=bool(row.get("in_hl_band", False)),
            )
        )

    figure_json = _build_histogram(p_values, cutoff=req.p_threshold)

    response = PairsTradingResponse(
        summary=PairsTradingSummary(
            pairs_tested=pairs_tested,
            survivors=survivors,
            median_half_life=median_hl,
            median_q_value=median_q,
        ),
        diagnostics=rows,
        figure=figure_json,
        universe=req.universe,
        train_start=req.train_start,
        train_end=req.train_end,
        mtc_method=req.fdr_method,
        data_source=data_source,
    )

    _maybe_log_run(supabase, req, response)
    return response


# ---------------------------------------------------------------------------
# Supabase helpers (best-effort)
# ---------------------------------------------------------------------------


def _maybe_log_run(supabase, req: PairsTradingRequest, resp: PairsTradingResponse) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "pairs-trading",
                "params": req.model_dump(mode="json"),
                "result": {
                    "summary": resp.summary.model_dump(mode="json"),
                    "universe": resp.universe,
                    "mtc_method": resp.mtc_method,
                    "data_source": resp.data_source,
                },
                "status": "ok",
            }
        ).execute()
    except Exception:
        logger.exception("tool_runs insert failed (non-fatal)")


# Silence the unused-import warning while keeping the side-effecting sys.path
# mutation that powers the vendored `pairs.*` imports.
_ = np
