"""Market Regime HMM tool — wraps the vendored ``regimehmm`` library.

Endpoint:
  POST /tools/regime-hmm/run — fit a Gaussian Hidden Markov Model to an index
  return series at REQUEST time (a cheap small-panel fit, no pre-trained
  artifact), CHARACTERIZE the persistent market regimes (low-vol bull /
  high-vol bear / crisis), then honestly test whether a regime-timing exposure
  overlay beats buy-and-hold out-of-sample after costs.

Mirrors ``api/routers/hrp_portfolio.py``: the headline question is "does the
regime-timing overlay beat buy-and-hold OOS, net of costs, once we deflate for
the configuration grid we explored?" The verdict is a PURE function of the OOS
inference — it cannot read "timing beats buy-and-hold" while the Memmel-Jobson-
Korkie test is insignificant or the Deflated Sharpe is non-positive.

LEAKAGE DISCIPLINE: the ONLY tradable regime signal is the ONLINE forward FILTER
posterior (data <= t). The smoothed (forward-backward) and Viterbi posteriors
peek ahead and are EDA-only, never tradable. This guard lives inside the vendored
library; the router never re-derives a signal.

v1 is single-shot synchronous: a ~6y daily SPY panel with a request-time EM fit
finishes comfortably under the request budget.

Caching:
  - Best-effort row in ``platform.tool_runs`` for run accounting; failures are
    logged but never propagate to the client.

Robustness:
  - The data layer lives inside the vendored library: Polygon failures (missing
    key / rate-limit / network / empty payload) degrade to a seeded synthetic
    regime-switch series, exactly like the markowitz/hrp tools. The call never
    hard-fails on a transient provider hiccup; ``summary.data_source`` reports
    the provenance (``polygon`` | ``synthetic``).
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date as _date
from typing import Any, Literal

import plotly.io as pio
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from ..deps import get_supabase

# Importing the vendor package side-effects ``sys.path`` so that
# ``import regimehmm`` resolves to the vendored copy. This MUST run before the
# ``from regimehmm import ...`` below, so the import block is hand-ordered and
# isort is suppressed (matching the live hrp-portfolio router convention).
from ..lib import regime_hmm as _vendor_marker  # noqa: F401

from regimehmm import (  # import resolves via the sys.path shim above
    assemble_regime_figures,
    run_regime_analysis,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools/regime-hmm", tags=["regime-hmm"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

FeatureSet = Literal["returns", "returns_vol", "returns_vol_macro"]
DataSourcePref = Literal["auto", "polygon", "synthetic"]
DataSource = Literal["polygon", "synthetic", "provided"]


class RegimeHmmRequest(BaseModel):
    """Wire contract for a regime-HMM run.

    The HMM is fit at request time on a small index panel; there is no
    pre-trained artifact. Every scalar is validated below.
    """

    ticker: str = Field(default="SPY", description="Index symbol to fit (e.g. SPY).")
    start: str | None = Field(
        default=None, description="ISO start date (YYYY-MM-DD). Defaults to ~6y ago."
    )
    end: str | None = Field(
        default=None, description="ISO end date (YYYY-MM-DD). Defaults to today."
    )
    n_states: int = Field(3, ge=2, le=4, description="Number of hidden regimes (2, 3 or 4).")
    feature_set: FeatureSet = Field(
        "returns_vol",
        description="Emission features: returns | returns_vol | returns_vol_macro.",
    )
    cost_bps: float = Field(
        10.0, ge=0.0, description="Per-side transaction cost (basis points) on exposure changes."
    )
    data_source_pref: DataSourcePref = Field(
        "auto",
        description="auto | polygon | synthetic. 'auto' tries Polygon then falls back to synthetic.",
    )
    seed: int = Field(7, ge=0, le=999_999, description="Master seed (deterministic fit + synthetic).")

    @field_validator("ticker")
    @classmethod
    def _normalise_ticker(cls, v: str) -> str:
        cleaned = v.strip().upper()
        if not cleaned:
            raise ValueError("ticker must be a non-empty string")
        return cleaned

    @field_validator("start", "end")
    @classmethod
    def _validate_iso_date(cls, v: str | None) -> str | None:
        if v is None:
            return None
        try:
            _date.fromisoformat(v)
        except ValueError as exc:
            raise ValueError(f"invalid ISO date: {v!r}") from exc
        return v

    @field_validator("end")
    @classmethod
    def _end_after_start(cls, v: str | None, info: Any) -> str | None:
        start = info.data.get("start")
        if v is not None and start is not None and v <= start:
            raise ValueError("end must be strictly after start")
        return v


class RegimeHmmResponse(BaseModel):
    summary: dict[str, Any] = Field(
        ...,
        description=(
            "Scalar summary: n_states, regime_stats (per-state mean/vol/persistence/"
            "duration/drawdown), overlay_oos_sharpe, buyhold_oos_sharpe, sharpe_diff, "
            "jk_pvalue, deflated_sharpe, n_effective_trials, verdict, data_source."
        ),
    )
    regime_figure: dict[str, Any] = Field(
        ..., description="Plotly figure JSON: regime-shaded cumulative-return series."
    )
    equity_figure: dict[str, Any] = Field(
        ..., description="Plotly figure JSON: OOS equity overlay vs buy-and-hold."
    )
    data_source: DataSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _figure_json(fig: Any) -> dict[str, Any]:
    """Normalise a library figure into a plain ``{data, layout}`` JSON dict.

    The vendored figure builders already return plain JSON-serializable mappings,
    but we round-trip through ``pio.to_json`` defensively so any stray numpy
    scalar / Timestamp is coerced exactly like the hrp-portfolio tool does.
    """
    payload: dict[str, Any] = json.loads(pio.to_json(fig, validate=False))
    return payload


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/run", response_model=RegimeHmmResponse)
def run(
    req: RegimeHmmRequest,
    supabase=Depends(get_supabase),
) -> RegimeHmmResponse:
    """Execute the regime-hmm compute pipeline (single-shot, sync).

    Pipeline (all inside the vendored library):
      load returns (Polygon → synthetic fallback) → causal features → fit +
      canonicalize HMM → ONLINE-FILTER decode (the only tradable signal) →
      characterize regimes → regime-conditioned exposure overlay vs buy-and-hold
      (after costs) → Memmel-JK Sharpe-difference + Deflated Sharpe (FULL
      effective n_trials) → honest, structurally-constrained verdict → figures.
    """
    from regimehmm import InsufficientDataError, ValidationError

    start = _date.fromisoformat(req.start) if req.start else None
    end = _date.fromisoformat(req.end) if req.end else None

    try:
        result = run_regime_analysis(
            n_states=int(req.n_states),
            feature_set=req.feature_set,
            cost_bps=float(req.cost_bps),
            seed=int(req.seed),
            ticker=req.ticker,
            start=start,
            end=end,
            data_source_pref=req.data_source_pref,
        )
    except (InsufficientDataError, ValidationError) as exc:
        logger.warning("regime-hmm run rejected: %s", exc)
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Regime-HMM run rejected: {exc.__class__.__name__}: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("regime-hmm run failed")
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Regime-HMM run failed: {exc.__class__.__name__}: {exc}",
        ) from exc

    # --- Summary scalars (every numeric coerced through _safe_float) ---------
    raw = result.summary
    summary: dict[str, Any] = {
        "n_states": int(raw["n_states"]),
        "feature_set": str(raw.get("feature_set", req.feature_set)),
        "cost_bps": _safe_float(raw.get("cost_bps", req.cost_bps)),
        "n_obs": int(raw.get("n_obs", 0)),
        "regime_stats": [
            {
                k: (_safe_float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                for k, v in stat.items()
            }
            for stat in raw.get("regime_stats", [])
        ],
        "overlay_oos_sharpe": _safe_float(raw.get("overlay_oos_sharpe")),
        "buyhold_oos_sharpe": _safe_float(raw.get("buyhold_oos_sharpe")),
        "sharpe_diff": _safe_float(raw.get("sharpe_diff")),
        "jk_pvalue": _safe_float(raw.get("jk_pvalue")),
        "deflated_sharpe": _safe_float(raw.get("deflated_sharpe")),
        "n_effective_trials": int(raw.get("n_effective_trials", 1)),
        "verdict": str(raw["verdict"]),
        "data_source": str(raw["data_source"]),
    }

    # --- Figures (best-effort; never fatal) ---------------------------------
    try:
        figures = assemble_regime_figures(result)
        regime_json = _figure_json(figures["regime_figure"])
        equity_json = _figure_json(figures["equity_figure"])
    except Exception:
        logger.exception("regime-hmm figures failed (non-fatal)")
        regime_json = _empty_figure("Regime-shaded series")
        equity_json = _empty_figure("OOS equity: overlay vs buy-and-hold")

    data_source = str(raw["data_source"])
    response = RegimeHmmResponse(
        summary=summary,
        regime_figure=regime_json,
        equity_figure=equity_json,
        data_source=data_source,  # type: ignore[arg-type]
    )

    _maybe_log_run(supabase, req, response)
    return response


def _empty_figure(title: str) -> dict[str, Any]:
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.update_layout(title=title, height=360)
    payload: dict[str, Any] = json.loads(pio.to_json(fig, validate=False))
    return payload


# ---------------------------------------------------------------------------
# Supabase helpers (best-effort)
# ---------------------------------------------------------------------------


def _maybe_log_run(supabase, req: RegimeHmmRequest, resp: RegimeHmmResponse) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "regime-hmm",
                "params": req.model_dump(mode="json"),
                "result": {
                    "data_source": resp.data_source,
                    "summary": resp.summary,
                },
                "status": "ok",
            }
        ).execute()
    except Exception:
        logger.exception("tool_runs insert failed (non-fatal)")


def _maybe_log_failure(supabase, req: RegimeHmmRequest, exc: BaseException) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "regime-hmm",
                "params": req.model_dump(mode="json"),
                "result": None,
                "status": "error",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        ).execute()
    except Exception:
        logger.exception("tool_runs failure insert failed (non-fatal)")
