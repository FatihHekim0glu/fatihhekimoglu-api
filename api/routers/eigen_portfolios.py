"""Eigen Portfolios (PCA) tool router.

Endpoint:
  POST /tools/eigen-portfolios/run — fetch a returns panel, PCA-decompose it,
  separate signal/noise eigenvalues via Marchenko-Pastur, return spectrum +
  top-K eigen-portfolios + plotly figures.

Pipeline
--------
1. Resolve the universe:
   - ``sp500-pit`` → S&P 500 membership as-of ``start_date`` (point-in-time).
   - ``custom``    → caller-provided list of tickers.
2. Fetch daily adjusted closes via :class:`PolygonProvider` (or yfinance
   fallback). Drop tickers with <60% coverage.
3. Compute simple daily returns; drop NaN rows; require ≥ ``2 * N`` obs.
4. PCA on the correlation matrix; MP bulk via :func:`fit_sigma`.
5. Build interpretation labels for the top-K eigenvectors using
   ``get_ticker_meta`` sector lookups (lru_cached; falls back silently on
   ``yfinance``).
6. Render three Plotly figures (spectrum, factor-return cum, weights heatmap).
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from datetime import date
from functools import lru_cache
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from ..deps import get_supabase
from ..lib.eigen_portfolios import (
    EigenResult,
    bulk_density,
    eigen_decompose,
    fit_sigma,
    marchenko_pastur_bulk,
)
from ..lib.polygon.exceptions import PolygonError
from ..lib.polygon.provider import (
    PolygonProvider,
    PolygonProviderFallback,
    make_provider,
)
from ..lib.polygon.sp500_universe import SP500UniverseBuilder

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools/eigen-portfolios", tags=["eigen-portfolios"])


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

Universe = Literal["sp500-pit", "custom"]
Rebalance = Literal["none"]

_MIN_COVERAGE = 0.60          # drop tickers with < 60% non-NaN closes
_MIN_OBS_PER_N = 2            # require T >= 2*N for the PCA to be well-posed
_TOP_WEIGHTS = 10             # per top-K eigvector, show this many heaviest names
_MAX_CUSTOM_TICKERS = 600
_MIN_CUSTOM_TICKERS = 2
_MAX_HEATMAP_N = 60           # heatmap is unreadable past ~60 rows


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class EigenPortfoliosRequest(BaseModel):
    universe: Universe = "sp500-pit"
    tickers: list[str] | None = None
    start_date: date = date(2020, 1, 2)
    end_date: date = date(2024, 12, 31)
    n_components_to_show: int = Field(5, ge=2, le=10)
    rebalance: Rebalance = "none"

    @field_validator("tickers")
    @classmethod
    def _normalise_tickers(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        cleaned: list[str] = []
        seen: set[str] = set()
        for raw in v:
            t = raw.strip().upper()
            if not t or t in seen:
                continue
            seen.add(t)
            cleaned.append(t)
        if len(cleaned) > _MAX_CUSTOM_TICKERS:
            raise ValueError(
                f"custom universe is capped at {_MAX_CUSTOM_TICKERS} tickers"
            )
        return cleaned

    @field_validator("end_date")
    @classmethod
    def _end_after_start(cls, v: date, info: Any) -> date:
        start = info.data.get("start_date")
        if start is not None and v <= start:
            raise ValueError("end_date must be strictly after start_date")
        return v


class TopWeight(BaseModel):
    ticker: str
    weight: float


class TopPortfolio(BaseModel):
    rank: int
    eigenvalue: float
    explained_variance: float
    top_weights: list[TopWeight]
    interpretation: str


class RmtBulk(BaseModel):
    lambda_min: float
    lambda_max: float
    sigma_sq: float


class EigenPortfoliosResponse(BaseModel):
    data_source: Literal["polygon", "yfinance"]
    universe_size: int
    obs: int
    eigenvalue_spectrum: list[float]
    rmt_bulk: RmtBulk
    significant_count: int
    top_portfolios: list[TopPortfolio]
    spectrum_figure: dict[str, Any]
    factor_returns_figure: dict[str, Any]
    weights_heatmap_figure: dict[str, Any]


# ---------------------------------------------------------------------------
# Sector lookup (cached)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4096)
def _lookup_sector(ticker: str) -> str | None:
    """Best-effort SIC sector lookup via Polygon. Returns None on any failure."""
    # We rebuild a provider per cache-miss because the API key is process-wide
    # and the provider is cheap to construct. The lru_cache then prevents
    # repeated calls for the same ticker within the process lifetime.
    try:
        provider = make_provider()
        if isinstance(provider, PolygonProviderFallback):
            return None
        meta = provider.get_ticker_meta(ticker)
        provider.close()
    except (PolygonError, Exception):  # noqa: BLE001 — best-effort only
        return None
    # Polygon stores sector under `sic_description` or `sector`; prefer the
    # higher-level GICS-style label when available.
    for field_name in ("sic_description", "sector", "sic_industry_description"):
        value = meta.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _interpret_eigvec(
    rank: int,
    weights: pd.Series,
    *,
    weight_threshold: float = 0.1,
    top_n_fallback: int = 5,
) -> str:
    """Heuristic label for an eigen-portfolio.

    Rank 0 (top eigval) is the market factor when all loadings share the same
    sign. For other components, we collect the dominant tickers and look up
    their sectors; the modal sector wins if it accounts for ≥ 40% of names.
    """
    abs_w = weights.abs()

    if rank == 0:
        signs = np.sign(weights.to_numpy())
        nonzero = signs[signs != 0]
        if nonzero.size > 0 and abs(nonzero.sum()) / nonzero.size >= 0.85:
            return "market factor (broad long bias)"
        return "first principal component"

    selected = weights[abs_w > weight_threshold]
    if selected.empty:
        selected = weights.reindex(abs_w.sort_values(ascending=False).index[:top_n_fallback])

    sectors: list[str] = []
    for ticker in selected.index:
        sector = _lookup_sector(str(ticker))
        if sector:
            sectors.append(sector)

    if not sectors:
        return "diversified / mixed exposure"

    counter = Counter(sectors)
    modal_sector, modal_count = counter.most_common(1)[0]
    share = modal_count / max(len(selected), 1)
    if share >= 0.40:
        return f"sector: {modal_sector.lower()} (≈{int(round(share * 100))}%)"
    return "diversified / mixed exposure"


# ---------------------------------------------------------------------------
# Universe + price plumbing
# ---------------------------------------------------------------------------


def _resolve_universe(
    req: EigenPortfoliosRequest,
    provider: PolygonProvider | PolygonProviderFallback,
    supabase: Any,
) -> list[str]:
    if req.universe == "custom":
        if not req.tickers or len(req.tickers) < _MIN_CUSTOM_TICKERS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"custom universe requires at least {_MIN_CUSTOM_TICKERS} tickers"
                ),
            )
        return list(req.tickers)

    if isinstance(provider, PolygonProviderFallback):
        # No grouped-daily without a Polygon key; we cannot resolve PIT
        # membership offline.
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "S&P 500 PIT universe requires POLYGON_API_KEY; "
                "switch to a custom ticker list or configure the key."
            ),
        )

    builder = SP500UniverseBuilder(provider, supabase_client=supabase)
    members = builder.get_membership_as_of(req.start_date)
    if not members:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"S&P 500 PIT membership empty for {req.start_date.isoformat()}",
        )
    return members


def _build_returns_panel(
    tickers: list[str],
    start: date,
    end: date,
    provider: PolygonProvider | PolygonProviderFallback,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Fetch closes for each ticker, align, drop low-coverage names.

    Returns ``(returns_df, kept_tickers, dropped_tickers)``.
    """
    closes: dict[str, pd.Series] = {}
    dropped: list[str] = []
    for ticker in tickers:
        try:
            df = provider.get_eod(ticker, start, end)
        except (PolygonError, Exception) as exc:  # noqa: BLE001
            logger.info("get_eod failed for %s (%s); dropping", ticker, exc)
            dropped.append(ticker)
            continue
        if df is None or df.empty or "Close" not in df.columns:
            dropped.append(ticker)
            continue
        closes[ticker] = df["Close"].astype(float)

    if not closes:
        return pd.DataFrame(), [], dropped

    panel = pd.DataFrame(closes).sort_index()
    # Coverage filter: drop tickers whose non-NaN ratio is below threshold.
    coverage = panel.notna().mean(axis=0)
    keep_cols = coverage[coverage >= _MIN_COVERAGE].index.tolist()
    dropped.extend(t for t in panel.columns if t not in keep_cols)
    panel = panel[keep_cols].sort_index()

    # Forward-fill small gaps, then compute simple returns, then drop NaN rows.
    panel = panel.ffill()
    returns = panel.pct_change().dropna(how="any")
    return returns, keep_cols, dropped


# ---------------------------------------------------------------------------
# Plotly figures
# ---------------------------------------------------------------------------


_NAVY = "#0B2E4F"
_AMBER = "#E0A82E"
_CRIMSON = "#A11D33"
_TEAL = "#0d9488"


def _spectrum_figure(
    eigvals: np.ndarray,
    bulk: dict[str, float],
    n_assets: int,
    n_obs: int,
) -> go.Figure:
    """Histogram of empirical eigenvalues plus the MP bulk density overlay."""
    fig = go.Figure()
    if eigvals.size > 0:
        # Bound histogram x-range by the bulk so the market eigvalue doesn't
        # blow up the scale. We surface the outliers via an annotation.
        cap = max(bulk["lambda_max"] * 2.0, float(np.percentile(eigvals, 90)))
        fig.add_trace(
            go.Histogram(
                x=eigvals[eigvals <= cap],
                xbins={"start": 0.0, "end": float(cap), "size": float(cap / 60.0)},
                histnorm="probability density",
                name="Empirical",
                marker={"color": _NAVY},
                opacity=0.7,
                hovertemplate="λ=%{x:.3f}<br>density=%{y:.3f}<extra></extra>",
            )
        )

    # Overlay MP density.
    edges_x = np.linspace(
        max(bulk["lambda_min"] * 0.5, 1e-3),
        bulk["lambda_max"] * 1.05,
        300,
    )
    density = bulk_density(n_assets, n_obs, sigma_sq=bulk["sigma_sq"])(edges_x)
    fig.add_trace(
        go.Scatter(
            x=edges_x,
            y=density,
            mode="lines",
            line={"color": _CRIMSON, "width": 2.5},
            name="Marchenko-Pastur",
            hovertemplate="λ=%{x:.3f}<br>ρ_MP=%{y:.3f}<extra></extra>",
        )
    )

    fig.add_vline(
        x=bulk["lambda_max"],
        line_dash="dash",
        line_color=_AMBER,
        annotation_text=f"λ+ = {bulk['lambda_max']:.2f}",
        annotation_position="top right",
    )

    outliers = int((eigvals > bulk["lambda_max"]).sum())
    fig.update_layout(
        title=f"Eigenvalue spectrum — {outliers} eigenvalues above the bulk",
        xaxis_title="Eigenvalue λ",
        yaxis_title="Density",
        bargap=0.04,
        template="plotly_white",
        height=420,
        legend={"orientation": "h", "y": -0.18},
    )
    return fig


def _factor_returns_figure(
    factor_returns: pd.DataFrame,
    top_k: int,
) -> go.Figure:
    """Cumulative z-score factor returns for the top-K eigen-portfolios."""
    fig = go.Figure()
    k = min(top_k, factor_returns.shape[1])
    palette = [_NAVY, _AMBER, _CRIMSON, _TEAL, "#7c3aed", "#2563eb", "#16a34a", "#dc2626", "#9333ea", "#0891b2"]
    for i in range(k):
        col = factor_returns.columns[i]
        cum = factor_returns[col].cumsum()
        fig.add_trace(
            go.Scatter(
                x=cum.index,
                y=cum.values,
                mode="lines",
                name=str(col),
                line={"color": palette[i % len(palette)], "width": 1.5},
                hovertemplate=(
                    f"{col}<br>%{{x|%Y-%m-%d}}<br>cum_z=%{{y:.2f}}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=f"Top {k} eigen-portfolio factor returns (cumulative z-score)",
        xaxis_title="",
        yaxis_title="Cumulative z-score return",
        template="plotly_white",
        height=420,
        legend={"orientation": "h", "y": -0.18},
    )
    return fig


def _weights_heatmap_figure(
    eigvecs: pd.DataFrame,
    top_k: int,
) -> go.Figure:
    """Heatmap of the top-K eigenvectors over the full universe (clipped).

    For universes wider than ``_MAX_HEATMAP_N`` we drop down to the union of
    each component's top-15 absolute loadings to keep the chart readable.
    """
    k = min(top_k, eigvecs.shape[1])
    panel = eigvecs.iloc[:, :k].copy()
    if panel.shape[0] > _MAX_HEATMAP_N:
        kept: set[str] = set()
        per_col_n = max(5, _MAX_HEATMAP_N // max(k, 1))
        for col in panel.columns:
            top_idx = panel[col].abs().sort_values(ascending=False).index[:per_col_n]
            kept.update(top_idx.tolist())
            if len(kept) >= _MAX_HEATMAP_N:
                break
        panel = panel.loc[list(kept)]
        panel = panel.reindex(
            panel.abs().sum(axis=1).sort_values(ascending=False).index
        )

    fig = go.Figure(
        go.Heatmap(
            z=panel.to_numpy(),
            x=list(panel.columns),
            y=list(panel.index),
            colorscale="RdBu",
            zmid=0.0,
            colorbar={"title": "Weight"},
            hovertemplate=(
                "%{y} | %{x}<br>weight=%{z:.3f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"Top {k} eigenvector loadings",
        xaxis_title="Component",
        yaxis_title="Ticker",
        template="plotly_white",
        height=max(320, 14 * panel.shape[0] + 80),
    )
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value: Any) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(f) or math.isinf(f):
        return 0.0
    return f


def _to_plotly_json(fig: go.Figure) -> dict[str, Any]:
    return json.loads(pio.to_json(fig, validate=False))


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/run", response_model=EigenPortfoliosResponse)
def run(
    req: EigenPortfoliosRequest,
    supabase=Depends(get_supabase),
) -> EigenPortfoliosResponse:
    """PCA decomposition of a returns panel with MP signal/noise separation."""
    try:
        provider = make_provider(supabase_client=supabase)
    except Exception as exc:
        logger.exception("provider construction failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not construct data provider: {exc}",
        ) from exc

    data_source: Literal["polygon", "yfinance"] = (
        "polygon" if isinstance(provider, PolygonProvider) else "yfinance"
    )

    try:
        tickers = _resolve_universe(req, provider, supabase)
        returns, kept, dropped = _build_returns_panel(
            tickers, req.start_date, req.end_date, provider
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("returns panel build failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Returns panel build failed: {exc}",
        ) from exc

    n_assets = returns.shape[1]
    t_obs = returns.shape[0]
    if n_assets < 2:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Fewer than 2 tickers survived coverage filtering; "
                f"dropped={len(dropped)} kept={n_assets}"
            ),
        )
    if t_obs < _MIN_OBS_PER_N * n_assets:
        # Soft-warn rather than fail — PCA still runs, but the eigenvalues are
        # unstable. The frontend surfaces ``obs`` so the caller can decide.
        logger.warning(
            "T=%d, N=%d (T < %dN) — PCA spectrum will be noisy",
            t_obs,
            n_assets,
            _MIN_OBS_PER_N,
        )

    try:
        result: EigenResult = eigen_decompose(returns)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    n_keep = result.n
    sigma_sq = fit_sigma(result.eigvals, n_keep, t_obs, n_signal=1)
    bulk = marchenko_pastur_bulk(n_keep, t_obs, sigma_sq=sigma_sq)
    sig_count = int((result.eigvals > bulk["lambda_max"]).sum())

    # Top-K portfolio details.
    k_show = min(req.n_components_to_show, result.eigvecs.shape[1])
    top_portfolios: list[TopPortfolio] = []
    for rank in range(k_show):
        col = result.eigvecs.columns[rank]
        weights = result.eigvecs[col]
        top_idx = weights.abs().sort_values(ascending=False).index[:_TOP_WEIGHTS]
        top_weights = [
            TopWeight(ticker=str(t), weight=_safe_float(weights.loc[t]))
            for t in top_idx
        ]
        top_portfolios.append(
            TopPortfolio(
                rank=rank,
                eigenvalue=_safe_float(result.eigvals[rank]),
                explained_variance=_safe_float(result.explained_var[rank]),
                top_weights=top_weights,
                interpretation=_interpret_eigvec(rank, weights),
            )
        )

    spectrum_fig = _spectrum_figure(result.eigvals, bulk, n_keep, t_obs)
    factor_fig = _factor_returns_figure(result.factor_returns, k_show)
    heatmap_fig = _weights_heatmap_figure(result.eigvecs, k_show)

    response = EigenPortfoliosResponse(
        data_source=data_source,
        universe_size=n_keep,
        obs=t_obs,
        eigenvalue_spectrum=[_safe_float(v) for v in result.eigvals.tolist()],
        rmt_bulk=RmtBulk(
            lambda_min=_safe_float(bulk["lambda_min"]),
            lambda_max=_safe_float(bulk["lambda_max"]),
            sigma_sq=_safe_float(bulk["sigma_sq"]),
        ),
        significant_count=sig_count,
        top_portfolios=top_portfolios,
        spectrum_figure=_to_plotly_json(spectrum_fig),
        factor_returns_figure=_to_plotly_json(factor_fig),
        weights_heatmap_figure=_to_plotly_json(heatmap_fig),
    )

    _maybe_log_run(supabase, req, response, dropped_count=len(dropped))
    return response


# ---------------------------------------------------------------------------
# Supabase helpers (best-effort)
# ---------------------------------------------------------------------------


def _maybe_log_run(
    supabase,
    req: EigenPortfoliosRequest,
    resp: EigenPortfoliosResponse,
    *,
    dropped_count: int,
) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "eigen-portfolios",
                "params": req.model_dump(mode="json"),
                "result": {
                    "data_source": resp.data_source,
                    "universe_size": resp.universe_size,
                    "obs": resp.obs,
                    "significant_count": resp.significant_count,
                    "dropped_count": dropped_count,
                },
                "status": "ok",
            }
        ).execute()
    except Exception:  # noqa: BLE001
        logger.exception("tool_runs insert failed (non-fatal)")
