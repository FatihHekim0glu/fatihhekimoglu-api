"""Markowitz Optimizer tool - wraps the vendored ``markowitz`` library.

Endpoint:
  POST /tools/markowitz-optimizer/run - fit estimators, compute efficient
  frontier and tangency portfolio, return frontier curve + weights + summary
  + Plotly figure JSON.

Mirrors app/pages/1_efficient_frontier.py from the source repo: the headline
page is the efficient frontier (40-point grid in vol/return space) plus the
tangency (max-Sharpe) portfolio under long-only + max-weight box constraints.

v1 is single-shot synchronous: a 10-ticker / 5-year universe finishes well
under 5 s on the default settings. Long-running flag was False in the recon.

Caching:
  - Best-effort row in ``platform.tool_runs`` for run accounting; failures are
    logged but never propagate to the client.

Robustness:
  - The vendored library degrades gracefully when ``cvxpy`` is unavailable
    (pseudo-inverse tangency + closed-form Merton frontier). We mirror the
    Streamlit fallback here so a missing solver does not break the request.
  - yfinance failures (rate-limit / network) fall back to a seeded synthetic
    return panel, matching app/pages/1_efficient_frontier.py.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date as _date
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from ..deps import get_supabase

# Importing the vendor package side-effects ``sys.path`` so that
# ``import markowitz`` resolves to the vendored copy. Keep this import even if
# unused for type-checking purposes.
from ..lib import markowitz_optimizer as _vendor_marker  # noqa: F401
from ..lib.polygon.provider import (
    PolygonProvider,
    PolygonProviderFallback,
    make_provider,
)
from ..lib.polygon.sp500_universe import SP500UniverseBuilder

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools/markowitz-optimizer", tags=["markowitz-optimizer"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

CovMethod = Literal["Sample", "LedoitWolf"]
MeanMethod = Literal["Sample", "JorionBayesStein", "CAPM"]
Universe = Literal["custom", "sp500-pit"]
DataSource = Literal["polygon", "yfinance", "cache", "synthetic"]

_TRADING_DAYS = 252
_FRONTIER_POINTS = 40
_ACTIVE_WEIGHT_TOL = 1e-4


class MarkowitzRequest(BaseModel):
    """Mirrors the source ``SidebarConfig`` dataclass.

    ``tickers`` is the comma-separated form so the wire contract stays in line
    with what the React form sends. We parse + dedupe + uppercase in the
    validator below.
    """

    tickers: str = Field(
        default="AAPL,MSFT,GOOG,AMZN,JPM,XOM,JNJ,PG,KO,WMT",
        description="Comma-separated ticker symbols (min 2).",
    )
    start: str = Field(..., description="ISO start date (YYYY-MM-DD).")
    end: str = Field(..., description="ISO end date (YYYY-MM-DD).")
    cov_method: CovMethod = Field("LedoitWolf")
    mean_method: MeanMethod = Field("JorionBayesStein")
    long_only: bool = Field(True)
    max_weight: float = Field(0.4, ge=0.05, le=1.0)
    risk_free_rate: float = Field(0.04, ge=-0.05, le=0.25)
    use_real_data: bool = Field(True)
    seed: int = Field(7, ge=0, le=999_999)
    universe: Universe = Field(
        "custom",
        description=(
            "'custom' uses the ``tickers`` field as-is; 'sp500-pit' builds a "
            "point-in-time S&P 500 membership from the Polygon survivorship "
            "builder over [start, end] and unions all members (only available "
            "when POLYGON_API_KEY is configured)."
        ),
    )

    @field_validator("tickers")
    @classmethod
    def _normalise_tickers(cls, v: str) -> str:
        # Accept commas and newlines like the Streamlit sidebar.
        raw = [t.strip().upper() for t in v.replace("\n", ",").split(",")]
        cleaned = [t for t in raw if t]
        if len(cleaned) < 2:
            raise ValueError("at least 2 ticker symbols are required")
        # Dedupe while preserving order.
        seen: dict[str, None] = {}
        for t in cleaned:
            seen.setdefault(t, None)
        return ",".join(seen)

    @field_validator("end")
    @classmethod
    def _end_after_start(cls, v: str, info: Any) -> str:
        start = info.data.get("start")
        if start is not None and v <= start:
            raise ValueError("end must be strictly after start")
        return v


class FrontierPoint(BaseModel):
    volatility: float
    expected_return: float
    label: str | None = None


class MarkowitzSummary(BaseModel):
    expected_return: float
    volatility: float
    sharpe: float | None
    n_assets: int


class MarkowitzResponse(BaseModel):
    frontier: list[FrontierPoint]
    weights: dict[str, float]
    summary: MarkowitzSummary
    frontier_figure: dict[str, Any] = Field(
        ..., description="Plotly figure JSON: {data, layout} - risk/return scatter."
    )
    weights_figure: dict[str, Any] = Field(
        ..., description="Plotly figure JSON: {data, layout} - horizontal weights bar."
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


def _parse_ticker_list(tickers: str) -> tuple[str, ...]:
    return tuple(t for t in (s.strip().upper() for s in tickers.split(",")) if t)


# ---------------------------------------------------------------------------
# Data loading - mirrors app/pages/1_efficient_frontier.py
# ---------------------------------------------------------------------------


def _synthetic_returns(tickers: tuple[str, ...], start: str, end: str, seed: int) -> pd.DataFrame:
    """Deterministic synthetic returns; identical math to the source page."""
    rng = np.random.default_rng(seed)
    n_assets = len(tickers)
    idx = pd.bdate_range(start=start, end=end)
    n = len(idx)
    if n < 30:
        idx = pd.bdate_range(end=end, periods=_TRADING_DAYS * 3)
        n = len(idx)
    drifts = rng.uniform(0.05, 0.14, size=n_assets) / float(_TRADING_DAYS)
    vols = rng.uniform(0.15, 0.35, size=n_assets) / np.sqrt(float(_TRADING_DAYS))
    a = rng.normal(size=(n_assets, n_assets))
    base = a @ a.T / n_assets
    d = np.sqrt(np.diag(base))
    corr = base / np.outer(d, d)
    cov = np.outer(vols, vols) * corr
    chol = np.linalg.cholesky(cov + 1e-8 * np.eye(n_assets))
    z = rng.standard_normal(size=(n, n_assets))
    eps = z @ chol.T
    rets = drifts + eps
    return pd.DataFrame(rets, index=idx, columns=list(tickers))


def _load_returns(req: MarkowitzRequest) -> tuple[pd.DataFrame, str]:
    """Legacy yfinance loader - retained for tests / non-provider callers.

    Returns ``(returns_df, data_source)``. Falls back to synthetic on any
    error so the demo still produces a plot when yfinance is unreachable.
    Prefer :func:`_load_returns_via_provider` when a Polygon provider is
    available - it shares cache with the rest of the platform.
    """
    tickers = _parse_ticker_list(req.tickers)
    if req.use_real_data:
        try:
            import yfinance as yf

            data = yf.download(
                list(tickers),
                start=req.start,
                end=req.end,
                progress=False,
                auto_adjust=True,
            )
            prices = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
            prices = prices.dropna(how="all").ffill().dropna()
            rets = prices.pct_change().dropna()
            if rets.empty:
                raise RuntimeError("no returns after cleaning")
            # Preserve column ordering against the request, dropping any
            # tickers yfinance silently dropped (e.g. delisted).
            keep = [t for t in tickers if t in rets.columns]
            if len(keep) < 2:
                raise RuntimeError("fewer than 2 usable tickers after fetch")
            return rets[keep], "yfinance"
        except Exception as exc:
            logger.warning(
                "yfinance fetch failed for %s (%s); using synthetic returns",
                tickers,
                exc.__class__.__name__,
            )
    return _synthetic_returns(tickers, req.start, req.end, req.seed), "synthetic"


def _resolve_universe_tickers(
    req: MarkowitzRequest,
    provider: PolygonProvider | PolygonProviderFallback,
    supabase: Any,
) -> tuple[str, ...]:
    """Return the ticker tuple to fetch given ``req.universe``.

    'custom' returns the parsed ticker list verbatim. 'sp500-pit' walks the
    SP500UniverseBuilder over the request window and unions every member
    seen in the period, giving Markowitz a survivorship-bias-free universe.
    Requires :class:`PolygonProvider` (not the fallback); raises ValueError
    otherwise so the caller can surface a clean 400.
    """
    if req.universe == "custom":
        return _parse_ticker_list(req.tickers)
    if not isinstance(provider, PolygonProvider):
        raise ValueError(
            "universe='sp500-pit' requires POLYGON_API_KEY; current provider "
            "is the yfinance fallback."
        )
    start = _date.fromisoformat(req.start)
    end = _date.fromisoformat(req.end)
    builder = SP500UniverseBuilder(provider=provider, supabase_client=supabase)
    window = builder.get_membership_window(start, end)
    members: set[str] = set()
    for members_list in window.values():
        members.update(members_list)
    if len(members) < 2:
        raise ValueError(
            "SP500UniverseBuilder returned fewer than 2 members for the requested window."
        )
    return tuple(sorted(members))


def _load_returns_via_provider(
    req: MarkowitzRequest,
    provider: PolygonProvider | PolygonProviderFallback,
    supabase: Any,
) -> tuple[pd.DataFrame, str]:
    """Fetch daily returns through the shared :class:`PolygonProvider`.

    Returns ``(returns_df, data_source)``. ``data_source`` is one of:
      * ``"polygon"`` when the real Polygon adapter served the data
      * ``"yfinance"`` when the fallback adapter served the data
      * ``"synthetic"`` when no real data could be loaded

    Errors degrade to the legacy yfinance / synthetic loader so the demo
    never blanks out on a transient provider hiccup.
    """
    if not req.use_real_data:
        tickers = _resolve_universe_tickers(req, provider, supabase)
        return _synthetic_returns(tickers, req.start, req.end, req.seed), "synthetic"

    try:
        tickers = _resolve_universe_tickers(req, provider, supabase)
    except ValueError:
        # sp500-pit on a fallback provider: degrade gracefully to legacy path.
        return _load_returns(req)

    try:
        start = _date.fromisoformat(req.start)
        end = _date.fromisoformat(req.end)
    except ValueError:
        return _load_returns(req)

    closes: dict[str, pd.Series] = {}
    for ticker in tickers:
        try:
            df = provider.get_eod(ticker, start, end)
        except Exception as exc:
            logger.warning(
                "provider.get_eod(%s) failed (%s); skipping",
                ticker,
                exc.__class__.__name__,
            )
            continue
        if df is None or df.empty or "Close" not in df.columns:
            continue
        closes[ticker] = df["Close"].astype(float)

    if len(closes) < 2:
        # Not enough tickers came back - fall back to the legacy loader so
        # we still produce a plot.
        return _load_returns(req)

    prices = pd.DataFrame(closes).sort_index()
    prices = prices.dropna(how="all").ffill().dropna()
    rets = prices.pct_change().dropna()
    if rets.empty:
        return _load_returns(req)

    keep = [t for t in tickers if t in rets.columns]
    if len(keep) < 2:
        return _load_returns(req)

    source: str = "polygon" if isinstance(provider, PolygonProvider) else "yfinance"
    return rets[keep], source


# ---------------------------------------------------------------------------
# Estimators & optimization
# ---------------------------------------------------------------------------


def _estimate_cov(returns: pd.DataFrame, method: CovMethod) -> pd.DataFrame:
    """Annualised covariance via the library's estimators."""
    from markowitz.estimators import LedoitWolfShrinkage, SampleCovariance

    if method == "LedoitWolf":
        est = LedoitWolfShrinkage(annualize=True, periods_per_year=_TRADING_DAYS)
    else:
        est = SampleCovariance(annualize=True, periods_per_year=_TRADING_DAYS)
    est.fit(returns)
    return pd.DataFrame(est.covariance_, index=returns.columns, columns=returns.columns)


def _estimate_mean(
    returns: pd.DataFrame,
    method: MeanMethod,
    *,
    cov: pd.DataFrame,
    risk_free_rate: float,
) -> pd.Series:
    """Annualised expected returns via the library's estimators."""
    from markowitz.estimators import (
        CAPMReturns,
        JorionBayesStein,
        SampleMean,
    )

    if method == "JorionBayesStein":
        est = JorionBayesStein(annualize=True, periods_per_year=_TRADING_DAYS)
        est.fit(returns, cov=cov.to_numpy())
    elif method == "CAPM":
        # Equal-weighted proxy market in the absence of an explicit benchmark.
        market = returns.mean(axis=1).to_numpy()
        est = CAPMReturns(
            risk_free_rate=risk_free_rate,
            annualize=True,
            periods_per_year=_TRADING_DAYS,
        )
        est.fit(returns, market=market)
    else:
        est = SampleMean(annualize=True, periods_per_year=_TRADING_DAYS)
        est.fit(returns)
    return pd.Series(est.mean_, index=returns.columns, name="mean")


def _fallback_frontier(mu: np.ndarray, cov: np.ndarray) -> pd.DataFrame:
    """Pure-numpy closed-form Merton frontier - kept for the no-cvxpy path."""
    inv = np.linalg.pinv(cov)
    ones = np.ones_like(mu)
    a = float(ones @ inv @ ones)
    b = float(ones @ inv @ mu)
    c = float(mu @ inv @ mu)
    d = a * c - b * b
    if d <= 0:
        d = 1e-9
    targets = np.linspace(mu.min(), mu.max(), _FRONTIER_POINTS)
    vols = np.sqrt((a * targets * targets - 2 * b * targets + c) / d)
    return pd.DataFrame({"volatility": vols, "expected_return": targets})


def _fallback_tangency(mu: np.ndarray, cov: np.ndarray, rf: float) -> np.ndarray:
    """Pseudo-inverse long-only tangency - graceful degradation when cvxpy is
    unavailable or the QP fails."""
    excess = mu - rf
    raw = np.linalg.pinv(cov) @ excess
    if raw.sum() == 0:
        return np.full_like(mu, 1.0 / len(mu))
    w = raw / raw.sum()
    w = np.clip(w, 0.0, None)
    s = w.sum()
    return w / s if s > 0 else np.full_like(mu, 1.0 / len(mu))


def _compute_frontier(returns: pd.DataFrame, req: MarkowitzRequest) -> dict[str, Any]:
    """Run the full estimator → AnalyticFrontier → MeanVariance pipeline."""
    from markowitz.core import AnalyticFrontier
    from markowitz.optimizer import MeanVariance

    rf = float(req.risk_free_rate)

    cov_df = _estimate_cov(returns, req.cov_method)
    mu_series = _estimate_mean(returns, req.mean_method, cov=cov_df, risk_free_rate=rf)

    mu = mu_series.to_numpy(dtype=float)
    cov = cov_df.to_numpy(dtype=float)
    tickers = list(returns.columns)

    # Frontier - closed-form Merton, falling back to pure numpy if the
    # analytic factorisation fails (singular covariance, degenerate D).
    try:
        af = AnalyticFrontier(mu, cov)
        grid = af.frontier(n_points=_FRONTIER_POINTS)
        frontier_df = pd.DataFrame(
            [(p.volatility, p.expected_return) for p in grid],
            columns=["volatility", "expected_return"],
        )
    except Exception:
        frontier_df = _fallback_frontier(mu, cov)

    # Tangency - long-only box [0, max_weight] when long_only is True; for the
    # long-short case we widen to [-max_weight, max_weight].
    if req.long_only:
        bounds: tuple[float | None, float | None] = (0.0, float(req.max_weight))
    else:
        bounds = (-float(req.max_weight), float(req.max_weight))

    try:
        opt = MeanVariance(
            mu_series,
            cov_df,
            weight_bounds=bounds,
        )
        weights_series = opt.max_sharpe(risk_free_rate=rf)
        tan_w = weights_series.reindex(tickers).fillna(0.0).to_numpy(dtype=float)
    except Exception as exc:
        logger.warning(
            "MeanVariance.max_sharpe failed (%s); using pseudo-inverse fallback",
            exc.__class__.__name__,
        )
        tan_w = _fallback_tangency(mu, cov, rf)

    # Label frontier rows (min_vol + max_sharpe) for the chart annotations.
    frontier_df["label"] = None
    min_vol_idx = int(frontier_df["volatility"].idxmin())
    vol_series = frontier_df["volatility"].replace(0, np.nan)
    sharpe_series = (frontier_df["expected_return"] - rf) / vol_series
    max_sharpe_idx = int(sharpe_series.idxmax()) if sharpe_series.notna().any() else min_vol_idx
    frontier_df.loc[min_vol_idx, "label"] = "min_vol"
    frontier_df.loc[max_sharpe_idx, "label"] = "max_sharpe"

    # Summary metrics on the realised tangency weights.
    port_ret = float(tan_w @ mu)
    port_var = float(tan_w @ cov @ tan_w)
    port_vol = float(np.sqrt(max(port_var, 0.0)))
    port_sharpe: float | None
    port_sharpe = (port_ret - rf) / port_vol if port_vol > 0 else None
    n_active = int((np.abs(tan_w) > _ACTIVE_WEIGHT_TOL).sum())

    weights_map = {t: float(w) for t, w in zip(tickers, tan_w, strict=True)}
    return {
        "frontier": frontier_df,
        "weights": weights_map,
        "tickers": tickers,
        "summary": {
            "expected_return": port_ret,
            "volatility": port_vol,
            "sharpe": port_sharpe,
            "n_assets": n_active,
        },
    }


# ---------------------------------------------------------------------------
# Plotly figures
# ---------------------------------------------------------------------------

_NAVY = "#0B2E4F"
_AMBER = "#E0A82E"
_CRIMSON = "#A11D33"


def _frontier_figure(frontier_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frontier_df["volatility"],
            y=frontier_df["expected_return"],
            mode="lines+markers",
            line={"color": _NAVY, "width": 2.5},
            marker={"size": 5, "color": _NAVY},
            name="Efficient frontier",
            hovertemplate="vol=%{x:.3%}<br>ret=%{y:.3%}<extra></extra>",
        )
    )
    marks = frontier_df.dropna(subset=["label"])
    for _, row in marks.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["volatility"]],
                y=[row["expected_return"]],
                mode="markers+text",
                text=[str(row["label"])],
                textposition="top center",
                marker={"size": 11, "color": _AMBER, "symbol": "diamond"},
                showlegend=False,
                hovertemplate=(
                    f"{row['label']}<br>vol=%{{x:.3%}}<br>ret=%{{y:.3%}}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title="Efficient frontier",
        xaxis={"title": "Volatility (annualized)", "tickformat": ".1%"},
        yaxis={"title": "Expected return (annualized)", "tickformat": ".1%"},
        height=460,
    )
    return fig


def _weights_figure(weights: dict[str, float]) -> go.Figure:
    series = pd.Series(weights).dropna().sort_values(ascending=True)
    colors = [_NAVY if v >= 0 else _CRIMSON for v in series.values]
    fig = go.Figure(
        go.Bar(
            x=series.values,
            y=series.index.astype(str),
            orientation="h",
            marker={"color": colors},
            hovertemplate="%{y}: %{x:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Tangency portfolio weights",
        xaxis={"title": "Weight", "tickformat": ".0%"},
        yaxis={"title": ""},
        height=max(280, 22 * len(series) + 120),
    )
    return fig


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


@router.post("/run", response_model=MarkowitzResponse)
def run(
    req: MarkowitzRequest,
    supabase=Depends(get_supabase),
    provider: PolygonProvider | PolygonProviderFallback = Depends(get_provider),
) -> MarkowitzResponse:
    """Execute the markowitz-optimizer compute pipeline (single-shot, sync).

    Pipeline (mirrors app/pages/1_efficient_frontier.py):
      load_returns → estimate_cov+mean → AnalyticFrontier.frontier(40)
      → MeanVariance.max_sharpe → summary metrics → plotly figures.
    """
    try:
        returns, data_source = _load_returns_via_provider(req, provider, supabase)
        result = _compute_frontier(returns, req)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("markowitz-optimizer run failed")
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(f"Markowitz optimisation failed: {exc.__class__.__name__}: {exc}"),
        ) from exc

    frontier_df: pd.DataFrame = result["frontier"]
    frontier_points = [
        FrontierPoint(
            volatility=_safe_float(float(row["volatility"])) or 0.0,
            expected_return=_safe_float(float(row["expected_return"])) or 0.0,
            label=row["label"] if isinstance(row["label"], str) else None,
        )
        for _, row in frontier_df.iterrows()
    ]

    summary_raw = result["summary"]
    summary = MarkowitzSummary(
        expected_return=_safe_float(summary_raw["expected_return"]) or 0.0,
        volatility=_safe_float(summary_raw["volatility"]) or 0.0,
        sharpe=_safe_float(summary_raw["sharpe"]),
        n_assets=int(summary_raw["n_assets"]),
    )

    weights: dict[str, float] = {k: _safe_float(v) or 0.0 for k, v in result["weights"].items()}

    frontier_fig = _frontier_figure(frontier_df)
    weights_fig = _weights_figure(weights)
    frontier_json: dict[str, Any] = json.loads(pio.to_json(frontier_fig, validate=False))
    weights_json: dict[str, Any] = json.loads(pio.to_json(weights_fig, validate=False))

    response = MarkowitzResponse(
        frontier=frontier_points,
        weights=weights,
        summary=summary,
        frontier_figure=frontier_json,
        weights_figure=weights_json,
        data_source=data_source,  # type: ignore[arg-type]
    )

    _maybe_log_run(supabase, req, response)
    return response


# ---------------------------------------------------------------------------
# Supabase helpers (best-effort)
# ---------------------------------------------------------------------------


def _maybe_log_run(supabase, req: MarkowitzRequest, resp: MarkowitzResponse) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "markowitz-optimizer",
                "params": req.model_dump(mode="json"),
                "result": {
                    "data_source": resp.data_source,
                    "summary": resp.summary.model_dump(mode="json"),
                    "n_frontier_points": len(resp.frontier),
                },
                "status": "ok",
            }
        ).execute()
    except Exception:
        logger.exception("tool_runs insert failed (non-fatal)")


def _maybe_log_failure(supabase, req: MarkowitzRequest, exc: BaseException) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "markowitz-optimizer",
                "params": req.model_dump(mode="json"),
                "result": None,
                "status": "error",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        ).execute()
    except Exception:
        logger.exception("tool_runs failure insert failed (non-fatal)")
