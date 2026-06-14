"""HRP Portfolio tool — wraps the vendored ``hrp`` library.

Endpoint:
  POST /tools/hrp-portfolio/run — run a leakage-guarded walk-forward horse race
  of Hierarchical Risk Parity (Lopez de Prado, 2016) against 1/N and IVP
  baselines, score the OOS Sharpe gap with Jobson-Korkie-Memmel + a stationary
  block-bootstrap CI + the Deflated Sharpe Ratio, derive an honest headline
  verdict, and return summary stats + five Plotly figures.

Mirrors the Streamlit demo backing the ``hrp`` library: the headline question is
"does HRP beat the DeMiguel 1/N portfolio out-of-sample, net of costs, after we
honestly deflate for the configuration grid we explored?" The verdict is a PURE
function of the inference outputs — it cannot read "HRP beats 1/N" while the
bootstrap CI straddles zero.

v1 is single-shot synchronous: a ~10-ticker / multi-year universe with a 2000-
draw bootstrap finishes comfortably under the request budget.

Caching:
  - Best-effort row in ``platform.tool_runs`` for run accounting; failures are
    logged but never propagate to the client.

Robustness:
  - The vendored library never inverts the full covariance, so HRP survives a
    singular covariance on which Markowitz CLA fails. Allocator-level failures
    inside the walk-forward degrade gracefully where the library allows.
  - Polygon failures (rate-limit / network) fall back to a seeded synthetic
    return panel, exactly like the markowitz-optimizer tool.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date as _date
from typing import Any, Literal, get_args

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from ..deps import get_supabase
from ..lib.polygon.provider import (
    PolygonProvider,
    PolygonProviderFallback,
    make_provider,
)

# Importing the vendor package side-effects ``sys.path`` so that ``import hrp``
# resolves to the vendored copy. Keep this import even if unused for
# type-checking purposes.
from ..lib import hrp as _vendor_marker  # noqa: F401

from hrp import (
    block_bootstrap_sharpe_gap,
    deflated_sharpe_ratio,
    dendrogram_figure,
    derive_verdict,
    hrp_allocate,
    ivp_weights,
    jobson_korkie_memmel,
    ledoit_wolf_cov,
    naive_weights,
    quasidiag_heatmap_figure,
    sharpe_gap_bootstrap_figure,
    walk_forward_backtest,
    weights_figure,
)
from hrp.backtest.stats import annualized_vol, sharpe_ratio
from hrp.estimators.covariance import oas_cov, sample_cov
from hrp.estimators.rmt import marchenko_pastur_clip

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools/hrp-portfolio", tags=["hrp-portfolio"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

Linkage = Literal["single", "ward", "complete", "average"]
Covariance = Literal["sample", "ledoit_wolf", "oas"]
Rebalance = Literal["monthly", "quarterly"]
DataSource = Literal["polygon", "yfinance", "synthetic"]

_TRADING_DAYS = 252
_MAX_BOOTSTRAP = 5_000
_ACTIVE_WEIGHT_TOL = 1e-4

#: The baselines we know how to allocate without an explicit ``mu`` estimate.
#: ``min_var`` is included opportunistically (it only needs the covariance);
#: ``1/N`` and ``IVP`` are always available.
_SUPPORTED_BASELINES = ("1/N", "IVP", "min_var")


class HRPRequest(BaseModel):
    """Wire contract for an HRP horse-race run.

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
    linkage: Linkage = Field("single")
    covariance: Covariance = Field("ledoit_wolf")
    rmt_denoise: bool = Field(False)
    rebalance: Rebalance = Field("monthly")
    cost_bps: float = Field(10.0, ge=0.0)
    lookback_window: int = Field(
        252,
        ge=2,
        description=(
            "In-sample window length in trading days. Enforced at runtime to be "
            ">= n_assets + 1 for a full-rank in-sample covariance."
        ),
    )
    include_baselines: list[str] = Field(default_factory=lambda: ["1/N", "IVP", "min_var"])
    n_bootstrap: int = Field(
        2000,
        ge=1,
        description="Block-bootstrap resamples for the Sharpe-gap CI (capped).",
    )
    data_source_pref: Literal["polygon", "yfinance", "synthetic", "auto"] = Field("auto")
    seed: int = Field(7, ge=0, le=999_999)

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

    @field_validator("include_baselines")
    @classmethod
    def _normalise_baselines(cls, v: list[str]) -> list[str]:
        seen: dict[str, None] = {}
        for raw in v:
            name = raw.strip()
            if name in _SUPPORTED_BASELINES:
                seen.setdefault(name, None)
        # Always guarantee the headline 1/N and IVP baselines are present.
        for required in ("1/N", "IVP"):
            seen.setdefault(required, None)
        return list(seen)

    @field_validator("n_bootstrap")
    @classmethod
    def _cap_bootstrap(cls, v: int) -> int:
        return min(int(v), _MAX_BOOTSTRAP)


class HRPResponse(BaseModel):
    summary: dict[str, Any] = Field(
        ...,
        description=(
            "Scalar summary: hrp_oos_sharpe, naive_1n_oos_sharpe, ivp_oos_sharpe, "
            "hrp_oos_vol, naive_1n_oos_vol, sharpe_gap_hrp_vs_1n, jkm_pvalue, "
            "deflated_sharpe, ci_low, ci_high, n_effective_trials, hrp_turnover, "
            "naive_1n_turnover, n_assets, n_rebalances, headline_verdict."
        ),
    )
    dendrogram_figure: dict[str, Any] = Field(..., description="Plotly figure JSON: {data, layout}.")
    quasidiag_heatmap_figure: dict[str, Any] = Field(
        ..., description="Plotly figure JSON: {data, layout}."
    )
    weights_figure: dict[str, Any] = Field(..., description="Plotly figure JSON: {data, layout}.")
    oos_equity_figure: dict[str, Any] = Field(..., description="Plotly figure JSON: {data, layout}.")
    sharpe_gap_bootstrap_figure: dict[str, Any] = Field(
        ..., description="Plotly figure JSON: {data, layout}."
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
# Data loading — mirrors app/pages/1_efficient_frontier.py (markowitz)
# ---------------------------------------------------------------------------


def _synthetic_returns(
    tickers: tuple[str, ...], start: str, end: str, seed: int
) -> pd.DataFrame:
    """Deterministic synthetic returns; identical math to the markowitz tool."""
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


def _load_returns_synthetic(req: HRPRequest) -> tuple[pd.DataFrame, str]:
    tickers = _parse_ticker_list(req.tickers)
    return _synthetic_returns(tickers, req.start, req.end, req.seed), "synthetic"


def _load_returns_via_provider(
    req: HRPRequest,
    provider: PolygonProvider | PolygonProviderFallback,
) -> tuple[pd.DataFrame, str]:
    """Fetch daily returns through the shared :class:`PolygonProvider`.

    Returns ``(returns_df, data_source)``. ``data_source`` is one of:
      * ``"polygon"`` when the real Polygon adapter served the data
      * ``"yfinance"`` when the fallback adapter served the data
      * ``"synthetic"`` when no real data could be loaded

    Errors degrade to the seeded synthetic panel so the demo never blanks out
    on a transient provider hiccup. Honours an explicit
    ``data_source_pref="synthetic"`` short-circuit.
    """
    if req.data_source_pref == "synthetic":
        return _load_returns_synthetic(req)

    tickers = _parse_ticker_list(req.tickers)

    try:
        start = _date.fromisoformat(req.start)
        end = _date.fromisoformat(req.end)
    except ValueError:
        return _load_returns_synthetic(req)

    closes: dict[str, pd.Series] = {}
    for ticker in tickers:
        try:
            df = provider.get_eod(ticker, start, end)
        except Exception as exc:  # noqa: BLE001 — degrade per-ticker
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
        # Not enough tickers came back — fall back to synthetic so we still
        # produce a result.
        return _load_returns_synthetic(req)

    prices = pd.DataFrame(closes).sort_index()
    prices = prices.dropna(how="all").ffill().dropna()
    rets = prices.pct_change().dropna()
    if rets.empty:
        return _load_returns_synthetic(req)

    keep = [t for t in tickers if t in rets.columns]
    if len(keep) < 2:
        return _load_returns_synthetic(req)

    source: str = "polygon" if isinstance(provider, PolygonProvider) else "yfinance"
    return rets[keep], source


# ---------------------------------------------------------------------------
# Estimators & allocators
# ---------------------------------------------------------------------------


def _estimate_cov(returns: pd.DataFrame, method: Covariance, *, rmt_denoise: bool) -> pd.DataFrame:
    """Covariance estimate with an optional Marchenko-Pastur denoise step."""
    if method == "sample":
        cov = sample_cov(returns)
    elif method == "oas":
        cov = oas_cov(returns)
    else:
        cov = ledoit_wolf_cov(returns)

    if rmt_denoise:
        cov = marchenko_pastur_clip(
            cov, n_obs=int(returns.shape[0]), n_assets=int(returns.shape[1])
        )
    return cov


def _build_allocators(req: HRPRequest):
    """Return ``{name: allocator}`` mapping for HRP + the requested baselines.

    Each allocator is ``Callable[[in_sample_returns_df], weights_series]`` and
    re-estimates the covariance on the window it is handed (the shared estimator
    is the request's ``covariance`` + optional ``rmt_denoise``), so every
    allocator is fed an identically-conditioned matrix per window.
    """

    def _window_cov(window: pd.DataFrame) -> pd.DataFrame:
        return _estimate_cov(window, req.covariance, rmt_denoise=req.rmt_denoise)

    allocators: dict[str, Any] = {
        "HRP": lambda window: hrp_allocate(
            window,
            cov=_window_cov(window),
            linkage_method=req.linkage,
        ).weights,
        "1/N": lambda window: naive_weights(list(window.columns)),
        "IVP": lambda window: ivp_weights(_window_cov(window)),
    }

    if "min_var" in req.include_baselines:
        # min_var only needs the covariance (no mu), so it is trivially available
        # via the vendored markowitz adapter. cvxpy is imported lazily inside the
        # adapter only when the long-only QP is actually needed.
        try:
            from hrp import min_var_weights

            allocators["min_var"] = lambda window: min_var_weights(_window_cov(window))
        except Exception:  # noqa: BLE001 — keep 1/N + IVP at minimum
            logger.warning("min_var_weights unavailable; skipping min_var baseline")

    return allocators


# ---------------------------------------------------------------------------
# Backtest + inference pipeline
# ---------------------------------------------------------------------------


def _equity_curve(returns: pd.Series) -> pd.Series:
    """Cumulative wealth curve from a per-period net return series."""
    return (1.0 + returns.astype("float64")).cumprod()


def _run_pipeline(returns: pd.DataFrame, req: HRPRequest) -> dict[str, Any]:
    """Run the full HRP horse race + inference and assemble figure inputs."""
    from hrp._exceptions import InsufficientDataError, ValidationError

    assets = list(returns.columns)
    n_assets = len(assets)

    # Enforce the library's full-rank in-sample requirement honestly: the
    # lookback window must hold at least n_assets + 1 observations.
    lookback = max(int(req.lookback_window), n_assets + 1)

    allocators = _build_allocators(req)

    # --- Walk-forward backtest for HRP + each baseline --------------------
    results: dict[str, Any] = {}
    for name, allocator in allocators.items():
        try:
            results[name] = walk_forward_backtest(
                returns,
                allocator,
                lookback_window=lookback,
                rebalance=req.rebalance,
                cost_bps=float(req.cost_bps),
                embargo=1,
                purge=1,
                anchored=False,
            )
        except (InsufficientDataError, ValidationError) as exc:
            # Surface insufficient-data / validation problems as a 422 below.
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"HRP backtest could not run: {exc.__class__.__name__}: {exc}",
            ) from exc

    hrp_result = results["HRP"]
    naive_result = results["1/N"]
    ivp_result = results.get("IVP")

    hrp_oos = hrp_result.oos_returns
    naive_oos = naive_result.oos_returns

    # --- OOS Sharpe / vol / turnover -------------------------------------
    hrp_sharpe = sharpe_ratio(hrp_oos, periods_per_year=_TRADING_DAYS)
    naive_sharpe = sharpe_ratio(naive_oos, periods_per_year=_TRADING_DAYS)
    ivp_sharpe = (
        sharpe_ratio(ivp_result.oos_returns, periods_per_year=_TRADING_DAYS)
        if ivp_result is not None
        else None
    )

    hrp_vol = annualized_vol(hrp_oos, periods_per_year=_TRADING_DAYS)
    naive_vol = annualized_vol(naive_oos, periods_per_year=_TRADING_DAYS)

    hrp_turnover = float(hrp_result.turnover.mean()) if len(hrp_result.turnover) else 0.0
    naive_turnover = float(naive_result.turnover.mean()) if len(naive_result.turnover) else 0.0

    # --- Sharpe-gap inference: JKM + bootstrap CI ------------------------
    jkm_pvalue = jobson_korkie_memmel(hrp_oos, naive_oos)
    comparison = block_bootstrap_sharpe_gap(
        hrp_oos,
        naive_oos,
        n_bootstrap=int(req.n_bootstrap),
        seed=int(req.seed),
    )

    # --- Deflated Sharpe Ratio (honest n_trials) -------------------------
    # n_trials counts the FULL explored configuration grid: #allocators we ran
    # times #linkages available (the linkage axis we honestly searched over).
    n_linkages = max(1, len(get_args(Linkage)))
    n_effective_trials = max(1, len(allocators) * n_linkages)

    # Per-observation (non-annualized) Sharpe of the selected HRP strategy.
    hrp_excess = hrp_oos.astype("float64")
    hrp_std = float(hrp_excess.std(ddof=1))
    hrp_per_obs_sharpe = (
        float(hrp_excess.mean()) / hrp_std if hrp_std > 0 and math.isfinite(hrp_std) else 0.0
    )
    skew = float(hrp_excess.skew()) if len(hrp_excess) > 2 else 0.0
    # FULL (non-excess) kurtosis: pandas .kurt() returns EXCESS kurtosis, so + 3.
    kurt = (float(hrp_excess.kurt()) + 3.0) if len(hrp_excess) > 3 else 3.0
    if not math.isfinite(skew):
        skew = 0.0
    if not math.isfinite(kurt):
        kurt = 3.0

    try:
        deflated = deflated_sharpe_ratio(
            hrp_per_obs_sharpe,
            n_obs=int(len(hrp_oos)),
            n_trials=int(n_effective_trials),
            variance_of_trial_sharpes=1.0,
            skew=skew,
            kurtosis=kurt,
        )
    except (InsufficientDataError, ValidationError):
        deflated = float("nan")

    # --- Headline verdict (pure function of the evidence) ----------------
    verdict = derive_verdict(
        jkm_pvalue,
        deflated if math.isfinite(deflated) else 0.0,
        comparison.ci_low,
        comparison.ci_high,
    )

    summary: dict[str, Any] = {
        "hrp_oos_sharpe": _safe_float(hrp_sharpe),
        "naive_1n_oos_sharpe": _safe_float(naive_sharpe),
        "ivp_oos_sharpe": _safe_float(ivp_sharpe),
        "hrp_oos_vol": _safe_float(hrp_vol),
        "naive_1n_oos_vol": _safe_float(naive_vol),
        "sharpe_gap_hrp_vs_1n": _safe_float(comparison.sharpe_gap),
        "jkm_pvalue": _safe_float(jkm_pvalue),
        "deflated_sharpe": _safe_float(deflated),
        "ci_low": _safe_float(comparison.ci_low),
        "ci_high": _safe_float(comparison.ci_high),
        "n_effective_trials": int(n_effective_trials),
        "hrp_turnover": _safe_float(hrp_turnover),
        "naive_1n_turnover": _safe_float(naive_turnover),
        "n_assets": int(n_assets),
        "n_rebalances": int(hrp_result.n_rebalances),
        "headline_verdict": verdict.value,
    }

    return {
        "summary": summary,
        "results": results,
        "comparison": comparison,
        "assets": assets,
        "lookback": lookback,
    }


# ---------------------------------------------------------------------------
# Plotly figures
# ---------------------------------------------------------------------------

_PALETTE = ("#0B2E4F", "#E0A82E", "#A11D33", "#2E7D5B", "#6A4C93", "#8A8D91")


def _full_window_hrp(returns: pd.DataFrame, req: HRPRequest):
    """Allocate HRP once over the FULL panel for the dendrogram / heatmap / weights
    figures (which need a single concrete tree + weight vector)."""
    cov = _estimate_cov(returns, req.covariance, rmt_denoise=req.rmt_denoise)
    return hrp_allocate(returns, cov=cov, linkage_method=req.linkage), cov


def _empty_figure(title: str) -> dict[str, Any]:
    fig = go.Figure()
    fig.update_layout(title=title, height=360)
    return json.loads(pio.to_json(fig, validate=False))


def _oos_equity_figure(results: dict[str, Any]) -> dict[str, Any]:
    """Overlay each allocator's net OOS equity curve via a constructed go.Figure."""
    fig = go.Figure()
    for i, (name, res) in enumerate(results.items()):
        equity = _equity_curve(res.oos_returns)
        fig.add_trace(
            go.Scatter(
                x=[
                    v.isoformat() if hasattr(v, "isoformat") else str(v)
                    for v in equity.index
                ],
                y=[_safe_float(v) for v in equity.to_numpy()],
                mode="lines",
                name=name,
                line={"color": _PALETTE[i % len(_PALETTE)], "width": 2.0},
            )
        )
    fig.update_layout(
        title="Out-of-sample equity curves (net of costs)",
        xaxis={"title": "Date"},
        yaxis={"title": "Cumulative wealth"},
        legend={"orientation": "h"},
        height=440,
    )
    return json.loads(pio.to_json(fig, validate=False))


def _correlation_from_cov(cov: pd.DataFrame) -> pd.DataFrame:
    std = np.sqrt(np.diag(cov.to_numpy(dtype="float64")))
    safe = np.where(std > 0.0, std, 1.0)
    inv = 1.0 / safe
    corr = cov.to_numpy(dtype="float64") * np.outer(inv, inv)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)


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


@router.post("/run", response_model=HRPResponse)
def run(
    req: HRPRequest,
    supabase=Depends(get_supabase),
    provider: PolygonProvider | PolygonProviderFallback = Depends(get_provider),
) -> HRPResponse:
    """Execute the hrp-portfolio compute pipeline (single-shot, sync).

    Pipeline:
      load_returns → estimate_cov → walk_forward_backtest(HRP + baselines)
      → OOS Sharpe/vol/turnover → JKM p-value + bootstrap CI + DSR
      → derive_verdict → five Plotly figures.
    """
    try:
        returns, data_source = _load_returns_via_provider(req, provider)
    except Exception as exc:
        logger.exception("hrp-portfolio data load failed")
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"HRP data load failed: {exc.__class__.__name__}: {exc}",
        ) from exc

    try:
        pipeline = _run_pipeline(returns, req)
    except HTTPException:
        raise
    except Exception as exc:
        from hrp._exceptions import InsufficientDataError, ValidationError

        logger.exception("hrp-portfolio run failed")
        _maybe_log_failure(supabase, req, exc)
        if isinstance(exc, (InsufficientDataError, ValidationError)):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"HRP run rejected: {exc.__class__.__name__}: {exc}",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"HRP run failed: {exc.__class__.__name__}: {exc}",
        ) from exc

    summary = pipeline["summary"]
    results = pipeline["results"]
    comparison = pipeline["comparison"]

    # --- Figures ---------------------------------------------------------
    try:
        hrp_full, cov_full = _full_window_hrp(returns, req)
        corr_full = _correlation_from_cov(cov_full)
        dendro_json = dendrogram_figure(
            np.asarray(hrp_full.link), [str(a) for a in returns.columns]
        )
        heatmap_json = quasidiag_heatmap_figure(corr_full, list(hrp_full.quasidiag_order))
        weights_json = weights_figure(hrp_full.weights, title="HRP portfolio weights")
    except Exception:  # noqa: BLE001 — figures are best-effort, never fatal
        logger.exception("hrp-portfolio tree/weights figures failed (non-fatal)")
        dendro_json = _empty_figure("HRP dendrogram")
        heatmap_json = _empty_figure("Quasi-diagonalized correlation")
        weights_json = _empty_figure("HRP portfolio weights")

    try:
        equity_json = _oos_equity_figure(results)
    except Exception:  # noqa: BLE001
        logger.exception("hrp-portfolio equity figure failed (non-fatal)")
        equity_json = _empty_figure("Out-of-sample equity curves")

    try:
        # Reconstruct the bootstrap gap distribution for the histogram from the
        # same seeded resample the inference layer used, so the figure matches
        # the reported CI exactly.
        gap_samples = _bootstrap_gap_samples(
            results["HRP"].oos_returns,
            results["1/N"].oos_returns,
            n_bootstrap=int(req.n_bootstrap),
            seed=int(req.seed),
        )
        bootstrap_json = sharpe_gap_bootstrap_figure(
            gap_samples,
            ci_low=comparison.ci_low,
            ci_high=comparison.ci_high,
        )
    except Exception:  # noqa: BLE001
        logger.exception("hrp-portfolio bootstrap figure failed (non-fatal)")
        bootstrap_json = _empty_figure("Bootstrap Sharpe-gap distribution")

    response = HRPResponse(
        summary=summary,
        dendrogram_figure=dendro_json,
        quasidiag_heatmap_figure=heatmap_json,
        weights_figure=weights_json,
        oos_equity_figure=equity_json,
        sharpe_gap_bootstrap_figure=bootstrap_json,
        data_source=data_source,  # type: ignore[arg-type]
    )

    _maybe_log_run(supabase, req, response)
    return response


def _bootstrap_gap_samples(
    returns_a: pd.Series,
    returns_b: pd.Series,
    *,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    """Reproduce the stationary-block-bootstrap Sharpe-gap samples for plotting.

    Mirrors :func:`hrp.evaluation.comparison.block_bootstrap_sharpe_gap` (same
    seeded PCG64 generator and geometric block scheme) so the histogram is
    consistent with the reported CI.
    """
    from hrp._constants import PERIODS_PER_YEAR
    from hrp._rng import make_rng

    a = returns_a.astype("float64")
    b = returns_b.astype("float64")
    common = a.index.intersection(b.index)
    if len(common) > 0:
        common = common.sort_values()
        a = a.reindex(common)
        b = b.reindex(common)
    arr_a = a.to_numpy(dtype="float64")
    arr_b = b.to_numpy(dtype="float64")
    t = arr_a.shape[0]
    if t < 3:
        return np.empty(0, dtype="float64")

    ann = math.sqrt(PERIODS_PER_YEAR)

    def _ps(x: np.ndarray) -> float:
        std = float(x.std(ddof=1))
        if std == 0.0 or not math.isfinite(std):
            return 0.0
        return float(x.mean()) / std

    eff_block = max(1, int(round(t ** (1.0 / 3.0))))
    p_restart = 1.0 / eff_block
    rng = make_rng(seed)
    gaps = np.empty(int(n_bootstrap), dtype="float64")
    for bi in range(int(n_bootstrap)):
        idx = np.empty(t, dtype=np.intp)
        pos = int(rng.integers(0, t))
        idx[0] = pos
        restarts = rng.random(t)
        steps = rng.integers(0, t, size=t)
        for i in range(1, t):
            if restarts[i] < p_restart:
                pos = int(steps[i])
            else:
                pos = pos + 1
                if pos >= t:
                    pos = 0
            idx[i] = pos
        gaps[bi] = (_ps(arr_a[idx]) - _ps(arr_b[idx])) * ann
    return gaps


# ---------------------------------------------------------------------------
# Supabase helpers (best-effort)
# ---------------------------------------------------------------------------


def _maybe_log_run(supabase, req: HRPRequest, resp: HRPResponse) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "hrp-portfolio",
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


def _maybe_log_failure(supabase, req: HRPRequest, exc: BaseException) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "hrp-portfolio",
                "params": req.model_dump(mode="json"),
                "result": None,
                "status": "error",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        ).execute()
    except Exception:
        logger.exception("tool_runs failure insert failed (non-fatal)")
