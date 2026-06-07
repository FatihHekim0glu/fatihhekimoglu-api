"""MA Crossover Backtest tool - wraps ma-crossover-backtest's src/ unchanged.

Endpoint:
  POST /tools/ma-crossover-backtest/run

Pipeline mirrors ma-crossover-backtest/streamlit_app.py default (live) view:
  load_close -> run_backtest -> run_buy_and_hold -> metrics + comparison -> figures.

The Streamlit app's heavier gated tabs (parameter sweep + walk-forward) are
deliberately NOT exposed here - the recon flagged them as button-gated and
multi-second; v1 is single-shot synchronous.
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import date
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from ..deps import get_supabase

# Importing the package triggers __init__.py, which aliases the legacy
# `ma_backtester` namespace into sys.modules so the vendored sources resolve.
from ..lib import ma_crossover_backtest as _vendor  # noqa: F401
from ..lib.ma_crossover_backtest import backtester, benchmark, config, costs, data, metrics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools/ma-crossover-backtest", tags=["ma-crossover-backtest"])


# ---------------------------------------------------------------------------
# Constants & validation
# ---------------------------------------------------------------------------

# Mirrors the data.py vendored regex (A-Z 0-9 . - ^, up to 15 chars).
_TICKER_RE = re.compile(r"^[A-Z0-9.\-^]{1,15}$")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class MaCrossoverRequest(BaseModel):
    """Input contract for POST /tools/ma-crossover-backtest/run."""

    ticker: str = Field("SPY", max_length=15)
    start_date: date = Field(date(2010, 1, 1))
    end_date: date = Field(date(2024, 12, 31))
    fast_window: int = Field(50, ge=5, le=100)
    slow_window: int = Field(200, ge=20, le=300)
    cost_bps: float = Field(5.0, ge=0.0, le=50.0)

    @field_validator("ticker")
    @classmethod
    def _ticker_shape(cls, v: str) -> str:
        v = v.strip().upper()
        if not v:
            raise ValueError("ticker cannot be empty")
        if not _TICKER_RE.match(v):
            raise ValueError(
                "ticker must contain only A-Z, 0-9, '.', '-', '^' (1-15 chars); "
                "examples: SPY, BRK-B, ^GSPC"
            )
        return v

    @field_validator("end_date")
    @classmethod
    def _end_after_start(cls, v: date, info: Any) -> date:
        start = info.data.get("start_date")
        if start is not None and v <= start:
            raise ValueError("end_date must be strictly after start_date")
        return v

    @field_validator("slow_window")
    @classmethod
    def _slow_gt_fast(cls, v: int, info: Any) -> int:
        fast = info.data.get("fast_window")
        if fast is not None and v <= fast:
            raise ValueError(
                f"slow_window ({v}) must be strictly greater than fast_window ({fast})"
            )
        return v


class MetricsRow(BaseModel):
    """One row in the headline metrics table - strategy vs buy & hold."""

    name: str
    strategy: float | None
    buy_and_hold: float | None
    delta: float | None


class BenchmarkStats(BaseModel):
    """Full BenchmarkComparison surface from benchmark.compare_strategies()."""

    alpha_annual: float | None
    alpha_t_stat: float | None
    alpha_p_value: float | None
    beta: float | None
    beta_se: float | None
    information_ratio: float | None
    tracking_error_annual: float | None
    active_return_annual: float | None
    sharpe_diff: float | None
    sharpe_diff_p_value: float | None
    n_observations: int
    hac_lags: int


class MaCrossoverSummary(BaseModel):
    """Top-of-page summary numbers."""

    ticker: str
    start: date
    end: date
    trading_days: int
    fast_window: int
    slow_window: int
    cost_bps: float
    n_trades: int


class MaCrossoverResponse(BaseModel):
    """Output contract for POST /tools/ma-crossover-backtest/run."""

    summary: MaCrossoverSummary
    metrics: list[MetricsRow]
    benchmark: BenchmarkStats
    equity_curve: dict[str, Any] = Field(..., description="Plotly figure JSON: log-scale equity.")
    underwater_drawdown: dict[str, Any] = Field(..., description="Plotly figure JSON: drawdown.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value: float | int | None) -> float | None:
    """Convert NaN / +-Inf to JSON-safe None so the React renderer shows em-dashes."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _delta(strat: float | None, bh: float | None) -> float | None:
    """Strategy minus buy-and-hold, JSON-safe."""
    if strat is None or bh is None:
        return None
    d = strat - bh
    if math.isnan(d) or math.isinf(d):
        return None
    return d


def _drawdown_series(equity: pd.Series) -> pd.Series:
    """Underwater curve - equity / cummax - 1."""
    peak = equity.cummax()
    return equity / peak - 1.0


def _build_equity_figure(
    strat_equity: pd.Series,
    bh_equity: pd.Series,
    ticker: str,
) -> go.Figure:
    """Log-scale equity curves for strategy vs buy-and-hold."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=strat_equity.index,
            y=strat_equity.values,
            mode="lines",
            name="Strategy",
            line=dict(color="#14b8a6", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bh_equity.index,
            y=bh_equity.values,
            mode="lines",
            name="Buy & Hold",
            line=dict(color="#94a3b8", width=2, dash="dash"),
        )
    )
    fig.update_layout(
        title=f"{ticker} - Equity Curve (log scale)",
        yaxis=dict(type="log", title="Equity"),
        xaxis=dict(title="Date"),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _build_drawdown_figure(
    strat_equity: pd.Series,
    bh_equity: pd.Series,
    ticker: str,
) -> go.Figure:
    """Underwater drawdown curves for strategy vs buy-and-hold."""
    strat_dd = _drawdown_series(strat_equity)
    bh_dd = _drawdown_series(bh_equity)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=strat_dd.index,
            y=strat_dd.values,
            mode="lines",
            name="Strategy",
            line=dict(color="#14b8a6", width=2),
            fill="tozeroy",
            fillcolor="rgba(20, 184, 166, 0.15)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bh_dd.index,
            y=bh_dd.values,
            mode="lines",
            name="Buy & Hold",
            line=dict(color="#94a3b8", width=2, dash="dash"),
        )
    )
    fig.update_layout(
        title=f"{ticker} - Underwater Drawdown",
        yaxis=dict(title="Drawdown", tickformat=".0%"),
        xaxis=dict(title="Date"),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/run", response_model=MaCrossoverResponse)
def run(
    req: MaCrossoverRequest,
    supabase=Depends(get_supabase),
) -> MaCrossoverResponse:
    """Execute the headline ma-crossover-backtest compute pipeline.

    Pipeline mirrors streamlit_app.py default view (no buttons required):
      data.load_close -> run_backtest + run_buy_and_hold (with FixedBpsCost)
        -> metrics on both -> benchmark.compare_strategies
        -> serialise equity / drawdown figures, metrics table, stats.
    """
    # Build configs - dataclasses raise ValueError on invalid pairings.
    try:
        strat_cfg = config.StrategyConfig(
            fast_window=req.fast_window,
            slow_window=req.slow_window,
        )
        cost_cfg = config.CostConfig(per_side_bps=req.cost_bps)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    cost_model = costs.FixedBpsCost(cost_cfg)

    # Load prices - parquet cache lives in cwd/data/cache/ohlcv/ by default.
    # That's acceptable for the API process; a deeper integration with the
    # platform-level ohlcv_cache table is a follow-up.
    try:
        close = data.load_close(req.ticker, start=req.start_date, end=req.end_date)
    except data.DataQualityError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{req.ticker}: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("unexpected error loading %s", req.ticker)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Upstream fetch failed for {req.ticker}",
        ) from exc

    if close.empty:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"No usable close prices for {req.ticker} over the requested range",
        )

    # Run both engines.
    try:
        strat_result = backtester.run_backtest(
            close=close,
            strategy_config=strat_cfg,
            cost_model=cost_model,
        )
        bh_result = backtester.run_buy_and_hold(
            close=close,
            cost_model=cost_model,
        )
    except Exception as exc:
        logger.exception("backtest engine failed for %s", req.ticker)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Backtest engine failed: {exc}",
        ) from exc

    # Headline metrics (CAGR, vol, Sharpe, Sortino, MaxDD) for both legs.
    def _metrics(equity: pd.Series, daily_returns: pd.Series) -> dict[str, float | None]:
        return {
            "CAGR": _safe_float(metrics.cagr(equity)),
            "Annual vol": _safe_float(metrics.annualised_volatility(daily_returns)),
            "Sharpe": _safe_float(metrics.sharpe_ratio(daily_returns)),
            "Sortino": _safe_float(metrics.sortino_ratio(daily_returns)),
            "Max drawdown": _safe_float(metrics.max_drawdown(equity)),
        }

    strat_m = _metrics(strat_result.equity, strat_result.daily_returns)
    bh_m = _metrics(bh_result.equity, bh_result.daily_returns)

    metrics_rows: list[MetricsRow] = [
        MetricsRow(name=name, strategy=strat_m[name], buy_and_hold=bh_m[name], delta=_delta(strat_m[name], bh_m[name]))
        for name in ("CAGR", "Annual vol", "Sharpe", "Sortino", "Max drawdown")
    ]

    # Statistical comparison - HAC alpha, IR, Memmel Sharpe diff.
    try:
        cmp = benchmark.compare_strategies(
            strategy_returns=strat_result.daily_returns,
            benchmark_returns=bh_result.daily_returns,
        )
        stats = BenchmarkStats(
            alpha_annual=_safe_float(cmp.alpha_annual),
            alpha_t_stat=_safe_float(cmp.alpha_t_stat),
            alpha_p_value=_safe_float(cmp.alpha_p_value),
            beta=_safe_float(cmp.beta),
            beta_se=_safe_float(cmp.beta_se),
            information_ratio=_safe_float(cmp.information_ratio),
            tracking_error_annual=_safe_float(cmp.tracking_error_annual),
            active_return_annual=_safe_float(cmp.active_return_annual),
            sharpe_diff=_safe_float(cmp.sharpe_diff),
            sharpe_diff_p_value=_safe_float(cmp.sharpe_diff_p_value),
            n_observations=int(cmp.n_observations),
            hac_lags=int(cmp.hac_lags),
        )
    except ValueError as exc:
        # Fewer than 30 aligned obs - degrade gracefully rather than 500.
        stats = BenchmarkStats(
            alpha_annual=None,
            alpha_t_stat=None,
            alpha_p_value=None,
            beta=None,
            beta_se=None,
            information_ratio=None,
            tracking_error_annual=None,
            active_return_annual=None,
            sharpe_diff=None,
            sharpe_diff_p_value=None,
            n_observations=0,
            hac_lags=0,
        )
        logger.info("benchmark comparison skipped (insufficient obs): %s", exc)

    # Figures.
    eq_fig = _build_equity_figure(strat_result.equity, bh_result.equity, req.ticker)
    dd_fig = _build_drawdown_figure(strat_result.equity, bh_result.equity, req.ticker)
    equity_json: dict[str, Any] = json.loads(pio.to_json(eq_fig, validate=False))
    drawdown_json: dict[str, Any] = json.loads(pio.to_json(dd_fig, validate=False))

    response = MaCrossoverResponse(
        summary=MaCrossoverSummary(
            ticker=req.ticker,
            start=close.index[0].date() if hasattr(close.index[0], "date") else req.start_date,
            end=close.index[-1].date() if hasattr(close.index[-1], "date") else req.end_date,
            trading_days=int(len(close)),
            fast_window=req.fast_window,
            slow_window=req.slow_window,
            cost_bps=req.cost_bps,
            n_trades=int(len(strat_result.trades)),
        ),
        metrics=metrics_rows,
        benchmark=stats,
        equity_curve=equity_json,
        underwater_drawdown=drawdown_json,
    )

    _maybe_log_run(supabase, req, response)
    return response


# ---------------------------------------------------------------------------
# Supabase logging (best-effort)
# ---------------------------------------------------------------------------


def _maybe_log_run(
    supabase, req: MaCrossoverRequest, resp: MaCrossoverResponse
) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "ma-crossover-backtest",
                "params": req.model_dump(mode="json"),
                "result": {
                    "summary": resp.summary.model_dump(mode="json"),
                    "metrics": [m.model_dump(mode="json") for m in resp.metrics],
                    "benchmark": resp.benchmark.model_dump(mode="json"),
                },
                "status": "ok",
            }
        ).execute()
    except Exception:
        logger.exception("tool_runs insert failed (non-fatal)")
