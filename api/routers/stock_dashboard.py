"""Stock Dashboard tool — wraps stock-dashboard's src/ unchanged.

Endpoints:
  GET  /tools/stock-dashboard/anchor — NYSE-today + market state for the React form
  POST /tools/stock-dashboard/run    — fetch + indicators + stats + Plotly figure JSON

Caching:
  - Per-(symbol, date) OHLCV rows in platform.ohlcv_cache (shared across tools).
  - Intraday bypass preserved verbatim from the source: when the requested range
    includes today and NYSE is still open, skip both read and write so the user
    sees the still-forming bar live.
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import date, datetime, timedelta
from typing import Any, Literal

import pandas as pd
import plotly.io as pio
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from ..deps import get_supabase
from ..lib.stock_dashboard import charts, data, indicators, stats
from ..lib.stock_dashboard.data import DataFetchError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools/stock-dashboard", tags=["stock-dashboard"])


# ---------------------------------------------------------------------------
# Constants & validation
# ---------------------------------------------------------------------------

_TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,15}$")

# Mirrors stock-dashboard/app.py INDICATOR_OPTIONS exactly.
IndicatorLabel = Literal[
    "SMA 20",
    "SMA 50",
    "EMA 12",
    "EMA 26",
    "Bollinger Bands",
    "RSI",
    "MACD",
]

# Label → tag protocol from stock-dashboard/app.py:121-133. Tags drive the chart
# column names and figure trace selection.
_LABEL_TO_TAG: dict[str, str] = {
    "SMA 20": "sma_20",
    "SMA 50": "sma_50",
    "EMA 12": "ema_12",
    "EMA 26": "ema_26",
    "Bollinger Bands": "bb",
    "RSI": "rsi",
    "MACD": "macd",
}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class AnchorResponse(BaseModel):
    """Used by the React form to anchor the date picker to NYSE-local time."""

    today_nyse: date = Field(..., description="NYSE-local today (ISO).")
    min_date: date = Field(date(1970, 1, 1))
    default_start: date = Field(..., description="today_nyse − 5y, the source's default.")
    market_is_open: bool = Field(..., description="True before today's NYSE close.")
    trading_day_stamp: str = Field(..., description="Cache-key stamp; flips at NYSE close.")


class StockDashboardRequest(BaseModel):
    ticker: str = Field(..., max_length=15)
    start_date: date
    end_date: date
    indicators: list[IndicatorLabel] = Field(default_factory=list)
    candlestick: bool = True
    rf_annual: float = Field(0.0, ge=-0.05, le=0.25)

    @field_validator("ticker")
    @classmethod
    def _ticker_shape(cls, v: str) -> str:
        v = v.strip().upper()
        if not v:
            raise ValueError("ticker cannot be empty")
        if not _TICKER_RE.match(v):
            raise ValueError(
                "ticker must contain only A-Z, 0-9, '.', '-' (1-15 chars); examples: AAPL, BRK-B, BP.L"
            )
        return v

    @field_validator("end_date")
    @classmethod
    def _end_after_start(cls, v: date, info: Any) -> date:
        start = info.data.get("start_date")
        if start is not None and v <= start:
            raise ValueError("end_date must be strictly after start_date")
        return v


class StockDashboardSummary(BaseModel):
    start: date
    end: date
    trading_days: int
    cumulative_return: float | None
    annualized_return: float | None
    annualized_volatility: float | None
    sharpe: float | None
    max_drawdown: float | None
    best_day: float | None
    worst_day: float | None


class StockDashboardResponse(BaseModel):
    figure: dict[str, Any] = Field(..., description="Plotly figure JSON: {data, layout}.")
    summary: StockDashboardSummary
    ohlcv_count: int
    data_source: Literal["yfinance", "stooq", "cache"]
    trading_day_stamp: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value: float) -> float | None:
    """Convert NaN / ±Inf to JSON-safe None so the React renderer shows em-dashes."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/anchor", response_model=AnchorResponse)
def anchor() -> AnchorResponse:
    """Return NYSE-local anchors so the client form doesn't need pandas-market-calendars.

    The source's `app.py:154` uses `datetime.now(ZoneInfo("America/New_York")).date()` —
    we mirror that exactly.
    """
    from zoneinfo import ZoneInfo

    ny = ZoneInfo("America/New_York")
    today_ny = datetime.now(ny).date()
    return AnchorResponse(
        today_nyse=today_ny,
        default_start=today_ny - timedelta(days=5 * 365),
        market_is_open=data._market_is_open_now(),
        trading_day_stamp=data.trading_day_stamp(),
    )


@router.post("/run", response_model=StockDashboardResponse)
def run(
    req: StockDashboardRequest,
    supabase=Depends(get_supabase),
) -> StockDashboardResponse:
    """Execute the stock-dashboard compute pipeline.

    Pipeline is identical to stock-dashboard/app.py:
      fetch_ohlcv → indicators → stats → build_figure
    """
    # Final guard: bound to NYSE today. Anchor endpoint should keep clients honest,
    # but never trust the wire.
    from zoneinfo import ZoneInfo

    ny = ZoneInfo("America/New_York")
    today_ny = datetime.now(ny).date()
    if req.end_date > today_ny:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"end_date {req.end_date.isoformat()} is after NYSE today ({today_ny.isoformat()})",
        )

    # Fetch — yfinance with Stooq fallback. The vendored data.fetch_ohlcv does
    # all the heavy lifting (retries, intraday bypass, diskcache, normalize).
    try:
        df = data.fetch_ohlcv(req.ticker, req.start_date, req.end_date)
    except DataFetchError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # network / parse / unexpected
        logger.exception("unexpected error fetching %s", req.ticker)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Upstream fetch failed for {req.ticker}",
        ) from exc

    if df.empty or "Close" not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"No usable OHLCV returned for {req.ticker}",
        )

    # Optionally upsert into Supabase ohlcv_cache (best-effort, never block).
    _maybe_upsert_ohlcv(supabase, req.ticker, df, source="yfinance")

    # Indicators — replicates stock-dashboard/app.py:78-99 exactly.
    df = df.copy()
    close = df["Close"]
    selected = set(req.indicators)
    if "SMA 20" in selected:
        df["sma_20"] = indicators.sma(close, length=20)
    if "SMA 50" in selected:
        df["sma_50"] = indicators.sma(close, length=50)
    if "EMA 12" in selected:
        df["ema_12"] = indicators.ema(close, length=12)
    if "EMA 26" in selected:
        df["ema_26"] = indicators.ema(close, length=26)
    if "Bollinger Bands" in selected:
        bb = indicators.bollinger_bands(close, length=20, k=2.0)
        df["bb_upper"] = bb["upper"]
        df["bb_middle"] = bb["middle"]
        df["bb_lower"] = bb["lower"]
    if "RSI" in selected:
        df["rsi"] = indicators.rsi(close, length=14)
    if "MACD" in selected:
        macd_df = indicators.macd(close, fast=12, slow=26, signal=9)
        df["macd"] = macd_df["macd"]
        df["macd_signal"] = macd_df["signal"]
        df["macd_hist"] = macd_df["histogram"]

    # Stats — pure function of close + rf.
    try:
        summary = stats.summary(close, rf_annual=req.rf_annual)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    # Figure — tag set translated from labels (mirrors app.py:121-133).
    indicator_tags = tuple(_LABEL_TO_TAG[lbl] for lbl in req.indicators if lbl in _LABEL_TO_TAG)
    fig = charts.build_figure(df, indicators=indicator_tags, candlestick=req.candlestick)
    # plotly.io.to_json handles numpy ndarrays and NaN/Inf natively; we then
    # parse back to a dict so Pydantic can serialize without seeing numpy types.
    figure_json: dict[str, Any] = json.loads(pio.to_json(fig, validate=False))

    response = StockDashboardResponse(
        figure=figure_json,
        summary=StockDashboardSummary(
            start=summary.start.date() if hasattr(summary.start, "date") else summary.start,
            end=summary.end.date() if hasattr(summary.end, "date") else summary.end,
            trading_days=summary.trading_days,
            cumulative_return=_safe_float(summary.cumulative_return),
            annualized_return=_safe_float(summary.annualized_return),
            annualized_volatility=_safe_float(summary.annualized_volatility),
            sharpe=_safe_float(summary.sharpe),
            max_drawdown=_safe_float(summary.max_drawdown),
            best_day=_safe_float(summary.best_day),
            worst_day=_safe_float(summary.worst_day),
        ),
        ohlcv_count=len(df),
        data_source="yfinance",  # TODO surface stooq when fallback fires
        trading_day_stamp=data.trading_day_stamp(),
    )

    _maybe_log_run(supabase, req, response)

    return response


# ---------------------------------------------------------------------------
# Supabase helpers (best-effort)
# ---------------------------------------------------------------------------


def _maybe_upsert_ohlcv(supabase, ticker: str, df: pd.DataFrame, source: str) -> None:
    """Best-effort upsert of OHLCV rows to platform.ohlcv_cache.

    Silent failure is intentional — the cache is a perf optimisation, not a
    correctness primitive. A Supabase outage must not break the dashboard.
    """
    if supabase is None or df.empty:
        return
    try:
        rows = []
        for ts, row in df.iterrows():
            rows.append(
                {
                    "symbol": ticker,
                    "date": (ts.date() if hasattr(ts, "date") else ts).isoformat(),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"]) if pd.notna(row["Volume"]) else 0,
                    "source": source,
                }
            )
        # Supabase Python SDK: schema().table().upsert()
        # platform schema is exposed via the PostgREST `schema` switch.
        supabase.schema("platform").table("ohlcv_cache").upsert(
            rows, on_conflict="symbol,date"
        ).execute()
    except Exception:
        logger.exception("ohlcv_cache upsert failed (non-fatal)")


def _maybe_log_run(supabase, req: StockDashboardRequest, resp: StockDashboardResponse) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "stock-dashboard",
                "params": req.model_dump(mode="json"),
                "result": {
                    "ohlcv_count": resp.ohlcv_count,
                    "data_source": resp.data_source,
                    "trading_day_stamp": resp.trading_day_stamp,
                    "summary": resp.summary.model_dump(mode="json"),
                },
                "status": "ok",
            }
        ).execute()
    except Exception:
        logger.exception("tool_runs insert failed (non-fatal)")


# _normalise_figure_json was removed: plotly.io.to_json + json.loads handles
# both numpy ndarrays and NaN/Inf coercion to JSON null in one pass.
