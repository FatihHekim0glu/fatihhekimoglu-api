"""Stock Price Forecast tool — wraps the vendored LSTM+Attention pipeline.

Endpoint:
  POST /tools/stock-price-forecast/run

Pipeline (single-shot synchronous; v1 acceptance):
  yfinance.download (2010-01-01..today)
    -> add_technical_indicators (RSI, MACD x3, BB, MAs, lags, rolling)
    -> MinMaxScaler.fit (14 features)
    -> autoregressive 60-step roll for `forecast_days` (LSTM+Attention)
    -> two Plotly figures (history line + forecast scatter overlay) + forecast table

Notes:
  - Long-running: yfinance pull (~2-10s) + sequential model.predict calls
    (~100-300ms each on CPU). A 30-day forecast can take 10-30s; v1 just
    blocks the request — clients should set a generous timeout.
  - The Keras model + 14-feature MinMaxScaler are not exposed to the wire;
    only dates + price arrays go over JSON.
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from ..deps import get_supabase
from ..lib.stock_price_forecast.inference import ForecastError, run_forecast

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools/stock-price-forecast", tags=["stock-price-forecast"])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,15}$")
# How many trailing history bars to send back to the client for plotting
# context. The full 2010..today series is ~3500 rows — too chunky for JSON.
_HISTORY_TAIL = 250


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class StockPriceForecastRequest(BaseModel):
    ticker: str = Field("AAPL", max_length=15)
    forecast_days: int = Field(7, ge=1, le=30)

    @field_validator("ticker")
    @classmethod
    def _ticker_shape(cls, v: str) -> str:
        v = v.strip().upper()
        if not v:
            raise ValueError("ticker cannot be empty")
        if not _TICKER_RE.match(v):
            raise ValueError(
                "ticker must contain only A-Z, 0-9, '.', '-' (1-15 chars); "
                "examples: AAPL, BRK-B, BP.L"
            )
        return v


class ForecastRow(BaseModel):
    date: str
    predicted_close: float


class StockPriceForecastSummary(BaseModel):
    ticker: str
    last_close: float
    last_close_date: str
    forecast_days: int
    first_forecast_close: float | None
    final_forecast_close: float | None
    forecast_change_pct: float | None = Field(
        None,
        description="Percent change from last actual close to the final forecast day (e.g. 0.0234 = +2.34%).",
    )


class StockPriceForecastResponse(BaseModel):
    price_history_chart: dict[str, Any] = Field(
        ..., description="Plotly figure JSON of historical Close vs Date."
    )
    forecast_chart: dict[str, Any] = Field(
        ...,
        description="Plotly figure JSON of recent history + predicted Close vs future dates.",
    )
    forecast_table: list[ForecastRow]
    summary: StockPriceForecastSummary


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


def _build_history_figure(
    ticker: str, dates: list[str], close: list[float]
) -> dict[str, Any]:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=close,
            mode="lines",
            name="Close",
            line={"width": 1.5},
        )
    )
    fig.update_layout(
        title=f"{ticker} — Historical Close (since 2010)",
        xaxis_title="Date",
        yaxis_title="Close (USD)",
        template="plotly_white",
        showlegend=False,
    )
    return json.loads(pio.to_json(fig, validate=False))


def _build_forecast_figure(
    ticker: str,
    hist_dates: list[str],
    hist_close: list[float],
    fcst_dates: list[str],
    fcst_close: list[float],
) -> dict[str, Any]:
    tail = min(_HISTORY_TAIL, len(hist_dates))
    tail_dates = hist_dates[-tail:]
    tail_close = hist_close[-tail:]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=tail_dates,
            y=tail_close,
            mode="lines",
            name="Recent close",
            line={"width": 1.5, "color": "#4b5563"},
        )
    )
    # Bridge segment from last known close to first forecast so the eye reads
    # a continuous curve.
    if tail_dates and fcst_dates:
        fig.add_trace(
            go.Scatter(
                x=[tail_dates[-1], fcst_dates[0]],
                y=[tail_close[-1], fcst_close[0]],
                mode="lines",
                line={"width": 1.5, "color": "#0d9488", "dash": "dot"},
                showlegend=False,
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=fcst_dates,
            y=fcst_close,
            mode="lines+markers",
            name="Forecast",
            marker={"size": 7, "color": "#0d9488"},
            line={"width": 2, "color": "#0d9488"},
        )
    )
    fig.update_layout(
        title=f"{ticker} — Next {len(fcst_dates)}-day forecast",
        xaxis_title="Date",
        yaxis_title="Close (USD)",
        template="plotly_white",
        legend={"orientation": "h", "y": -0.18},
    )
    return json.loads(pio.to_json(fig, validate=False))


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/run", response_model=StockPriceForecastResponse)
def run(
    req: StockPriceForecastRequest,
    supabase=Depends(get_supabase),
) -> StockPriceForecastResponse:
    """Execute the LSTM+Attention forecast pipeline for one ticker.

    Synchronous single-shot. A 30-day request runs ~10-30s on CPU because the
    autoregressive loop runs `forecast_days` sequential `model.predict` calls.
    """
    try:
        result = run_forecast(req.ticker, req.forecast_days)
    except ForecastError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # network / parse / TF / unexpected
        logger.exception(
            "unexpected error running forecast for %s (%d days)",
            req.ticker,
            req.forecast_days,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                f"Forecast pipeline failed for {req.ticker}. "
                "The upstream data feed or model inference returned an error."
            ),
        ) from exc

    hist_dates: list[str] = result["history_dates"]
    hist_close: list[float] = result["history_close"]
    fcst_dates: list[str] = result["forecast_dates"]
    fcst_close: list[float] = result["forecast_close"]
    last_close: float = float(result["last_close"])
    last_close_date: str = result["last_close_date"]

    if not fcst_close:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Forecast pipeline returned no predictions for {req.ticker}",
        )

    first_fcst = _safe_float(fcst_close[0])
    final_fcst = _safe_float(fcst_close[-1])
    change_pct: float | None = None
    if final_fcst is not None and last_close != 0:
        change_pct = (final_fcst - last_close) / last_close

    price_history_chart = _build_history_figure(req.ticker, hist_dates, hist_close)
    forecast_chart = _build_forecast_figure(
        req.ticker, hist_dates, hist_close, fcst_dates, fcst_close
    )
    forecast_table = [
        ForecastRow(date=d, predicted_close=float(p))
        for d, p in zip(fcst_dates, fcst_close, strict=True)
    ]

    response = StockPriceForecastResponse(
        price_history_chart=price_history_chart,
        forecast_chart=forecast_chart,
        forecast_table=forecast_table,
        summary=StockPriceForecastSummary(
            ticker=req.ticker,
            last_close=last_close,
            last_close_date=last_close_date,
            forecast_days=req.forecast_days,
            first_forecast_close=first_fcst,
            final_forecast_close=final_fcst,
            forecast_change_pct=_safe_float(change_pct),
        ),
    )

    _maybe_log_run(supabase, req, response)
    return response


# ---------------------------------------------------------------------------
# Supabase helpers (best-effort)
# ---------------------------------------------------------------------------


def _maybe_log_run(
    supabase,
    req: StockPriceForecastRequest,
    resp: StockPriceForecastResponse,
) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "stock-price-forecast",
                "params": req.model_dump(mode="json"),
                "result": {
                    "summary": resp.summary.model_dump(mode="json"),
                    "forecast_table": [r.model_dump(mode="json") for r in resp.forecast_table],
                },
                "status": "ok",
            }
        ).execute()
    except Exception:
        logger.exception("tool_runs insert failed (non-fatal)")
