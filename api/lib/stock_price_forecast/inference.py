"""Inference-only re-implementation of the forecast pipeline.

This module is a thin shim that re-uses the EXACT logic from the vendored
`app.py` (data fetch, technical indicators, autoregressive 60-step roll) but
without the Streamlit calls and the eager module-level model load. The shim
exists so the FastAPI router can import compute helpers without dragging in
Streamlit or paying the model-load cost at import time.

Pipeline (mirrors app.py:14-74 exactly):
  yfinance.download(ticker, 2010-01-01..today)
    -> add_technical_indicators (RSI, MACD x3, BB, MAs, lags, rolling)
    -> MinMaxScaler.fit on 14 features
    -> last 60-step window, autoregressive roll for forecast_days
    -> inverse-scale the Close column only
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Suppress noisy TF logs before any TF import (matches main.py:22).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# 14-feature ordered list — must match app.py:98-99 exactly.
FEATURES: list[str] = [
    "Close",
    "RSI",
    "MACD",
    "MACD_diff",
    "MACD_signal",
    "BB_high",
    "BB_low",
    "MA10",
    "MA50",
    "MA200",
    "Close_lag1",
    "Close_lag2",
    "Close_roll_mean",
    "Close_roll_std",
]

LOOK_BACK: int = 60

_MODEL_PATH = Path(__file__).parent / "stock_model.h5"

# Lazy-loaded singleton — load is ~1-2s and TF init is heavy.
_model: Any = None
_model_lock = threading.Lock()


class ForecastError(RuntimeError):
    """Raised when the forecast pipeline cannot produce a result."""


def _load_model() -> Any:
    """Load the Keras model on first use, then memoize. Thread-safe."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:  # double-checked
            return _model
        import tensorflow as tf
        from tensorflow.keras.layers import Attention

        if not _MODEL_PATH.exists():
            raise ForecastError(
                f"Pretrained model file not found at {_MODEL_PATH}; cannot serve forecasts."
            )
        logger.info("loading keras model from %s", _MODEL_PATH)
        _model = tf.keras.models.load_model(
            str(_MODEL_PATH), custom_objects={"Attention": Attention}
        )
        return _model


def get_stock_data(ticker: str) -> pd.DataFrame:
    """Download 2010-01-01..today daily OHLCV from yfinance and add indicators.

    Mirrors app.py:17-21 verbatim.
    """
    import yfinance as yf

    data = yf.download(
        ticker,
        start="2010-01-01",
        end=_dt.date.today(),
        progress=False,
        auto_adjust=False,
    )
    if data is None or data.empty:
        raise ForecastError(f"yfinance returned no rows for {ticker!r}")

    # yfinance ≥0.2.x returns a MultiIndex column frame even for a single ticker
    # in some versions; flatten so `data['Close']` works like in the source.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.reset_index(inplace=True)
    data = add_technical_indicators(data)
    return data


def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Mirrors app.py:23-52 verbatim."""
    import ta

    # Ensure 1-D Series for the `ta` library.
    close = data["Close"]
    if isinstance(close, pd.DataFrame):  # defensive flatten
        close = close.iloc[:, 0]
    close = pd.Series(np.asarray(close).ravel(), index=data.index, name="Close")

    # RSI
    data["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    # MACD
    macd = ta.trend.MACD(close)
    data["MACD"] = macd.macd()
    data["MACD_diff"] = macd.macd_diff()
    data["MACD_signal"] = macd.macd_signal()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close, window=20)
    data["BB_high"] = bb.bollinger_hband()
    data["BB_low"] = bb.bollinger_lband()

    # Moving Averages
    data["MA10"] = close.rolling(window=10).mean()
    data["MA50"] = close.rolling(window=50).mean()
    data["MA200"] = close.rolling(window=200).mean()

    # Lag features
    data["Close_lag1"] = close.shift(1)
    data["Close_lag2"] = close.shift(2)

    # Rolling statistics
    data["Close_roll_mean"] = close.rolling(window=5).mean()
    data["Close_roll_std"] = close.rolling(window=5).std()

    data.dropna(inplace=True)
    return data


def make_future_predictions(
    data: pd.DataFrame, forecast_days: int, scaler: Any, features: list[str]
) -> np.ndarray:
    """Autoregressive 60-step roll. Mirrors app.py:54-74 verbatim."""
    model = _load_model()

    data_scaled = scaler.transform(data[features])
    last_sequence = data_scaled[-LOOK_BACK:]
    last_sequence = np.expand_dims(last_sequence, axis=0)
    future_predictions: list[np.ndarray] = []
    current_sequence = last_sequence.copy()

    for _ in range(forecast_days):
        next_pred = model.predict(current_sequence, verbose=0)
        future_predictions.append(next_pred[0])
        next_pred_scaled = np.concatenate(
            (next_pred, current_sequence[:, -1, 1:]), axis=1
        )
        current_sequence = np.append(
            current_sequence[:, 1:, :], [next_pred_scaled], axis=1
        )

    fut_arr = np.array(future_predictions)
    fut_inverse = scaler.inverse_transform(
        np.concatenate(
            (fut_arr, np.zeros((fut_arr.shape[0], len(features) - 1))), axis=1
        )
    )[:, 0]

    return fut_inverse


def run_forecast(ticker: str, forecast_days: int) -> dict[str, Any]:
    """Top-level entrypoint used by the FastAPI router.

    Returns a dict with:
        history_dates   : list[str]  YYYY-MM-DD, full series since 2010 (post-dropna)
        history_close   : list[float]
        forecast_dates  : list[str]  YYYY-MM-DD, next `forecast_days` calendar days
        forecast_close  : list[float]
        last_close      : float
        last_close_date : str
    """
    from sklearn.preprocessing import MinMaxScaler

    data = get_stock_data(ticker)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data[FEATURES])

    predictions = make_future_predictions(data, forecast_days, scaler, FEATURES)

    last_date = pd.Timestamp(data["Date"].iloc[-1])
    prediction_dates = pd.date_range(
        last_date + pd.Timedelta(days=1), periods=forecast_days
    )

    history_close_arr = np.asarray(data["Close"]).ravel().astype(float)
    last_close = float(history_close_arr[-1])

    return {
        "history_dates": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in data["Date"]],
        "history_close": history_close_arr.tolist(),
        "forecast_dates": [d.strftime("%Y-%m-%d") for d in prediction_dates],
        "forecast_close": [float(p) for p in predictions],
        "last_close": last_close,
        "last_close_date": last_date.strftime("%Y-%m-%d"),
    }
