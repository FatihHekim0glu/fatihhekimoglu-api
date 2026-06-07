"""Price download and return computation helpers."""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def load_prices(ticker: str, start, end) -> pd.Series:
    """Adjusted-close price Series, tz-naive DatetimeIndex."""
    if not ticker:
        raise ValueError("no data for ")
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if raw is None or len(raw) == 0 or "Close" not in raw.columns:
        raise ValueError(f"no data for {ticker}")
    prices = raw["Close"]
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]
    if prices.empty:
        raise ValueError(f"no data for {ticker}")
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices.name = ticker
    return prices


def compute_returns(prices: pd.Series, freq: str = "M") -> pd.Series:
    """Period returns as decimal; freq='M' resamples to month-end last trading day."""
    if freq == "M":
        rets = prices.resample("ME").last().pct_change()
    elif freq == "D":
        rets = prices.pct_change()
    else:
        raise ValueError(f"unsupported freq: {freq}")
    return rets.dropna()
