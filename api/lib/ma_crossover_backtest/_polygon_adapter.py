"""Polygon-backed adapter for ma-crossover-backtest's close-price fetch.

The vendored data.py module is preserved byte-for-byte - adding the Polygon
path here keeps the upstream fork clean. The router calls
``load_close_via_provider`` instead of ``data.load_close`` when it has a
provider in hand.

Return semantics match ``data.load_close``: a tz-naive ``DatetimeIndex``-keyed
``pd.Series`` of adjusted close, NaN-free, named after the ticker.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Protocol

import pandas as pd

from .data import DataQualityError, load_close, validate

logger = logging.getLogger(__name__)


class _ProviderProto(Protocol):
    """Structural type for the bits of PolygonProvider we use."""

    def get_eod(self, ticker: str, start: date, end: date) -> pd.DataFrame: ...


def _is_polygon_provider(provider: object) -> bool:
    """Cheap check: real PolygonProvider vs. the yfinance fallback shell."""
    cls_name = provider.__class__.__name__
    return cls_name == "PolygonProvider"


def load_close_via_provider(
    provider: _ProviderProto | None,
    ticker: str,
    *,
    start: date,
    end: date,
) -> tuple[pd.Series, str]:
    """Load adjusted-close for ``ticker`` via the provider, with a sane fallback.

    Returns
    -------
    (close, data_source)
        ``close`` is the same series shape ``data.load_close`` produces.
        ``data_source`` is ``"polygon"`` when the real Polygon path served the
        bars, ``"yfinance"`` when the local parquet/yfinance fallback ran.
        ``"cache"`` is reserved for a future fast-path that proves every bar
        came from the Supabase cache without re-fetching - today the provider
        merges cache + Polygon transparently so we cannot distinguish.
    """
    ticker = ticker.strip().upper()

    # Provider absent (test injected None, etc.) → fall back to the vendored
    # path. Same behaviour as the pre-retrofit router.
    if provider is None:
        close = load_close(ticker, start=start, end=end)
        return close, "yfinance"

    # If the factory handed us the yfinance fallback, route through the
    # vendored data.py which already has parquet caching + Stooq retry.
    if not _is_polygon_provider(provider):
        close = load_close(ticker, start=start, end=end)
        return close, "yfinance"

    # Real Polygon path: fetch OHLCV, validate, drop to close series.
    try:
        df = provider.get_eod(ticker, start, end)
    except Exception as exc:
        logger.warning(
            "Polygon fetch failed for %s (%s); falling back to yfinance",
            ticker,
            exc.__class__.__name__,
        )
        close = load_close(ticker, start=start, end=end)
        return close, "yfinance"

    if df is None or df.empty:
        raise DataQualityError(f"Polygon returned no rows for {ticker}")

    # Run the same validator the vendored loader uses so downstream
    # backtester invariants hold (OHLC sanity, jump count, gap warnings).
    report = validate(df, ticker)
    for w in report.warnings_emitted:
        logger.warning("%s: %s", ticker, w)
    df = df.dropna(how="any")

    s = df["Close"].astype("float64")
    s.name = ticker
    return s, "polygon"
