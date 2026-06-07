"""Polygon adapter for the pairs-trading tool.

Bridges the shared :class:`PolygonProvider` to the
``(ticker, field)``-MultiIndex DataFrame that the vendored
``pairs.selection.screen_cointegration`` (via the router's
``_flatten_close_prices``) expects.

Keeping the adapter here — outside the vendored ``_vendor/`` tree — preserves
the rule that vendored sources are never edited. The router decides whether
to call this adapter (Polygon path) or fall back to the vendored
``load_prices`` (yfinance path).
"""

from __future__ import annotations

import logging
from datetime import date as _date
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..polygon.provider import PolygonProvider, PolygonProviderFallback

logger = logging.getLogger(__name__)


def load_prices_via_provider(
    provider: "PolygonProvider | PolygonProviderFallback",
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Fetch OHLCV for ``tickers`` over ``[start, end)`` via the provider.

    Returns a DataFrame with a tz-naive ``DatetimeIndex`` and ``MultiIndex``
    columns ``(ticker, field)`` where ``field`` is one of
    ``Open/High/Low/Close/Volume``. Matches the column shape returned by
    ``pairs.data.loader.load_prices`` so the existing
    ``_flatten_close_prices`` helper keeps working unchanged.

    Per-ticker failures are logged and the symbol is dropped from the result
    (mirroring the vendored loader's behaviour). Range is half-open ``[start,
    end)``, but Polygon EOD bounds are inclusive — we subtract one day from
    ``end`` to keep parity.
    """
    if not tickers:
        return pd.DataFrame()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts >= end_ts:
        raise ValueError(f"start {start} must be strictly before end {end}")
    start_d: _date = start_ts.date()
    # end is exclusive in the loader contract; Polygon get_eod is inclusive.
    end_d: _date = (end_ts - pd.Timedelta(days=1)).date()
    if end_d < start_d:
        end_d = start_d

    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        try:
            df = provider.get_eod(ticker, start_d, end_d)
        except Exception as exc:  # noqa: BLE001 — degrade per-ticker
            logger.warning(
                "polygon get_eod(%s) failed (%s); skipping",
                ticker,
                exc.__class__.__name__,
            )
            continue
        if df is None or df.empty:
            continue
        wanted = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
        if not wanted:
            continue
        sub = df[wanted].copy()
        sub.columns = pd.MultiIndex.from_product([[ticker], sub.columns])
        frames.append(sub)

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, axis=1).sort_index()
    if getattr(merged.index, "tz", None) is not None:
        merged.index = merged.index.tz_localize(None)
    mask = (merged.index >= start_ts) & (merged.index < end_ts)
    return merged.loc[mask]
