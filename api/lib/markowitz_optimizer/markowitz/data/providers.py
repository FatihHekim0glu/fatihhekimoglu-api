"""Price providers.

The :class:`PriceProvider` protocol defines the minimal interface every
price source must implement. :class:`YFinanceProvider` is the default
implementation; :class:`CachedProvider` wraps any provider with the
on-disk Parquet cache; :class:`FallbackProvider` tries a list of
providers in order, falling back on rate-limit / unavailable errors.
"""

from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

import pandas as pd

from .cache import read_cache, write_cache
from .exceptions import (
    DataIntegrityError,
    EmptyDataError,
    ProviderUnavailableError,
    RateLimitError,
)

DateLike = str | _dt.date | pd.Timestamp


@runtime_checkable
class PriceProvider(Protocol):
    """Protocol every price source must satisfy."""

    name: str

    def fetch(
        self,
        ticker: str,
        start: DateLike,
        end: DateLike,
        *,
        frequency: str = "1d",
    ) -> pd.DataFrame:
        """Return a frame indexed by tz-naive midnight, single ``close`` column."""
        ...


def _normalize_frame(df: pd.DataFrame, *, ticker: str) -> pd.DataFrame:
    """Coerce a raw price frame into the canonical contract."""
    if df.empty:
        raise EmptyDataError(f"no rows returned for {ticker}")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.DatetimeIndex(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    if "close" not in df.columns:
        # Common aliases
        for cand in ("Close", "Adj Close", "adj_close"):
            if cand in df.columns:
                df = df.rename(columns={cand: "close"})
                break
    if "close" not in df.columns:
        raise DataIntegrityError(
            f"frame for {ticker} has no 'close' column (got {list(df.columns)})"
        )

    out = df[["close"]].astype("float64")
    if bool(out["close"].isna().all()):
        raise EmptyDataError(f"all-NaN close series for {ticker}")
    return cast(pd.DataFrame, out)


class YFinanceProvider:
    """Provider backed by `yfinance.download`."""

    name = "yfinance"

    def __init__(self) -> None:
        try:
            import yfinance as _yf  # noqa: F401, PLC0415
        except ImportError as exc:
            raise ProviderUnavailableError(
                "yfinance is not installed; install with `pip install yfinance`"
            ) from exc

    def fetch(
        self,
        ticker: str,
        start: DateLike,
        end: DateLike,
        *,
        frequency: str = "1d",
    ) -> pd.DataFrame:
        import yfinance as yf  # noqa: PLC0415

        try:
            raw = yf.download(
                tickers=ticker,
                start=pd.Timestamp(start).strftime("%Y-%m-%d"),
                end=pd.Timestamp(end).strftime("%Y-%m-%d"),
                interval=frequency,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
        except Exception as exc:  # network or upstream parser failures
            msg = str(exc).lower()
            if "rate" in msg and "limit" in msg:
                raise RateLimitError(f"yfinance rate-limited on {ticker}") from exc
            raise ProviderUnavailableError(f"yfinance failed to fetch {ticker}: {exc}") from exc

        if raw is None or raw.empty:
            raise EmptyDataError(f"yfinance returned no data for {ticker}")

        # yfinance may return a MultiIndex column for single tickers in newer
        # versions; collapse it.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        if "Close" not in raw.columns:
            raise DataIntegrityError(f"yfinance response for {ticker} missing 'Close' column")
        frame = pd.DataFrame({"close": raw["Close"]})
        return _normalize_frame(frame, ticker=ticker)


class CachedProvider:
    """Wrap a :class:`PriceProvider` with on-disk Parquet caching."""

    def __init__(self, inner: PriceProvider, root: str | os.PathLike[str]) -> None:
        self._inner = inner
        self._root = Path(root)
        self.name = f"cached({inner.name})"

    def fetch(
        self,
        ticker: str,
        start: DateLike,
        end: DateLike,
        *,
        frequency: str = "1d",
    ) -> pd.DataFrame:
        cached = read_cache(self._root, ticker, self._inner.name, frequency)
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        if cached is not None and not cached.empty:
            covered_start = cached.index.min()
            covered_end = cached.index.max()
            if covered_start <= start_ts and covered_end >= end_ts - pd.Timedelta(days=1):
                sliced = cached.loc[(cached.index >= start_ts) & (cached.index < end_ts)].copy()
                return cast(pd.DataFrame, sliced)

        fresh = self._inner.fetch(ticker, start, end, frequency=frequency)
        write_cache(fresh, self._root, ticker, self._inner.name, frequency)
        return fresh


class FallbackProvider:
    """Try each provider in order, advancing past rate-limit / unavailable."""

    name = "fallback"

    def __init__(self, providers: list[PriceProvider]) -> None:
        if not providers:
            raise ValueError("FallbackProvider requires at least one inner provider")
        self._providers = providers

    def fetch(
        self,
        ticker: str,
        start: DateLike,
        end: DateLike,
        *,
        frequency: str = "1d",
    ) -> pd.DataFrame:
        last_exc: Exception | None = None
        for prov in self._providers:
            try:
                return prov.fetch(ticker, start, end, frequency=frequency)
            except (RateLimitError, ProviderUnavailableError) as exc:
                last_exc = exc
                continue
        assert last_exc is not None
        raise ProviderUnavailableError(
            f"all providers failed for {ticker}; last error: {last_exc}"
        ) from last_exc
