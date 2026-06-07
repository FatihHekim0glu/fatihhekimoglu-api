"""High-level price loader.

This module is the single entry point used by the rest of the package to obtain
OHLCV bars. It:

* Validates ticker strings and the requested date range.
* Reads previously cached shards from ``<cache_dir>/prices/<TICKER>.parquet``.
* If a shard is missing or ``force_refresh`` is set, delegates to
  :mod:`pairs.data._yfinance_adapter` to fetch fresh bars, writes them
  atomically to the cache, and updates the manifest.
* If a shard is present but does not cover the full range, fetches only the
  missing tail and concatenates.

The returned frame has MultiIndex columns ``(ticker, field)`` so downstream
code can vectorize over both axes.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from pairs._config import get_settings
from pairs._exceptions import InputError, InsufficientDataError
from pairs.data._yfinance_adapter import _batch_download
from pairs.data.cache import _atomic_write_parquet, _read_parquet
from pairs.data.manifest import build_entry, update_manifest_entry

logger = logging.getLogger(__name__)

_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
_MANIFEST_NAME = "manifest.json"
_PRICES_SUBDIR = "prices"


def _validate_tickers(tickers: list[str]) -> list[str]:
    """Uppercase, dedupe (preserving order), and validate tickers.

    Parameters
    ----------
    tickers
        Caller-supplied symbols. Must be non-empty and conform to ``_TICKER_RE``.

    Returns
    -------
    list of str
        Cleaned ticker list.
    """
    if not tickers:
        raise InputError("tickers must be non-empty")
    seen: set[str] = set()
    cleaned: list[str] = []
    for raw in tickers:
        upper = str(raw).strip().upper()
        if not _TICKER_RE.fullmatch(upper):
            raise InputError(f"invalid ticker format: {raw!r}")
        if upper in seen:
            continue
        seen.add(upper)
        cleaned.append(upper)
    return cleaned


def _validate_range(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Parse and bounds-check the requested date range."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts >= end_ts:
        raise InputError(f"start {start_ts} must be strictly before end {end_ts}")
    return start_ts, end_ts


def _cache_paths() -> tuple[Path, Path, Path]:
    """Return ``(cache_root, prices_dir, manifest_path)`` for the active settings."""
    settings = get_settings()
    cache_root = Path(settings.cache_dir)
    prices_dir = cache_root / _PRICES_SUBDIR
    manifest_path = cache_root / _MANIFEST_NAME
    return cache_root, prices_dir, manifest_path


def _load_cached(ticker: str, prices_dir: Path) -> pd.DataFrame | None:
    """Read the cached shard for ``ticker``, or return ``None`` if absent."""
    path = prices_dir / f"{ticker}.parquet"
    if not path.is_file():
        return None
    return _read_parquet(path)


def _fetch_and_cache(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    prices_dir: Path,
    manifest_path: Path,
) -> pd.DataFrame:
    """Fetch ``ticker`` over ``[start, end)`` and persist it to the cache.

    Parameters
    ----------
    ticker
        Single uppercase symbol.
    start, end
        Half-open range; ``end`` is exclusive (yfinance semantics).
    prices_dir
        Directory under which ``<TICKER>.parquet`` is written.
    manifest_path
        Manifest JSON to update.

    Returns
    -------
    pandas.DataFrame
        MultiIndex column frame restricted to ``ticker``.

    Raises
    ------
    InsufficientDataError
        If the adapter returns no rows (e.g. delisted or never-listed symbol).
    """
    raw = _batch_download([ticker], str(start.date()), str(end.date()))
    if ticker in raw.columns.get_level_values(0):
        frame = raw.loc[:, [ticker]].dropna(how="all")
    else:
        frame = raw.dropna(how="all")
    if frame.empty:
        raise InsufficientDataError(f"no rows returned for {ticker} in [{start}, {end})")
    path = prices_dir / f"{ticker}.parquet"
    _atomic_write_parquet(frame, path)
    entry = build_entry(
        relpath=f"{_PRICES_SUBDIR}/{ticker}.parquet",
        file_path=path,
        rows=len(frame),
        start=str(frame.index.min()),
        end=str(frame.index.max()),
        provider="yfinance",
    )
    update_manifest_entry(manifest_path, entry)
    return frame


def _ensure_coverage(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    prices_dir: Path,
    manifest_path: Path,
) -> pd.DataFrame:
    """Return a cached + freshly-fetched frame covering ``[start, end)``.

    If the cached shard already covers the tail, no fetch is performed. Only the
    missing window past the cached ``max`` is fetched.
    """
    cached = _load_cached(ticker, prices_dir)
    if cached is None or cached.empty:
        return _fetch_and_cache(ticker, start, end, prices_dir, manifest_path)
    cached_max = pd.Timestamp(cached.index.max())
    # Compare in tz-naive UTC dates to avoid tz-mismatch surprises.
    needed_end_date = end.tz_localize(None) if end.tz is not None else end
    cached_max_naive = cached_max.tz_localize(None) if cached_max.tz is not None else cached_max
    if cached_max_naive >= needed_end_date - pd.Timedelta(days=1):
        return cached
    next_start = (cached_max_naive + pd.Timedelta(days=1)).normalize()
    try:
        fresh = _fetch_and_cache(ticker, next_start, end, prices_dir, manifest_path)
    except InsufficientDataError:
        return cached
    combined = pd.concat([cached, fresh])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    path = prices_dir / f"{ticker}.parquet"
    _atomic_write_parquet(combined, path)
    entry = build_entry(
        relpath=f"{_PRICES_SUBDIR}/{ticker}.parquet",
        file_path=path,
        rows=len(combined),
        start=str(combined.index.min()),
        end=str(combined.index.max()),
        provider="yfinance",
    )
    update_manifest_entry(manifest_path, entry)
    return combined


def load_prices(
    tickers: list[str],
    start: str,
    end: str,
    *,
    force_refresh: bool = False,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Load OHLCV bars for ``tickers`` over ``[start, end)``.

    Cache-aware: previously fetched shards are reused unless ``force_refresh``
    is set. Partial windows are extended by fetching only the missing tail.

    Parameters
    ----------
    tickers
        List of ticker symbols. Case-insensitive; duplicates are dropped.
    start, end
        ISO date strings. Half-open: ``end`` is exclusive.
    force_refresh
        If True, ignore any cached shard and re-fetch from the provider.
    rng
        Optional random generator. Currently unused at this layer but accepted
        for API consistency with the project-wide stochastic-function contract;
        downstream stochastic helpers (e.g. backoff jitter) accept their own.

    Returns
    -------
    pandas.DataFrame
        Frame with a tz-aware ``DatetimeIndex`` and ``MultiIndex`` columns
        ``(ticker, field)`` over the union of tickers' available rows.

    Raises
    ------
    InputError
        For empty ticker lists, invalid ticker formats, or an invalid range.
    InsufficientDataError
        If every ticker returns zero rows.
    """
    del rng  # accepted for contract symmetry; not used at this layer
    cleaned = _validate_tickers(tickers)
    start_ts, end_ts = _validate_range(start, end)
    _, prices_dir, manifest_path = _cache_paths()
    prices_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for ticker in cleaned:
        if force_refresh:
            path = prices_dir / f"{ticker}.parquet"
            if path.exists():
                path.unlink()
        try:
            frame = _ensure_coverage(ticker, start_ts, end_ts, prices_dir, manifest_path)
        except InsufficientDataError as exc:
            logger.warning("skipping %s: %s", ticker, exc)
            continue
        frames.append(frame)

    if not frames:
        raise InsufficientDataError(f"no data for any of {cleaned}")
    merged = pd.concat(frames, axis=1).sort_index()
    if merged.index.tz is None:
        merged.index = merged.index.tz_localize("UTC")
    mask = (merged.index >= start_ts.tz_localize("UTC") if start_ts.tz is None else start_ts) & (
        merged.index < end_ts.tz_localize("UTC") if end_ts.tz is None else end_ts
    )
    return merged.loc[mask]
