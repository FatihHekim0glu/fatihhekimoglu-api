"""Data acquisition: yfinance primary with Stooq fallback, disk-cached.

Yahoo's anti-scrape throttles aggressively, so every call is treated as
eventually-failing: chrome impersonation via curl_cffi, exponential backoff,
cache on success, fall back to a direct Stooq CSV download when yfinance
keeps failing.

Key gotchas baked in here:
  - Use curl_cffi.requests.Session(impersonate="chrome124"), NOT requests.Session
    (vanilla requests broke in yfinance 0.2.59).
  - Pin auto_adjust=True explicitly; yfinance changed the default in 0.2.51.
  - Detect failure on empty DataFrame OR all-NaN Close, not just exceptions -
    yfinance often silently returns empty.
  - Cache on disk via diskcache, key = (cache_version, ticker, start, end,
    day_stamp); the day_stamp flips after each day's actual NYSE close (read
    from the XNYS schedule, which handles DST + half-day closes). When the
    requested range includes today AND market is still open, cache is
    bypassed entirely so the user sees fresh intraday data - only finalised
    historical bars get persisted.
  - Stooq fallback uses a direct CSV download via urllib so we don't need
    pandas-datareader (that package imports the removed `distutils` module on
    Python 3.12 and breaks at import time).
"""

from __future__ import annotations

import functools
import io
import logging
import random
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from typing import Final
from zoneinfo import ZoneInfo

import diskcache
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf
from curl_cffi import requests as curl_requests

logger = logging.getLogger(__name__)

_MAX_RETRIES: Final[int] = 4
_OHLCV_COLUMNS: Final[tuple[str, ...]] = ("Open", "High", "Low", "Close", "Volume")
_STOOQ_CSV_URL: Final[str] = "https://stooq.com/q/d/l/"
_STOOQ_MAX_BYTES: Final[int] = 10 * 1024 * 1024  # cap response read at 10 MB
# Bump this when the cached DataFrame format/normalization changes so old
# entries are skipped (e.g. after the tz-localize fix, stale pre-fix frames
# in `.yfcache` must be ignored). Bumped to v3 for the end-date inclusivity
# fix: pre-v3 entries were fetched with yfinance's exclusive `end`, but the
# current code passes `end + 1 day` to make both bounds inclusive.
_CACHE_VERSION: Final[str] = "v3"


# Lazy singletons via lru_cache: Streamlit reruns each user interaction on a
# fresh worker thread, so a true module-level singleton can be touched
# concurrently. diskcache.Cache is SQLite-backed and thread-safe internally,
# so wrapping it here only guarantees a single instance per process.
# curl_cffi.requests.Session is NOT documented as thread-safe; lru_cache keeps
# us to one shared instance (acceptable for v1 - worst case is two reruns
# racing on the same TLS context, both succeed but slightly slower). A full
# fix would use a threading.local pool; revisit if we see concurrency errors.
@functools.lru_cache(maxsize=1)
def _get_cache() -> diskcache.Cache:
    return diskcache.Cache(".yfcache")


@functools.lru_cache(maxsize=1)
def _get_session():
    return curl_requests.Session(impersonate="chrome124")


class DataFetchError(Exception):
    """Raised when both yfinance and Stooq fail to return usable OHLCV data."""


def fetch_ohlcv(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Fetch daily OHLCV for `ticker` between `start` and `end` (both inclusive).

    Returns a DataFrame indexed by date with columns Open, High, Low, Close,
    Volume - Close is split- and dividend-adjusted.

    Raises `DataFetchError` when the ticker is invalid or no data is available
    from either yfinance or Stooq.
    """
    cache = _get_cache()
    cache_key = (
        f"{_CACHE_VERSION}|{ticker}|{start.isoformat()}|{end.isoformat()}|{trading_day_stamp()}"
    )

    # Intraday cache bypass: if the requested range includes today AND we are
    # still before today's NYSE close, the today-bar is changing in real time.
    # Skip BOTH read and write so the user sees fresh data during the session
    # and we don't poison the cache with a partial bar. Historical-only ranges
    # and post-close fetches go through cache normally.
    today_et = datetime.now(_NYSE).date()
    intraday_bypass = end >= today_et and _market_is_open_now()

    if not intraday_bypass:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    # Primary: yfinance with chrome-impersonating session, up to 4 attempts.
    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            # yfinance's `end` is EXCLUSIVE; Stooq's `d2` is inclusive. Add a
            # day so both sources return the same row count for identical
            # (start, end) UI inputs - otherwise cache hit vs source switch
            # would silently change the row count under the user.
            df = yf.Ticker(ticker, session=_get_session()).history(
                start=start,
                end=end + timedelta(days=1),
                auto_adjust=True,
            )
            if df is None or df.empty or df["Close"].isna().all():
                # yfinance silently returns empty on rate-limit / bad ticker;
                # treat as a transient failure and back off.
                last_error = DataFetchError(
                    f"yfinance returned empty/all-NaN for {ticker} (attempt {attempt + 1})"
                )
            else:
                normalized = _normalize_ohlcv(df)
                # Drop trailing rows with NaN Close - during pre-market hours
                # yfinance returns a today-dated stub row with NaN prices for
                # the bar that has not yet opened. Caching the stub would
                # freeze the chart's tail at NaN until the day_stamp flips at
                # 16:00 ET. Historical bars never have NaN Close, so trimming
                # to the last valid Close index is safe.
                last_valid = normalized["Close"].last_valid_index()
                if last_valid is None:
                    last_error = DataFetchError(
                        f"All-NaN Close after normalize for {ticker} (attempt {attempt + 1})"
                    )
                else:
                    normalized = normalized.loc[:last_valid]
                    if not intraday_bypass:
                        cache.set(cache_key, normalized)
                    return normalized
        except Exception as exc:  # YFRateLimitError, YFDataException, network, etc.
            last_error = exc

        # Sleep on every failed attempt EXCEPT the last - no point sleeping
        # right before giving up and switching to Stooq.
        if attempt < _MAX_RETRIES - 1:
            time.sleep((2**attempt) + random.random() * 2)

    # Fallback: direct Stooq CSV download.
    logger.info("falling back to Stooq for %s", ticker)
    try:
        stooq_ticker = _to_stooq_ticker(ticker)
        df = _fetch_stooq_csv(stooq_ticker, start, end)
        if df is None or df.empty or df["Close"].isna().all():
            # Treat empty/all-NaN as a fallback failure; the outer handler
            # below will log diagnostic detail and surface the polished
            # user-facing message.
            raise DataFetchError(
                f"No data available for {ticker} from Stooq or yfinance; "
                f"check the ticker and date range."
            )
        normalized = _normalize_ohlcv(df)
        if not intraday_bypass:
            cache.set(cache_key, normalized)
        return normalized
    except Exception as exc:
        # Catch ALL fallback failures (DataFetchError with diagnostic detail
        # like "Stooq returned HTML for X..." AND unexpected exceptions).
        # Log full diagnostic context server-side, then surface a single
        # consistent polished message to the user regardless of failure mode.
        logger.warning(
            "Fetch failed for %s; yfinance_error=%r stooq_error=%r",
            ticker,
            last_error,
            exc,
        )
        raise DataFetchError(f"Could not fetch data for {ticker}. Please retry shortly.") from exc


_NYSE = ZoneInfo("America/New_York")
_NYSE_CAL = mcal.get_calendar("XNYS")


@functools.lru_cache(maxsize=64)
def _market_close_et(day_iso: str) -> datetime | None:
    """Return today's NYSE close as a tz-aware ET datetime, or None if not a trading day.

    Uses the NYSE calendar schedule, which automatically encodes both
    daylight-saving shifts AND half-day closes (13:00 ET on day-after-
    Thanksgiving, Christmas Eve, July 3 when on a weekday). Cached so the
    schedule lookup costs once per (process, day).
    """
    try:
        sched = _NYSE_CAL.schedule(start_date=day_iso, end_date=day_iso)
    except Exception:
        return None
    if sched.empty:
        return None
    return sched.iloc[0]["market_close"].tz_convert(_NYSE).to_pydatetime()


def trading_day_stamp() -> str:
    """Return a stamp that bumps once per NY day after that day's NYSE close.

    Uses pandas_market_calendars to read the actual close time - handles DST
    (EDT close 20:00 UTC, EST close 21:00 UTC) and half-day closes (13:00 ET
    on day-after-Thanksgiving, Christmas Eve, July 3) without hardcoding any
    hour. On weekends and full-day holidays there is no bar to wait for, so
    the stamp moves to today immediately.

    Before today's close the stamp is yesterday's ISO date (so today's
    still-forming bar isn't cached as final). After close it flips to today.
    """
    now_et = datetime.now(_NYSE)
    today = now_et.date()
    close_et = _market_close_et(today.isoformat())
    if close_et is None:
        # Non-trading day (weekend / full holiday) - nothing in-progress.
        return today.isoformat()
    if now_et >= close_et:
        return today.isoformat()
    return (today - timedelta(days=1)).isoformat()


def _market_is_open_now() -> bool:
    """True if right now is between today's market open and close (or pre-open early).

    Conservatively returns True from 00:00 ET up until today's close - this is
    enough for the intraday cache-bypass decision: if the requested range
    includes today AND we're not yet past close, the data is still changing.
    """
    now_et = datetime.now(_NYSE)
    close_et = _market_close_et(now_et.date().isoformat())
    return close_et is not None and now_et < close_et


def _to_stooq_ticker(ticker: str) -> str:
    """Map a yfinance ticker to Stooq's symbol convention.

    yfinance uses `.L` for London-listed equities; Stooq uses `.UK`. US tickers
    work bare on Stooq, so pass them through unchanged.
    """
    if ticker.endswith(".L"):
        return ticker[:-2] + ".UK"
    return ticker


def _fetch_stooq_csv(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Download a daily OHLCV CSV directly from Stooq.

    Stooq's CSV endpoint is case-sensitive on the ticker (lowercase) and
    accepts inclusive YYYYMMDD date bounds. Returns a DataFrame indexed by Date,
    sorted ascending, with TitleCase columns.
    """
    # URL-encode params so a ticker like "foo&i=h" cannot smuggle extra
    # parameters into the request. Defence-in-depth on top of app.py's
    # ticker regex.
    params = urllib.parse.urlencode(
        {
            "s": ticker.lower(),
            "d1": start.strftime("%Y%m%d"),
            "d2": end.strftime("%Y%m%d"),
            "i": "d",
        }
    )
    url = f"{_STOOQ_CSV_URL}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    # Cap the read at _STOOQ_MAX_BYTES so a misbehaving response cannot fill
    # memory. read(N+1) lets us detect overruns without buffering more than +1B.
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read(_STOOQ_MAX_BYTES + 1)
    if len(body) > _STOOQ_MAX_BYTES:
        raise DataFetchError(f"Stooq response for {ticker} exceeded {_STOOQ_MAX_BYTES} bytes")

    # Stooq returns HTTP 200 with an HTML error page when rate-limited or when
    # the symbol is unknown - pd.read_csv would either fail cryptically or
    # parse a single garbage row. Sniff the first bytes to fail loudly instead.
    preview = body[:200].lstrip().lower()
    if preview.startswith(b"<") or b"<html" in preview:
        raise DataFetchError(
            f"Stooq returned HTML for {ticker} (likely rate-limited or symbol unknown)"
        )

    try:
        df = pd.read_csv(io.BytesIO(body), parse_dates=["Date"])
    except ValueError as exc:
        # Missing 'Date' column or otherwise malformed payload (e.g. one-line
        # 'No data' response with no header).
        raise DataFetchError(
            f"Stooq CSV for {ticker} is malformed or missing 'Date' column: {exc}"
        ) from exc

    if df.empty or "Date" not in df.columns:
        raise DataFetchError(
            f"Stooq CSV for {ticker} is empty or missing 'Date' column; "
            f"check the ticker symbol and date range."
        )

    df = df.set_index("Date").sort_index(ascending=True)
    return df


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce arbitrary OHLCV frame to TitleCase OHLCV-only, tz-naive index."""
    # TitleCase every column so Stooq's already-TitleCase and yfinance's
    # already-TitleCase both pass through cleanly.
    renamed = df.rename(columns={c: str(c).title() for c in df.columns})
    # Keep only the canonical five; drop Dividends, Stock Splits, etc.
    keep = [c for c in _OHLCV_COLUMNS if c in renamed.columns]
    out = renamed[keep].copy()
    # yfinance returns tz-aware index (America/New_York); Stooq returns naive.
    # Strip tz so downstream code never needs to special-case the source.
    if isinstance(out.index, pd.DatetimeIndex) and out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    return out
