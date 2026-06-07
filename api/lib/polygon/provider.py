"""Polygon.io REST client with Supabase-backed OHLCV caching.

Public surface is :class:`PolygonProvider` plus :func:`make_provider` — the
factory selects the real provider when ``POLYGON_API_KEY`` is set, otherwise
falls back to the existing yfinance/Stooq pipeline so local dev keeps working
without a key.

Cache semantics
---------------
``get_eod`` checks ``platform.ohlcv_cache`` filtered on ``source='polygon'``.
If every expected trading day is present, the cached frame is returned. If any
day is missing, the FULL window is re-fetched from Polygon and upserted — this
keeps the request count low (one network call per range) at the cost of mildly
redundant cell writes, which Postgres handles via the (symbol,date) PK conflict.

Adjusted vs raw
---------------
We always request ``adjusted=true`` (split + dividend). Mirror behaviour in
the ``adjusted`` column so a future raw-feed consumer can coexist without
collisions.

Rate-limit / retry
------------------
A simple token bucket caps outbound traffic at ~100 requests/minute (Starter
tier ceiling). Each transient failure (429 / 5xx / network) triggers
exponential backoff with three attempts (1s, 2s, 4s + jitter). The bucket is
process-local; the assumption is that we run one API server instance per Fly
region — revisit when we scale out.
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from collections import deque
from datetime import date, datetime, timedelta
from typing import Any

import httpx
import pandas as pd

from .exceptions import (
    PolygonAuthError,
    PolygonDataError,
    PolygonError,
    PolygonRateLimitError,
)

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.polygon.io"
_OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
_MAX_ATTEMPTS = 3
_BACKOFF_BASE_SECONDS = 1.0
_STARTER_RPM = 100
_RATE_WINDOW_SECONDS = 60.0
_HTTP_TIMEOUT = 30.0


# ---------------------------------------------------------------------------
# Token bucket
# ---------------------------------------------------------------------------


class _TokenBucket:
    """Sliding-window rate limiter — at most ``rpm`` requests per 60 seconds."""

    def __init__(self, rpm: int = _STARTER_RPM) -> None:
        self._rpm = rpm
        self._window: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            cutoff = now - _RATE_WINDOW_SECONDS
            while self._window and self._window[0] < cutoff:
                self._window.popleft()
            if len(self._window) >= self._rpm:
                wait = _RATE_WINDOW_SECONDS - (now - self._window[0]) + 0.05
                if wait > 0:
                    time.sleep(wait)
                now = time.monotonic()
                cutoff = now - _RATE_WINDOW_SECONDS
                while self._window and self._window[0] < cutoff:
                    self._window.popleft()
            self._window.append(now)


# ---------------------------------------------------------------------------
# Main provider
# ---------------------------------------------------------------------------


class PolygonProvider:
    """Polygon.io REST adapter with point-in-time accuracy and shared caching."""

    def __init__(
        self,
        api_key: str | None = None,
        supabase_client: Any = None,
        session: httpx.Client | None = None,
        rpm: int = _STARTER_RPM,
    ) -> None:
        self._api_key = api_key or os.environ.get("POLYGON_API_KEY", "")
        if not self._api_key:
            raise PolygonAuthError(
                "POLYGON_API_KEY is required to instantiate PolygonProvider"
            )
        self._supabase = supabase_client
        self._owns_session = session is None
        self._session = session or httpx.Client(timeout=_HTTP_TIMEOUT)
        self._bucket = _TokenBucket(rpm=rpm)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_ticker_meta(self, ticker: str) -> dict[str, Any]:
        """Return the /v3/reference/tickers/{ticker} payload as a dict."""
        ticker = ticker.strip().upper()
        payload = self._request("GET", f"/v3/reference/tickers/{ticker}")
        results = payload.get("results")
        if not isinstance(results, dict):
            raise PolygonDataError(f"No reference data returned for {ticker}")
        return results

    def get_eod(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Return cached-or-fetched daily OHLCV for ``ticker`` in ``[start, end]``.

        Both bounds are inclusive. Output is TitleCase
        (Open/High/Low/Close/Volume) with a tz-naive ``DatetimeIndex`` named
        ``Date``. Close is split- and dividend-adjusted (Polygon
        ``adjusted=true``).
        """
        ticker = ticker.strip().upper()
        if start > end:
            raise ValueError(f"start ({start}) must be <= end ({end})")

        cached = self._read_cache(ticker, start, end)
        if cached is not None and not cached.empty:
            expected_days = self._expected_trading_days(start, end)
            if expected_days > 0 and len(cached) >= expected_days:
                return cached

        fetched = self._fetch_aggs(ticker, start, end)
        if fetched.empty:
            return cached if cached is not None else fetched
        self._write_cache(ticker, fetched)
        if cached is not None and not cached.empty:
            merged = pd.concat([cached, fetched])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            return merged
        return fetched

    def get_grouped_daily(self, date_: date) -> pd.DataFrame:
        """Grouped-daily snapshot of every actively-traded US stock on ``date_``.

        Index is the ticker symbol; columns are TitleCase OHLCV. Used by the
        S&P 500 universe builder to know which symbols actually traded on a
        given historical date.
        """
        path = f"/v2/aggs/grouped/locale/us/market/stocks/{date_.isoformat()}"
        payload = self._request("GET", path, params={"adjusted": "true"})
        results = payload.get("results") or []
        if not results:
            return pd.DataFrame(columns=list(_OHLCV_COLUMNS))
        rows: dict[str, dict[str, float]] = {}
        for bar in results:
            symbol = bar.get("T")
            if not symbol:
                continue
            rows[symbol] = {
                "Open": float(bar.get("o", 0.0)),
                "High": float(bar.get("h", 0.0)),
                "Low": float(bar.get("l", 0.0)),
                "Close": float(bar.get("c", 0.0)),
                "Volume": float(bar.get("v", 0.0)),
            }
        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index.name = "ticker"
        return df

    def close(self) -> None:
        if self._owns_session:
            self._session.close()

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{_BASE_URL}{path}"
        merged_params = dict(params or {})
        merged_params["apiKey"] = self._api_key
        last_exc: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            self._bucket.acquire()
            try:
                response = self._session.request(method, url, params=merged_params)
            except httpx.HTTPError as exc:
                last_exc = exc
                self._sleep_backoff(attempt)
                continue
            status_code = response.status_code
            if status_code in (401, 403):
                raise PolygonAuthError(
                    f"Polygon auth rejected ({status_code}); check POLYGON_API_KEY"
                )
            if status_code == 429:
                last_exc = PolygonRateLimitError(
                    f"Polygon rate limit hit (attempt {attempt + 1}/{_MAX_ATTEMPTS})"
                )
                self._sleep_backoff(attempt)
                continue
            if status_code >= 500:
                last_exc = PolygonError(
                    f"Polygon {status_code} on {path} (attempt {attempt + 1}/{_MAX_ATTEMPTS})"
                )
                self._sleep_backoff(attempt)
                continue
            if status_code >= 400:
                raise PolygonError(f"Polygon {status_code} on {path}: {response.text[:200]}")
            try:
                return response.json()
            except ValueError as exc:
                raise PolygonDataError(f"Polygon returned non-JSON for {path}") from exc
        if isinstance(last_exc, PolygonRateLimitError):
            raise last_exc
        raise PolygonError(
            f"Polygon request failed after {_MAX_ATTEMPTS} attempts on {path}"
        ) from last_exc

    @staticmethod
    def _sleep_backoff(attempt: int) -> None:
        delay = _BACKOFF_BASE_SECONDS * (2 ** attempt) + random.uniform(0.0, 0.25)
        time.sleep(delay)

    # ------------------------------------------------------------------
    # Aggregates
    # ------------------------------------------------------------------

    def _fetch_aggs(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        path = f"/v2/aggs/ticker/{ticker}/range/1/day/{start.isoformat()}/{end.isoformat()}"
        payload = self._request(
            "GET",
            path,
            params={"adjusted": "true", "sort": "asc", "limit": 50_000},
        )
        results = payload.get("results") or []
        if not results:
            return pd.DataFrame(columns=list(_OHLCV_COLUMNS))
        records: list[dict[str, Any]] = []
        for bar in results:
            ts_ms = bar.get("t")
            if ts_ms is None:
                continue
            # Polygon's daily aggregates encode the trading day as the UTC
            # midnight timestamp of that calendar date. Do NOT shift to
            # America/New_York — that would move the bar back one day.
            records.append(
                {
                    "Date": pd.Timestamp(ts_ms, unit="ms", tz="UTC")
                    .tz_localize(None)
                    .normalize(),
                    "Open": float(bar.get("o", 0.0)),
                    "High": float(bar.get("h", 0.0)),
                    "Low": float(bar.get("l", 0.0)),
                    "Close": float(bar.get("c", 0.0)),
                    "Volume": float(bar.get("v", 0.0)),
                    "vwap": bar.get("vw"),
                    "trade_count": bar.get("n"),
                }
            )
        if not records:
            return pd.DataFrame(columns=list(_OHLCV_COLUMNS))
        df = pd.DataFrame.from_records(records).set_index("Date").sort_index()
        df.index = pd.DatetimeIndex(df.index, name="Date")
        return df

    # ------------------------------------------------------------------
    # Supabase cache
    # ------------------------------------------------------------------

    def _read_cache(
        self, ticker: str, start: date, end: date
    ) -> pd.DataFrame | None:
        if self._supabase is None:
            return None
        try:
            response = (
                self._supabase.schema("platform")
                .table("ohlcv_cache")
                .select("date, open, high, low, close, volume")
                .eq("symbol", ticker)
                .eq("source", "polygon")
                .gte("date", start.isoformat())
                .lte("date", end.isoformat())
                .order("date")
                .execute()
            )
        except Exception:  # noqa: BLE001 — cache is opportunistic
            logger.exception("ohlcv_cache read failed for %s (non-fatal)", ticker)
            return None
        rows = getattr(response, "data", None) or []
        if not rows:
            return None
        df = pd.DataFrame.from_records(rows)
        df["Date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        df = df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
        df = df[["Date", *_OHLCV_COLUMNS]].set_index("Date").sort_index()
        for col in _OHLCV_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.index = pd.DatetimeIndex(df.index, name="Date")
        return df

    def _write_cache(self, ticker: str, df: pd.DataFrame) -> None:
        if self._supabase is None or df.empty:
            return
        try:
            rows: list[dict[str, Any]] = []
            for ts, row in df.iterrows():
                day = ts.date() if hasattr(ts, "date") else ts
                record: dict[str, Any] = {
                    "symbol": ticker,
                    "date": day.isoformat(),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"]) if pd.notna(row["Volume"]) else 0,
                    "source": "polygon",
                    "adjusted": True,
                }
                if "vwap" in df.columns and pd.notna(row.get("vwap")):
                    record["vwap"] = float(row["vwap"])
                if "trade_count" in df.columns and pd.notna(row.get("trade_count")):
                    record["trade_count"] = int(row["trade_count"])
                rows.append(record)
            self._supabase.schema("platform").table("ohlcv_cache").upsert(
                rows, on_conflict="symbol,date"
            ).execute()
        except Exception:  # noqa: BLE001 — cache is opportunistic
            logger.exception("ohlcv_cache upsert failed for %s (non-fatal)", ticker)

    @staticmethod
    def _expected_trading_days(start: date, end: date) -> int:
        """Approximate count of US trading days in ``[start, end]``.

        Uses pandas business-day count, which ignores US holidays. Good enough
        for the cache-completeness check — when it overcounts by a holiday, the
        worst case is one extra Polygon round-trip.
        """
        if start > end:
            return 0
        return int(pd.bdate_range(start, end).size)


# ---------------------------------------------------------------------------
# Fallback (yfinance + Stooq)
# ---------------------------------------------------------------------------


class PolygonProviderFallback:
    """Drop-in fallback that mirrors :class:`PolygonProvider`'s public surface.

    Routes ``get_eod`` through the existing ``stock_dashboard.data.fetch_ohlcv``
    pipeline so local dev (and CI) keeps working without a Polygon key. Methods
    that have no yfinance equivalent (``get_grouped_daily``, ``get_ticker_meta``)
    raise :class:`PolygonError` so callers can degrade gracefully.
    """

    def __init__(self, supabase_client: Any = None) -> None:
        self._supabase = supabase_client

    def get_ticker_meta(self, ticker: str) -> dict[str, Any]:
        raise PolygonError(
            "get_ticker_meta requires POLYGON_API_KEY; fallback provider has no equivalent"
        )

    def get_eod(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        # Lazy import: stock_dashboard.data drags in yfinance, curl_cffi etc.
        # which we don't want at the top of the polygon module.
        from ..stock_dashboard import data as sd_data

        return sd_data.fetch_ohlcv(ticker, start, end)

    def get_grouped_daily(self, date_: date) -> pd.DataFrame:
        raise PolygonError(
            "get_grouped_daily requires POLYGON_API_KEY; no yfinance equivalent"
        )

    def close(self) -> None:  # parity with PolygonProvider
        return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_provider(
    supabase_client: Any = None,
) -> PolygonProvider | PolygonProviderFallback:
    """Return the real Polygon provider when an API key is configured, else fallback."""
    api_key = os.environ.get("POLYGON_API_KEY", "").strip()
    if api_key:
        return PolygonProvider(api_key=api_key, supabase_client=supabase_client)
    logger.info("POLYGON_API_KEY not set — using yfinance/Stooq fallback provider")
    return PolygonProviderFallback(supabase_client=supabase_client)


# Silence "imported but unused" for datetime/timedelta — kept for future use in
# rate-limit reset windows. (Removed if linter complains.)
_ = (datetime, timedelta)
