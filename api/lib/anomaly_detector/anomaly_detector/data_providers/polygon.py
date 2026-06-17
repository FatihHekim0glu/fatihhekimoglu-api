"""Real Polygon.io EOD data provider.

Fetches daily adjusted-close bars from Polygon's aggregates REST API::

    GET https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}
        ?adjusted=true&sort=asc&limit=50000&apiKey=...

and assembles a wide ``date x ticker`` panel of adjusted closes (inner-joined
across tickers, so only dates present for *every* requested ticker survive).

Design notes
------------
- The API key is resolved (in order) from an explicit ``api_key`` argument, the
  ``POLYGON_API_KEY`` environment variable, or a ``.env`` file in the repo root.
- ``import httpx`` is LAZY (inside :meth:`PolygonProvider.fetch`); a urllib
  fallback is used when httpx is absent, so the ``data`` extra is optional and
  importing this module has no side effects.
- HTTP ``429 Too Many Requests`` is retried with exponential backoff.
- Network failures raise; callers (see :func:`anomaly_detector.data.get_prices`) decide
  whether to fall through to the yfinance -> stooq -> synthetic chain.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from anomaly_detector._exceptions import ValidationError

#: Polygon aggregates ("aggs") REST endpoint template.
_AGGS_URL = (
    "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
    "?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
)

#: Environment variable that holds the Polygon API key.
_API_KEY_ENV = "POLYGON_API_KEY"


def _load_api_key_from_dotenv() -> str | None:
    """Best-effort read of ``POLYGON_API_KEY`` from a ``.env`` in the repo root.

    Walks up from this file looking for a ``.env`` and parses ``KEY=VALUE`` lines
    (ignoring blanks, comments, and an optional ``export`` prefix). Returns the
    value if found, else ``None``. Never raises on a malformed file.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / ".env"
        if not candidate.is_file():
            continue
        try:
            for raw_line in candidate.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].lstrip()
                key, _, value = line.partition("=")
                if key.strip() == _API_KEY_ENV:
                    return value.strip().strip("'\"")
        except OSError:
            return None
        # Found a .env but no key in it: stop searching further up.
        return None
    return None


def _resolve_api_key(explicit: str | None) -> str:
    """Resolve the API key from (in order) an explicit arg, env var, or ``.env``."""
    if explicit:
        return explicit
    env_key = os.environ.get(_API_KEY_ENV)
    if env_key:
        return env_key
    dotenv_key = _load_api_key_from_dotenv()
    if dotenv_key:
        return dotenv_key
    raise ValidationError(
        "PolygonProvider: no API key. Pass api_key=..., set the "
        f"{_API_KEY_ENV} environment variable, or add it to a .env file."
    )


class PolygonProvider:
    """Fetch daily adjusted-close bars from the Polygon.io aggregates REST API.

    Parameters
    ----------
    api_key:
        Explicit Polygon API key. When omitted, the key is read from the
        ``POLYGON_API_KEY`` environment variable, then from a ``.env`` file in
        the repo root.
    max_retries:
        How many times to retry a single request after an HTTP ``429`` before
        giving up.
    backoff_base:
        Base seconds for exponential backoff between ``429`` retries (attempt
        ``i`` sleeps ``backoff_base * 2**i``).
    timeout:
        Per-request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        max_retries: int = 3,
        backoff_base: float = 0.5,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = _resolve_api_key(api_key)
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def fetch(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        """Fetch a wide panel of adjusted closes, inner-joined across tickers.

        LAZY IMPORT: ``httpx`` (the ``data`` extra) is imported inside this
        method; when it is absent a stdlib ``urllib`` fallback is used. Either
        way, importing this module touches no network.

        Parameters
        ----------
        tickers:
            Asset symbols to fetch (e.g. ``["AAPL", "MSFT"]``).
        start, end:
            Inclusive date range.

        Returns
        -------
        pandas.DataFrame
            Adjusted closes with rows = date and columns = ticker (in request
            order), inner-joined so every row has a value for every ticker.

        Raises
        ------
        ValidationError
            If ``tickers`` is empty or ``end <= start``.
        """
        symbols = list(tickers)
        if len(symbols) == 0:
            raise ValidationError("PolygonProvider.fetch: tickers must be non-empty.")
        if end <= start:
            raise ValidationError(
                f"PolygonProvider.fetch: end ({end}) must be after start ({start})."
            )

        columns: dict[str, pd.Series] = {}
        for ticker in symbols:
            payload = self._get_json(ticker, start, end)
            columns[ticker] = self._series_from_payload(payload, ticker)

        # Inner-join across tickers: only dates present for every ticker survive.
        frame = pd.concat(columns, axis=1, join="inner")
        # Preserve request order and ensure a clean float64 panel.
        frame = frame.reindex(columns=symbols)
        frame.index = pd.to_datetime(frame.index)
        frame = frame.sort_index()
        return frame.astype("float64")

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #
    def _url(self, ticker: str, start: date, end: date) -> str:
        return _AGGS_URL.format(
            ticker=ticker,
            start=start.isoformat(),
            end=end.isoformat(),
            api_key=self._api_key,
        )

    def _get_json(self, ticker: str, start: date, end: date) -> dict[str, Any]:
        """GET the aggregates JSON for one ticker, retrying on HTTP 429.

        Tries ``httpx`` first (lazy import) and falls back to ``urllib`` when it
        is unavailable. Raises on a non-retryable HTTP error or after exhausting
        the ``429`` retry budget.
        """
        url = self._url(ticker, start, end)

        try:
            import httpx  # lazy: the ``data`` extra
        except ImportError:
            return self._get_json_urllib(url)

        import time

        last_status: int | None = None
        with httpx.Client(timeout=self.timeout) as client:
            for attempt in range(self.max_retries + 1):
                response = client.get(url)
                last_status = response.status_code
                if response.status_code == 429:
                    if attempt < self.max_retries:
                        time.sleep(self.backoff_base * (2**attempt))
                        continue
                    break
                response.raise_for_status()
                data = response.json()
                return dict(data)
        raise RuntimeError(
            f"PolygonProvider: HTTP {last_status} for {ticker} after {self.max_retries} retries."
        )

    def _get_json_urllib(self, url: str) -> dict[str, Any]:
        """stdlib fallback GET (no httpx), retrying on HTTP 429."""
        import time
        import urllib.error
        import urllib.request

        last_status: int | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                    raw = resp.read()
                data = json.loads(raw)
                return dict(data)
            except urllib.error.HTTPError as exc:
                last_status = exc.code
                if exc.code == 429 and attempt < self.max_retries:
                    time.sleep(self.backoff_base * (2**attempt))
                    continue
                raise
        raise RuntimeError(f"PolygonProvider: HTTP {last_status} after {self.max_retries} retries.")

    @staticmethod
    def _series_from_payload(payload: dict[str, Any], ticker: str) -> pd.Series:
        """Parse a Polygon aggregates payload into a date-indexed close Series.

        Polygon returns ``results`` as a list of bar dicts with ``t`` (epoch ms,
        UTC) and ``c`` (adjusted close; ``adjusted=true`` was requested).
        """
        results = payload.get("results") or []
        if not results:
            raise ValueError(
                f"PolygonProvider: no results for {ticker} (status={payload.get('status')!r})."
            )
        timestamps = [pd.Timestamp(bar["t"], unit="ms").normalize() for bar in results]
        closes = [float(bar["c"]) for bar in results]
        series = pd.Series(closes, index=pd.DatetimeIndex(timestamps), name=ticker)
        # Guard against duplicate timestamps; keep the last bar for a given day.
        series = series[~series.index.duplicated(keep="last")]
        return series.astype("float64")
