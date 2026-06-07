"""OHLCV data loading with parquet cache.

Workflow
--------
1. On request, check the parquet cache (``data/cache/ohlcv/<TICKER>.parquet``).
2. If absent or stale, fetch from yfinance (``auto_adjust=True``) and write.
3. Run sanity checks on every loaded frame: continuity, OHLC consistency,
   day-over-day jump detection, NaN scan.
4. Return a tidy DataFrame indexed by tz-naive DatetimeIndex with columns
   ``Open``, ``High``, ``Low``, ``Close``, ``Volume``.

Stooq fallback is exposed via ``load_from_stooq`` so notebook 4 can perform a
cross-source agreement check on at least one ticker.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import time
import warnings
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR: Final[Path] = Path("data/cache/ohlcv")
_META_FILE: Final[str] = "_meta.json"
_JUMP_THRESHOLD: Final[float] = 0.5
_DEFAULT_TTL_HOURS: Final[float] = 24.0

# Ticker regex permits A-Z, 0-9, dot, hyphen, caret (covers BRK-B, BF.B, ^GSPC
# style variants). Length capped at 15 to defeat path traversal via cache filename.
_TICKER_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Z0-9.\-^]{1,15}$")


class DataQualityError(ValueError):
    """Raised when loaded OHLCV fails a sanity check that can't be auto-fixed."""


def _safe_ticker(ticker: str) -> str:
    """Normalise and validate a ticker for use in filenames and URLs."""
    cleaned = ticker.strip().upper()
    if not _TICKER_RE.fullmatch(cleaned):
        raise DataQualityError(f"invalid ticker symbol: {ticker!r}")
    return cleaned


@dataclass(frozen=True, slots=True)
class DataQualityReport:
    ticker: str
    n_rows: int
    first_date: pd.Timestamp
    last_date: pd.Timestamp
    n_gaps: int
    n_jumps: int
    n_nans: int
    warnings_emitted: tuple[str, ...]


def _meta_path(cache_dir: Path) -> Path:
    return cache_dir / _META_FILE


def _read_meta(cache_dir: Path) -> dict[str, dict[str, str | int]]:
    p = _meta_path(cache_dir)
    if not p.exists():
        return {}
    try:
        return dict(json.loads(p.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_meta(cache_dir: Path, meta: dict[str, dict[str, str | int]]) -> None:
    _meta_path(cache_dir).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def _is_stale(meta_entry: dict[str, str | int] | None, ttl_hours: float) -> bool:
    if meta_entry is None:
        return True
    fetched = meta_entry.get("last_fetched_utc")
    if not isinstance(fetched, str):
        return True
    try:
        ts = datetime.fromisoformat(fetched)
    except ValueError:
        return True
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    age_hours = (datetime.now(UTC) - ts).total_seconds() / 3600.0
    return age_hours > ttl_hours


def _yfinance_download(
    ticker: str,
    start: str | date,
    end: str | date,
    *,
    max_retries: int = 3,
    base_delay_sec: float = 1.5,
) -> pd.DataFrame:
    import yfinance as yf

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = yf.download(
                    ticker,
                    start=str(start),
                    end=str(end),
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                    actions=False,
                )
            if df is None or len(df) == 0:
                raise DataQualityError(f"yfinance returned no rows for {ticker}")
            return _normalise_yf_columns(df)
        except DataQualityError:
            raise
        except Exception as exc:
            last_err = exc
            time.sleep(base_delay_sec * (2**attempt))
    raise DataQualityError(
        f"yfinance failed for {ticker} after {max_retries} retries"
    ) from last_err


def _normalise_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)
    df = df.rename(columns=str.title)
    cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DataQualityError(f"missing columns from yfinance: {missing}")
    df = df[cols].copy()
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"
    return df.astype("float64")


def _normalise_stooq_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=str.title)
    cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DataQualityError(f"missing columns from Stooq: {missing}")
    df = df[cols].copy()
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"
    return df.astype("float64")


def validate(df: pd.DataFrame, ticker: str) -> DataQualityReport:
    """Run sanity checks. Raises ``DataQualityError`` on hard failures."""
    if df.empty:
        raise DataQualityError(f"{ticker}: empty frame")

    bdays = pd.bdate_range(df.index[0], df.index[-1])
    n_gaps = int(bdays.difference(df.index).size)

    # Split/dividend adjustment in yfinance introduces sub-bp rounding errors
    # that occasionally break OHLC ordering on adjusted prices. We allow a
    # 1bp relative tolerance — well below any cost we model. Only flag a bar
    # when Low *materially* exceeds min(O,C) (or High materially below
    # max(O,C)).
    tol = 1e-4
    oc_min = df[["Open", "Close"]].min(axis=1)
    oc_max = df[["Open", "Close"]].max(axis=1)
    inconsistent = (
        (df["Low"] > oc_min * (1.0 + tol))
        | (df["High"] < oc_max * (1.0 - tol))
        | (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
    )
    n_inconsistent = int(inconsistent.sum())
    if n_inconsistent > 0:
        bad_dates = df.index[inconsistent][:5].strftime("%Y-%m-%d").tolist()
        raise DataQualityError(
            f"{ticker}: {n_inconsistent} bars violate OHLC ordering or sign "
            f"(first offenders: {bad_dates})"
        )

    log_ret = np.log(df["Close"]).diff()
    jumps = log_ret[log_ret.abs() > _JUMP_THRESHOLD]
    n_jumps = int(jumps.size)

    n_nans = int(df.isna().sum().sum())

    warnings_emitted: list[str] = []
    if n_gaps > 5:
        warnings_emitted.append(f"{n_gaps} business-day gaps detected")
    if n_jumps > 0:
        warnings_emitted.append(f"{n_jumps} large day-over-day jumps (>{_JUMP_THRESHOLD} log-ret)")
    if n_nans > 0:
        warnings_emitted.append(f"{n_nans} NaN entries (will be dropped)")

    return DataQualityReport(
        ticker=ticker,
        n_rows=len(df),
        first_date=df.index[0],
        last_date=df.index[-1],
        n_gaps=n_gaps,
        n_jumps=n_jumps,
        n_nans=n_nans,
        warnings_emitted=tuple(warnings_emitted),
    )


def load_ohlcv(
    ticker: str,
    *,
    start: str | date = "2005-01-01",
    end: str | date | None = None,
    cache_dir: Path = _DEFAULT_CACHE_DIR,
    ttl_hours: float = _DEFAULT_TTL_HOURS,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load OHLCV for one ticker, hitting cache when fresh."""
    ticker = _safe_ticker(ticker)
    end = end or date.today().isoformat()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{ticker}.parquet"
    meta = _read_meta(cache_dir)

    cache_hit = (
        not force_refresh and cache_file.exists() and not _is_stale(meta.get(ticker), ttl_hours)
    )
    # df is initialised here so pyright can prove it's bound after either branch.
    # Mutating cache_hit inside the except clause is not enough for pyright to
    # narrow the second `if df is None` — using the Optional+sentinel pattern
    # makes the binding contract explicit.
    df: pd.DataFrame | None = None
    if cache_hit:
        try:
            df = pd.read_parquet(cache_file)
        except Exception as exc:
            # pyarrow can raise its own non-stdlib exceptions; the semantic
            # contract is "if the cache can't be read for any reason, refetch".
            _log.warning("corrupt cache for %s (%s); refetching", ticker, exc)
            cache_file.unlink(missing_ok=True)
            df = None
    if df is None:
        source = "yfinance"
        try:
            df = _yfinance_download(ticker, start, end)
        except Exception as yf_exc:
            _log.warning("yfinance failed for %s, falling back to Stooq: %s", ticker, yf_exc)
            try:
                df = _normalise_stooq_columns(load_from_stooq(ticker))
            except Exception as stooq_exc:
                raise yf_exc from stooq_exc
            source = "stooq"
        tmp = cache_file.with_suffix(".parquet.tmp")
        df.to_parquet(tmp)
        os.replace(tmp, cache_file)  # atomic on POSIX and Windows
        meta[ticker] = {
            "last_fetched_utc": datetime.now(UTC).isoformat(),
            "first_date": str(df.index[0].date()),
            "last_date": str(df.index[-1].date()),
            "n_rows": len(df),
            "source": source,
        }
        _write_meta(cache_dir, meta)

    assert df is not None  # narrowed by both branches above
    df = df.loc[str(start) : str(end)]
    # Validate BEFORE dropna so n_nans is reported and warnings surface.
    report = validate(df, ticker)
    for w in report.warnings_emitted:
        _log.warning("%s: %s", ticker, w)
    df = df.dropna(how="any")
    return df


def load_close(
    ticker: str,
    *,
    start: str | date = "2005-01-01",
    end: str | date | None = None,
    cache_dir: Path = _DEFAULT_CACHE_DIR,
) -> pd.Series:
    """Load the adjusted-close series for one ticker.

    Thin wrapper around :func:`load_ohlcv` that returns only the ``Close``
    column as a float64 ``pd.Series`` named after the ticker.

    Parameters
    ----------
    ticker : str
        US-equity ticker, e.g. ``"SPY"``. Validated and uppercased by
        :func:`_safe_ticker`.
    start, end : str | date | None
        ISO-formatted dates or ``date`` objects. ``end=None`` means today.
    cache_dir : Path
        Parquet cache root. Defaults to ``data/cache/ohlcv/`` relative to cwd.

    Returns
    -------
    pd.Series
        Adjusted close, indexed by tz-naive DatetimeIndex, float64, NaN-free.

    Raises
    ------
    DataQualityError
        Both yfinance and the Stooq fallback returned unusable data, OR the
        loaded frame failed the OHLC sanity checks in :func:`validate`.

    Examples
    --------
    >>> close = load_close("SPY", start="2020-01-01", end="2020-12-31")  # doctest: +SKIP
    >>> close.name  # doctest: +SKIP
    'SPY'
    """
    df = load_ohlcv(ticker, start=start, end=end, cache_dir=cache_dir)
    s = df["Close"].astype("float64")
    s.name = ticker
    return s


def load_from_stooq(ticker: str) -> pd.DataFrame:
    """Cross-source check via Stooq CSV (no API key required).

    Stooq tickers are usually ``<TICKER>.US`` for US equities.
    """
    ticker = _safe_ticker(ticker)
    stooq_symbol = f"{ticker.lower()}.us"
    query = urlencode({"s": stooq_symbol, "i": "d"})
    url = f"https://stooq.com/q/d/l/?{query}"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (ma-backtester)"})
    with urlopen(req, timeout=20) as resp:
        body = resp.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(body))
    if df.empty or "Date" not in df.columns:
        raise DataQualityError(f"Stooq returned no data for {ticker}")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return df.astype("float64")
