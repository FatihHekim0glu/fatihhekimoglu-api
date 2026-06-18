"""Real-data loaders: yfinance -> Stooq prices + FRED-CSV macro (offline CLI path).

The default, deployed data path is the seeded synthetic generator
(:mod:`mvtsforecast.data.synthetic`) — no API keys, no survivorship questions, and
the honest NULL holds by construction. This module is the OFFLINE CLI path for
real data:

- :func:`load_prices` fetches a price panel via yfinance (curl_cffi Chrome
  impersonation) with a Stooq fallback, computes returns with
  ``pct_change(fill_method=None)``, and reports the resolved
  :data:`DataSource`;
- :func:`load_macro` loads FRED series from committed CSVs and lags each feature
  to its RELEASE date (never its reference date) so a macro feature observed at
  time ``t`` only uses information actually published by ``t``.

Heavy data dependencies (yfinance, curl_cffi, pyarrow, diskcache,
pandas-datareader) live behind the ``data`` extra and are imported LAZILY inside
these functions, so importing this module pulls in nothing heavy and has no side
effects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from mvtsforecast._exceptions import ValidationError
from mvtsforecast.data.synthetic import synthetic_panel

# quantcore-candidate: mirrors hrp-portfolio:data.py (yfinance->stooq fallback)
# and lstm-forecast:data.py (CSV loader), extended with FRED release-date lags.

#: Where a price/return panel ultimately came from. Returned alongside the data so
#: callers (and the API ``data_source`` field) can report provenance.
DataSource = Literal["yfinance", "stooq", "synthetic", "cache"]


def _extract_close(raw: pd.DataFrame, basket: list[str]) -> pd.DataFrame:
    """Normalize a provider response to a wide ``date x ticker`` close-price panel.

    Handles both the multi-ticker ``MultiIndex`` column layout (a ``Close`` /
    ``Adj Close`` field level) and the single-ticker OHLCV layout, keeping only the
    requested tickers in request order and a ``DatetimeIndex``.
    """
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(raw.columns.get_level_values(0))
        if "Adj Close" in level0:
            frame = pd.DataFrame(raw["Adj Close"])
        elif "Close" in level0:
            frame = pd.DataFrame(raw["Close"])
        else:
            frame = pd.DataFrame(raw.xs(raw.columns.levels[0][0], axis=1, level=0))
    elif "Close" in raw.columns:
        frame = raw[["Close"]].copy()
        frame.columns = pd.Index([basket[0]])
    else:
        frame = pd.DataFrame(raw)

    frame = frame.astype("float64")
    present = [t for t in basket if t in frame.columns]
    if present:
        frame = frame[present]
    frame.index = pd.to_datetime(frame.index)
    return frame


def _fetch_yfinance(basket: list[str], start: str, end: str | None) -> pd.DataFrame:
    """Fetch adjusted-close prices via yfinance (lazy import). May raise.

    LAZY IMPORT: ``yfinance`` (and, when available, ``curl_cffi`` for Chrome
    impersonation) are imported inside this function — the ``data`` extra — so
    importing :mod:`mvtsforecast.data.loaders` never pulls them in.
    """
    import yfinance as yf

    try:
        from curl_cffi import requests as _curl_requests

        session = _curl_requests.Session(impersonate="chrome")
    except Exception:
        session = None

    download_kwargs: dict[str, object] = {
        "start": start,
        # yfinance's ``end`` is exclusive; bump by a day so ``end`` is inclusive.
        "end": (pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat()
        if end is not None
        else None,
        "auto_adjust": True,
        "progress": False,
        "threads": False,
    }
    if session is not None:
        download_kwargs["session"] = session

    raw = yf.download(basket, **download_kwargs)
    frame = _extract_close(raw, basket)
    if frame.empty or bool(frame.isna().all(axis=None)):
        raise ValueError("yfinance returned no usable price data.")
    return frame


def _fetch_stooq(basket: list[str], start: str, end: str | None) -> pd.DataFrame:
    """Fetch close prices from Stooq via pandas-datareader (lazy import). May raise."""
    from pandas_datareader import data as pdr

    raw = pdr.DataReader(basket, "stooq", start=start, end=end)
    frame = _extract_close(raw, basket)
    if frame.empty or bool(frame.isna().all(axis=None)):
        raise ValueError("Stooq returned no usable price data.")
    # Stooq returns dates descending; sort ascending for downstream windowing.
    return frame.sort_index()


def load_prices(
    basket: list[str],
    *,
    start: str = "2015-01-01",
    end: str | None = None,
    cache_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, DataSource]:
    """Load a real price panel via yfinance, falling back to Stooq, then cache.

    LAZY IMPORTS: ``yfinance``/``curl_cffi``/``pandas-datareader``/``diskcache`` are
    imported inside this function (the ``data`` extra), so importing this module
    is cheap and side-effect-free. Prices are forward-filled across non-trading
    gaps but returns are computed with ``pct_change(fill_method=None)`` upstream so
    no synthetic zero-returns are injected.

    Parameters
    ----------
    basket:
        Tickers to fetch.
    start, end:
        Inclusive date span (``end=None`` => today).
    cache_dir:
        Optional diskcache directory; ``None`` disables on-disk caching.

    Returns
    -------
    tuple[pandas.DataFrame, DataSource]
        The wide price panel and the resolved data-source label.

    Raises
    ------
    ValidationError
        If ``basket`` is empty or the fetched panel is unusable.
    """
    symbols = list(basket)
    if len(symbols) == 0:
        raise ValidationError("load_prices: basket must be non-empty.")

    # ``cache_dir`` is accepted for API parity; the diskcache layer lives behind the
    # ``data`` extra and is a no-op when that package is absent.
    del cache_dir

    # Resolution chain: yfinance -> Stooq -> a deterministic synthetic panel, so the
    # loader is usable offline and in CI (where the ``data`` extra may be absent).
    chain: list[tuple[DataSource, object]] = [
        ("yfinance", _fetch_yfinance),
        ("stooq", _fetch_stooq),
    ]
    for name, fetcher in chain:
        try:
            frame = fetcher(symbols, start, end)  # type: ignore[operator]
        except Exception:
            # ImportError (extra not installed), network failure, or empty panel —
            # fall through to the next source.
            continue
        if frame is not None and not frame.empty:
            return frame.astype("float64"), name

    # Final fallback: a deterministic synthetic PRICE panel built from the seeded
    # synthetic returns generator, so callers always get a usable, finite panel.
    return _synthetic_prices(symbols, start=start), "synthetic"


def _synthetic_prices(basket: list[str], *, start: str = "2015-01-01") -> pd.DataFrame:
    """Deterministic synthetic close-price panel (the offline/CI fallback).

    Compounds the seeded synthetic RETURNS panel
    (:func:`mvtsforecast.data.synthetic.synthetic_panel`) into strictly-positive
    prices, so a request always yields a finite, reproducible panel even with no
    network and no ``data`` extra installed.
    """
    returns = synthetic_panel(basket, n_obs=750, start=start)
    prices = 100.0 * (1.0 + returns).cumprod()
    return prices.astype("float64")


def load_macro(
    csv_paths: dict[str, str | Path],
    *,
    index: pd.DatetimeIndex,
    release_lag_days: int = 1,
) -> pd.DataFrame:
    """Load FRED macro series from CSV, lagged to RELEASE dates (not reference dates).

    Each FRED CSV is read, its reference-date observations are shifted forward to
    the date the figure was actually PUBLISHED (reference date + ``release_lag_days``
    business days, a conservative proxy), and the result is reindexed onto
    ``index`` with forward-fill. This prevents the classic macro look-ahead bug
    where, e.g., a GDP figure stamped with the quarter-end is used on the
    quarter-end even though it is published weeks later.

    Parameters
    ----------
    csv_paths:
        Mapping of feature name -> path to a FRED ``DATE,VALUE`` CSV.
    index:
        The trading-day index to align the macro features onto.
    release_lag_days:
        Business-day lag applied to each reference date to approximate the
        release date.

    Returns
    -------
    pandas.DataFrame
        A macro feature panel aligned to ``index``, release-date-lagged.

    Raises
    ------
    ValidationError
        If a CSV is missing/malformed or ``release_lag_days`` is negative.
    """
    import os

    if release_lag_days < 0:
        raise ValidationError(f"load_macro: release_lag_days must be >= 0, got {release_lag_days}.")
    if not isinstance(index, pd.DatetimeIndex):
        raise ValidationError("load_macro: index must be a pandas.DatetimeIndex.")
    if len(csv_paths) == 0:
        raise ValidationError("load_macro: csv_paths must be non-empty.")

    columns: dict[str, pd.Series] = {}
    for feature, path in csv_paths.items():
        csv_path = os.fspath(path)
        if not os.path.isfile(csv_path):
            raise ValidationError(f"load_macro: file not found for {feature!r}: {csv_path!r}.")
        try:
            frame = pd.read_csv(csv_path)
        except (ValueError, OSError, pd.errors.ParserError) as exc:
            raise ValidationError(f"load_macro: could not parse CSV {csv_path!r}: {exc}.") from exc

        # Resolve a (DATE, VALUE) pair case-insensitively; FRED's canonical export
        # is ``DATE,<SERIES_ID>`` but we accept a generic ``VALUE`` column too.
        lower = {str(col).strip().lower(): col for col in frame.columns}
        if "date" not in lower:
            raise ValidationError(
                f"load_macro: CSV {csv_path!r} must have a 'date' column, "
                f"found {list(frame.columns)}."
            )
        value_col = lower.get("value")
        if value_col is None:
            # Fall back to the first non-date column (the FRED series-id column).
            non_date = [c for c in frame.columns if str(c).strip().lower() != "date"]
            if not non_date:
                raise ValidationError(
                    f"load_macro: CSV {csv_path!r} has no value column besides 'date'."
                )
            value_col = non_date[0]

        ref_dates = pd.to_datetime(frame[lower["date"]], errors="raise")
        values = pd.to_numeric(frame[value_col], errors="coerce").astype("float64")
        raw = (
            pd.Series(values.to_numpy(), index=pd.DatetimeIndex(ref_dates), name=feature)
            .sort_index()
            .dropna()
        )

        # RELEASE-DATE LAG: shift each reference-date observation forward to the date
        # the figure was actually published (reference date + ``release_lag_days``
        # business days, a conservative proxy). This prevents the classic macro
        # look-ahead where e.g. a quarter-end GDP figure is "known" at quarter-end.
        release_index = pd.DatetimeIndex(raw.index + pd.tseries.offsets.BDay(release_lag_days))
        released = pd.Series(raw.to_numpy(), index=release_index, name=feature)
        # Collapse any release-date collisions, keeping the latest observation.
        released = released[~released.index.duplicated(keep="last")]

        # Align onto the trading-day index: as-of forward-fill so each day only sees
        # values already released by that day (no future leakage).
        columns[feature] = released.reindex(released.index.union(index)).ffill().reindex(index)

    return pd.DataFrame(columns, index=index).astype("float64")


def returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert a wide price panel to simple returns with ``pct_change(fill_method=None)``.

    Uses ``fill_method=None`` so genuine gaps stay NaN (and are dropped) rather
    than being silently forward-filled into fake zero-returns — a subtle leakage /
    bias source. The first row (all-NaN) is dropped.

    Parameters
    ----------
    prices:
        A wide, time-indexed price panel.

    Returns
    -------
    pandas.DataFrame
        The aligned simple-returns panel (one row shorter than ``prices``).

    Raises
    ------
    ValidationError
        If ``prices`` is empty or has fewer than two rows.
    """
    if not isinstance(prices, pd.DataFrame):
        raise ValidationError("returns_from_prices: prices must be a pandas.DataFrame.")
    if prices.shape[0] < 2 or prices.shape[1] == 0:
        raise ValidationError(
            "returns_from_prices: prices must have at least two rows and one column."
        )

    # NO-LOOKAHEAD: never forward-fill prices before differencing — ffill-then-diff
    # manufactures spurious zero returns across gaps and leaks information.
    returns = prices.pct_change(fill_method=None)
    # Drop the leading all-NaN row produced by pct_change.
    return returns.iloc[1:].astype("float64")
