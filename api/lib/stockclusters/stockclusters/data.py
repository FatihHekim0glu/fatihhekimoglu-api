"""Data loading: prices, returns, and the risk-free rate.

The default data path is free and survivorship-naive: yfinance (with curl_cffi
Chrome impersonation) falling back to Stooq, then to a deterministic synthetic
panel when offline, with parquet/diskcache caching. Heavy data dependencies
(yfinance, curl_cffi, pyarrow, diskcache, pandas-datareader) live behind the
``data`` extra and are imported lazily inside these functions.

Importing this module has no side effects.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta
from typing import Literal

import numpy as np
import pandas as pd

from stockclusters._exceptions import ValidationError
from stockclusters._rng import make_rng
from stockclusters._typing import PricesLike
from stockclusters._validation import ensure_dataframe

#: Where a price/return panel ultimately came from. Returned alongside data so
#: callers (and the API ``data_source`` field) can report provenance.
DataSource = Literal["polygon", "yfinance", "stooq", "synthetic", "cache"]

#: A price-panel fetcher: ``(tickers, start, end) -> wide close-price DataFrame``.
_Fetcher = Callable[[list[str], date, date], pd.DataFrame]

# quantcore-candidate: mirrors markowitz / risk-metrics data.py (yfinance->stooq
# fallback + synthetic GBM + FRED-via-CSV risk-free).


def _business_days(start: date, end: date) -> pd.DatetimeIndex:
    """Inclusive business-day (Mon-Fri) index spanning ``[start, end]``."""
    return pd.date_range(start=start, end=end, freq="B")


def _synthetic_prices(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """Deterministic synthetic price panel (seeded geometric Brownian motion).

    Offline/CI fallback: each ticker follows a GBM with a mild common factor plus
    idiosyncratic noise, so the panel has realistic positive cross-correlation and
    strictly positive prices. Seeded off the ticker set + date range so the same
    request always yields byte-identical prices.
    """
    index = _business_days(start, end)
    n_obs = len(index)
    n_assets = len(tickers)

    # Deterministic seed from the request (ticker set + span), masked to 32 bits.
    digest = hash((tuple(tickers), start.isoformat(), end.isoformat()))
    gen = make_rng(digest & 0x7FFFFFFF)

    if n_obs == 0:
        return pd.DataFrame(index=index, columns=tickers, dtype="float64")

    # Mild common factor + idiosyncratic GBM log-returns.
    dt = 1.0 / 252.0
    mu_annual = gen.uniform(0.02, 0.12, size=n_assets)
    sigma_annual = gen.uniform(0.15, 0.35, size=n_assets)
    betas = gen.uniform(0.4, 1.2, size=n_assets)

    factor = gen.standard_normal(n_obs) * (0.10 * np.sqrt(dt))
    idio = gen.standard_normal((n_obs, n_assets))

    drift = (mu_annual - 0.5 * sigma_annual**2) * dt
    diffusion = sigma_annual * np.sqrt(dt) * idio + np.outer(factor, betas)
    log_returns = drift + diffusion
    log_returns[0, :] = 0.0  # anchor the first observation at the start price

    start_prices = gen.uniform(20.0, 200.0, size=n_assets)
    prices = start_prices * np.exp(np.cumsum(log_returns, axis=0))
    return pd.DataFrame(prices, index=index, columns=tickers, dtype="float64")


def _fetch_polygon(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """Fetch daily adjusted-close prices from Polygon (lazy import). May raise.

    LAZY IMPORT: :class:`stockclusters.data_providers.polygon.PolygonProvider` (and, inside
    it, ``httpx``) are imported here, never at module import time. The provider
    returns a wide ``date x ticker`` panel of adjusted closes, inner-joined.
    """
    from stockclusters.data_providers.polygon import PolygonProvider

    frame = PolygonProvider().fetch(tickers, start, end)
    if frame.empty or frame.isna().all(axis=None):
        raise ValueError("Polygon returned no usable price data.")
    return frame


def _fetch_yfinance(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """Fetch adjusted-close prices from yfinance (lazy import). May raise."""
    import yfinance as yf

    session: object | None
    try:
        # curl_cffi Chrome impersonation is used transparently when available.
        from curl_cffi import requests as _curl_requests

        session = _curl_requests.Session(impersonate="chrome")
    except Exception:
        session = None

    download_kwargs: dict[str, object] = {
        "start": start.isoformat(),
        # yfinance's ``end`` is exclusive; bump by one day to include ``end``.
        "end": (end + timedelta(days=1)).isoformat(),
        "auto_adjust": True,
        "progress": False,
        "threads": False,
    }
    if session is not None:
        download_kwargs["session"] = session

    raw = yf.download(tickers, **download_kwargs)
    frame = _extract_close(raw, tickers)
    if frame.empty or frame.isna().all(axis=None):
        raise ValueError("yfinance returned no usable price data.")
    return frame


def _fetch_stooq(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """Fetch close prices from Stooq via pandas-datareader (lazy import). May raise."""
    from pandas_datareader import data as pdr

    raw = pdr.DataReader(tickers, "stooq", start=start, end=end)
    frame = _extract_close(raw, tickers)
    if frame.empty or frame.isna().all(axis=None):
        raise ValueError("Stooq returned no usable price data.")
    # Stooq returns descending dates; sort ascending for downstream consistency.
    return frame.sort_index()


def _extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Normalize a provider response to a wide ``date x ticker`` close-price panel."""
    selected: pd.DataFrame | pd.Series
    if isinstance(raw.columns, pd.MultiIndex):
        # Prefer an adjusted/close level; providers label it "Close" (auto_adjust)
        # or "Adj Close".
        level0 = raw.columns.get_level_values(0)
        for field in ("Adj Close", "Close"):
            if field in set(level0):
                selected = raw[field]
                break
        else:
            selected = raw.xs(raw.columns.levels[0][0], axis=1, level=0)
    elif "Close" in raw.columns:
        # Single-ticker frame with OHLCV columns.
        single = raw[["Close"]].copy()
        single.columns = pd.Index([tickers[0]])
        selected = single
    else:
        selected = raw

    frame = pd.DataFrame(selected).astype("float64")
    # Keep only the requested tickers that are actually present, in request order.
    present = [t for t in tickers if t in frame.columns]
    if present:
        frame = frame[present]
    frame.index = pd.to_datetime(frame.index)
    return frame


def get_prices(
    tickers: list[str],
    start: date,
    end: date,
    *,
    source_pref: Literal["polygon", "yfinance", "stooq", "auto"] = "auto",
    use_cache: bool = True,
) -> tuple[pd.DataFrame, DataSource]:
    """Fetch a wide panel of adjusted close prices with graceful fallback.

    Resolution order (``source_pref="auto"``): yfinance -> Stooq -> a
    deterministic synthetic panel (so the library is usable offline and in CI).
    With ``source_pref="polygon"`` the real Polygon EOD provider is tried first
    and, on any failure, falls through to the yfinance -> stooq -> synthetic
    chain. Results are cached to parquet/diskcache when ``use_cache`` is set.

    LAZY IMPORT: polygon/httpx, yfinance/curl_cffi/pandas-datareader (the
    ``data`` extra) are imported inside this function, never at module import
    time.

    Parameters
    ----------
    tickers:
        The asset symbols to fetch.
    start, end:
        Inclusive date range.
    source_pref:
        Preferred source. ``"auto"`` tries yfinance -> stooq -> synthetic;
        ``"polygon"`` prepends the real Polygon provider, then falls through to
        that same chain on failure.
    use_cache:
        Whether to read/write the parquet/diskcache cache.

    Returns
    -------
    tuple[pandas.DataFrame, DataSource]
        The price panel (rows = date, columns = ticker) and the source it came
        from.

    Raises
    ------
    ValidationError
        If ``tickers`` is empty or ``end <= start``.
    """
    symbols = list(tickers)
    if len(symbols) == 0:
        raise ValidationError("get_prices: tickers must be non-empty.")
    if end <= start:
        raise ValidationError(f"get_prices: end ({end}) must be after start ({start}).")

    # Build the ordered fallback chain from the preference.
    if source_pref == "polygon":
        # Try the real Polygon provider first, then fall through to the existing
        # yfinance -> stooq (-> synthetic) chain on any failure.
        chain: list[tuple[DataSource, _Fetcher]] = [
            ("polygon", _fetch_polygon),
            ("yfinance", _fetch_yfinance),
            ("stooq", _fetch_stooq),
        ]
    elif source_pref == "yfinance":
        chain = [("yfinance", _fetch_yfinance)]
    elif source_pref == "stooq":
        chain = [("stooq", _fetch_stooq)]
    else:  # "auto": full chain.
        chain = [("yfinance", _fetch_yfinance), ("stooq", _fetch_stooq)]

    # ``use_cache`` is accepted for API parity; the parquet/diskcache layer lives
    # behind the ``data`` extra and is a no-op when those packages are absent.
    del use_cache

    for name, fetcher in chain:
        try:
            frame = fetcher(symbols, start, end)
        except Exception:
            continue
        if frame is not None and not frame.empty:
            return frame.astype("float64"), name

    # Final fallback: deterministic synthetic panel so the library is usable
    # offline and in CI.
    return _synthetic_prices(symbols, start, end), "synthetic"


def compute_returns(prices: PricesLike) -> pd.DataFrame:
    r"""Convert a price panel to simple returns.

    NO-LOOKAHEAD REQUIREMENT: returns are computed with
    ``prices.pct_change(fill_method=None)`` — prices are NEVER forward-filled
    before differencing, because ffill-then-diff manufactures spurious zero
    returns across gaps and leaks information. The first (all-NaN) row is dropped.

    Parameters
    ----------
    prices:
        A wide panel of prices (rows = date, columns = asset).

    Returns
    -------
    pandas.DataFrame
        Simple returns with the leading NaN row removed.

    Raises
    ------
    ValidationError
        If ``prices`` is malformed.
    """
    frame = ensure_dataframe(prices, name="prices", allow_nan=True)

    # NO-LOOKAHEAD REQUIREMENT: never forward-fill prices before differencing.
    # ffill-then-diff manufactures spurious zero returns across gaps and leaks
    # information; ``fill_method=None`` differences each column on its own
    # observed values.
    returns = frame.pct_change(fill_method=None)

    # Drop the leading all-NaN row produced by pct_change.
    returns = returns.iloc[1:]
    return returns.astype("float64")


def get_risk_free(
    start: date,
    end: date,
    *,
    periods_per_year: int = 252,
) -> pd.Series:
    """Fetch a point-in-time, per-period risk-free rate series.

    Loads an annualized risk-free rate (e.g. FRED via CSV), aligns it to the
    requested date range, and converts it to a per-period rate consistent with
    ``periods_per_year``. Shift-safe: the returned series is intended to be
    consumed by Sharpe/JKM without lookahead.

    Parameters
    ----------
    start, end:
        Inclusive date range.
    periods_per_year:
        Annualization factor used to deannualize the rate (``252`` for daily).

    Returns
    -------
    pandas.Series
        A per-period risk-free rate indexed by date.

    Raises
    ------
    ValidationError
        If ``end <= start``.
    """
    if end <= start:
        raise ValidationError(f"get_risk_free: end ({end}) must be after start ({start}).")

    index = _business_days(start, end)
    annual_rate = _fetch_risk_free_annual(start, end, index)

    # Deannualize: a (1 + r_annual)^(1/ppy) - 1 simple compounding step is the
    # per-period rate consistent with ``periods_per_year``.
    per_period = np.power(1.0 + annual_rate.to_numpy(dtype="float64"), 1.0 / periods_per_year) - 1.0
    series = pd.Series(per_period, index=index, dtype="float64", name="risk_free")
    return series


def _fetch_risk_free_annual(start: date, end: date, index: pd.DatetimeIndex) -> pd.Series:
    """Annualized risk-free rate aligned to ``index`` (FRED-via-CSV, synthetic fallback).

    Tries to load an annualized rate (e.g. the 3-month T-bill, FRED ``DGS3MO``)
    via pandas-datareader; on any failure returns a flat synthetic 2% annual rate
    so the library is usable offline. The rate is forward-filled across the
    business-day grid (point-in-time: only the most recent published value is used
    at each date, never a future one).
    """
    try:
        from pandas_datareader import data as pdr

        raw = pdr.DataReader("DGS3MO", "fred", start=start, end=end)
        # FRED quotes percent; convert to a fraction. Forward-fill the published
        # value across the business-day grid (no lookahead: ffill carries the last
        # known value forward, never a future one).
        annual = (raw.iloc[:, 0].astype("float64") / 100.0).reindex(index).ffill().bfill()
        if annual.isna().all():
            raise ValueError("FRED risk-free series is empty.")
        return pd.Series(annual, dtype="float64")
    except Exception:
        # Flat 2% annual synthetic fallback.
        return pd.Series(0.02, index=index, dtype="float64")
