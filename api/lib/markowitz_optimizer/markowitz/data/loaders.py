"""High-level loaders and return / statistics utilities.

These functions form the public entry-point used by the rest of the
``markowitz`` package. They orchestrate the provider stack, enforce
integrity invariants, and convert raw prices into the return-and-stat
representations consumed by optimizers.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Iterable
from typing import cast

import numpy as np
import pandas as pd

from ._periods import PERIODS_PER_YEAR
from .exceptions import (
    DataIntegrityError,
    EmptyDataError,
    InsufficientDataError,
)
from .providers import PriceProvider

DateLike = str | _dt.date | pd.Timestamp


def _periods_per_year(frequency: str) -> int:
    try:
        return PERIODS_PER_YEAR[frequency]
    except KeyError as exc:
        raise ValueError(
            f"unknown frequency {frequency!r}; expected one of {sorted(PERIODS_PER_YEAR)}"
        ) from exc


def _coerce_tickers(tickers: str | Iterable[str]) -> list[str]:
    if isinstance(tickers, str):
        return [tickers]
    out = [str(t) for t in tickers]
    if not out:
        raise ValueError("tickers must be a non-empty iterable")
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate tickers detected in input: {out}")
    return out


def load_prices(
    tickers: str | Iterable[str],
    start: DateLike,
    end: DateLike,
    *,
    provider: PriceProvider | None = None,
    frequency: str = "1d",
    min_history_sessions: int = 252,
) -> pd.DataFrame:
    """Fetch adjusted close prices for ``tickers`` as a wide DataFrame.

    Each column is a ticker; the index is a tz-naive, monotonic,
    unique :class:`~pandas.DatetimeIndex` aligned across tickers via an
    outer join. Tickers without at least ``min_history_sessions``
    non-NaN observations raise :class:`InsufficientDataError`.
    """
    if provider is None:
        # Late import so missing optional deps don't break test collection.
        from .providers import YFinanceProvider  # noqa: PLC0415

        provider = YFinanceProvider()

    ticker_list = _coerce_tickers(tickers)
    frames: dict[str, pd.Series] = {}
    for tkr in ticker_list:
        frame = provider.fetch(tkr, start, end, frequency=frequency)
        if "close" not in frame.columns:
            raise DataIntegrityError(
                f"provider {provider.name!r} returned unexpected columns "
                f"{list(frame.columns)} for {tkr}"
            )
        series = frame["close"].astype("float64")
        non_na = int(series.notna().sum())
        if non_na < min_history_sessions:
            raise InsufficientDataError(
                f"{tkr} has {non_na} observations; need >= {min_history_sessions}"
            )
        frames[tkr] = series

    combined = pd.concat(frames, axis=1)
    combined.columns = pd.Index(list(frames.keys()))
    combined = combined.sort_index()
    if combined.index.has_duplicates:
        combined = combined.loc[~combined.index.duplicated(keep="last")]
    assert isinstance(combined, pd.DataFrame)
    return combined


def compute_returns(
    prices: pd.DataFrame,
    *,
    method: str = "simple",
    dropna: bool = True,
) -> pd.DataFrame:
    """Convert a price panel into period-over-period returns.

    ``method="simple"`` is the Markowitz default. ``method="log"`` is
    provided for portfolio diagnostics and for users who prefer
    continuously compounded returns.
    """
    if prices.empty:
        raise EmptyDataError("prices is empty; cannot compute returns")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise DataIntegrityError("prices.index must be a DatetimeIndex")

    if method == "simple":
        rets = prices.pct_change(fill_method=None)
    elif method == "log":
        rets = np.log(prices / prices.shift(1))
    else:
        raise ValueError(f"unknown method {method!r}; expected 'simple' or 'log'")

    if dropna:
        rets = rets.dropna(how="any")
    if rets.empty:
        raise InsufficientDataError(
            "computed return frame is empty after dropna; check input price coverage"
        )
    return cast(pd.DataFrame, rets)


def align_universe(
    returns: pd.DataFrame,
    *,
    min_obs_per_ticker: int = 252,
    common_window: bool = True,
) -> pd.DataFrame:
    """Drop tickers with sparse history; optionally restrict to common window."""
    if returns.empty:
        raise EmptyDataError("returns is empty; nothing to align")

    obs_per_ticker = returns.notna().sum()
    survivors = obs_per_ticker[obs_per_ticker >= min_obs_per_ticker].index.tolist()
    if not survivors:
        raise InsufficientDataError(
            f"no ticker has >= {min_obs_per_ticker} observations; "
            f"max available is {int(obs_per_ticker.max())}"
        )

    aligned = returns[survivors]
    if common_window:
        aligned = aligned.dropna(how="any")
        if aligned.shape[0] < min_obs_per_ticker:
            raise InsufficientDataError(
                f"common window has {aligned.shape[0]} rows; need >= {min_obs_per_ticker}"
            )
    return cast(pd.DataFrame, aligned)


def summary_stats(
    returns: pd.DataFrame,
    *,
    frequency: str = "daily",
    annualize: bool = True,
) -> pd.DataFrame:
    """Per-asset summary statistics used as covariance / mean targets.

    Returned columns: ``mu_hat``, ``sigma_hat``, ``sharpe``, ``skew``,
    ``kurtosis``, ``n_obs``, ``min``, ``max``.

    When ``annualize=True`` (default), ``mu_hat`` is scaled by the
    periods-per-year factor and ``sigma_hat`` by its square root.
    """
    if returns.empty:
        raise EmptyDataError("returns is empty; cannot compute summary stats")

    n = _periods_per_year(frequency)
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    if annualize:
        mu = mu * n
        sigma = sigma * np.sqrt(n)

    sigma_arr = sigma.to_numpy()
    mu_arr = mu.to_numpy()
    safe = sigma_arr > 0
    ratio = np.full_like(mu_arr, np.nan, dtype="float64")
    np.divide(mu_arr, sigma_arr, out=ratio, where=safe)
    sharpe = pd.Series(ratio, index=mu.index, dtype="float64")

    stats = pd.DataFrame(
        {
            "mu_hat": mu.astype("float64"),
            "sigma_hat": sigma.astype("float64"),
            "sharpe": sharpe,
            "skew": returns.skew().astype("float64"),
            "kurtosis": returns.kurt().astype("float64"),
            "n_obs": returns.notna().sum().astype("int64"),
            "min": returns.min().astype("float64"),
            "max": returns.max().astype("float64"),
        }
    )
    return stats
