"""Risk-free rate utilities backed by FRED.

The wrapper exposes a tidy ``risk_free_rate`` Series indexed by tz-naive
midnight timestamps at the requested frequency, and a
``de_annualize`` helper for converting annualized yields into the
per-period rates used in Sharpe ratio computations.
"""

from __future__ import annotations

import datetime as _dt
import math
import os
from typing import cast

import pandas as pd

from ._periods import PERIODS_PER_YEAR
from .exceptions import EmptyDataError, ProviderUnavailableError

DateLike = str | _dt.date | pd.Timestamp


def _periods_per_year(frequency: str) -> int:
    try:
        return PERIODS_PER_YEAR[frequency]
    except KeyError as exc:
        raise ValueError(
            f"unknown frequency {frequency!r}; expected one of {sorted(PERIODS_PER_YEAR)}"
        ) from exc


def de_annualize(annual_pct: pd.Series, frequency: str) -> pd.Series:
    """Convert an annual rate (already in decimal, e.g. 0.05) to per-period.

    Uses geometric de-annualization::

        r_period = (1 + r_annual) ** (1 / periods_per_year) - 1

    so a round-trip via the matching annualizer is exact.
    """
    n = _periods_per_year(frequency)
    return (1.0 + annual_pct) ** (1.0 / n) - 1.0


def risk_free_rate(
    start: DateLike,
    end: DateLike,
    *,
    series_id: str = "DGS1MO",
    frequency: str = "daily",
    api_key: str | None = None,
) -> pd.Series:
    """Fetch a FRED risk-free series and return it at ``frequency``.

    The FRED yield series are quoted in annualized percent (e.g. ``5.32``
    means 5.32% per year). This function converts to decimal,
    forward-fills any missing days inside the window, and resamples /
    de-annualizes to the requested frequency.
    """
    try:
        from fredapi import Fred  # noqa: PLC0415
    except ImportError as exc:
        raise ProviderUnavailableError(
            "fredapi is not installed; install with `pip install fredapi`"
        ) from exc

    key = api_key or os.environ.get("FRED_API_KEY")
    try:
        client = Fred(api_key=key)
        raw = client.get_series(
            series_id,
            observation_start=pd.Timestamp(start).strftime("%Y-%m-%d"),
            observation_end=pd.Timestamp(end).strftime("%Y-%m-%d"),
        )
    except Exception as exc:
        raise ProviderUnavailableError(
            f"FRED fetch failed for series {series_id!r}: {exc}"
        ) from exc

    if raw is None or len(raw) == 0:
        raise EmptyDataError(f"FRED returned no data for series {series_id!r}")

    series = pd.Series(raw, dtype="float64").sort_index() / 100.0
    series.index = pd.DatetimeIndex(series.index).tz_localize(None).normalize()
    series = series[~series.index.duplicated(keep="last")]
    series.name = series_id

    # Quoted as annualized; convert to per-period.
    per_period = de_annualize(series, frequency)
    if frequency == "daily":
        return per_period
    rule = {"weekly": "W", "monthly": "ME", "quarterly": "QE", "annual": "YE"}[frequency]
    # last observation in each period; equivalent to "rate in effect at period end"
    resampled = per_period.resample(rule).last().dropna()
    if resampled.empty:
        raise EmptyDataError(
            f"FRED series {series_id!r} contained no observations after resampling"
        )
    return cast(pd.Series, resampled)


def annualize_rate(per_period: pd.Series, frequency: str) -> pd.Series:
    """Inverse of :func:`de_annualize`. Provided for round-trip testing."""
    n = _periods_per_year(frequency)
    return (1.0 + per_period) ** n - 1.0


def constant_rate(value: float, index: pd.DatetimeIndex) -> pd.Series:
    """Construct a constant-rate series - useful for tests and quick demos."""
    if not math.isfinite(value):
        raise ValueError(f"value must be finite, got {value!r}")
    return pd.Series(value, index=index, name="rf", dtype="float64")
