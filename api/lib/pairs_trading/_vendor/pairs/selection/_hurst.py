"""Rescaled-range (R/S) estimator of the Hurst exponent.

The Hurst exponent ``H`` characterises long-range dependence of a stationary
series:

* ``H < 0.5`` indicates mean-reversion (anti-persistence);
* ``H == 0.5`` is consistent with a random walk / iid white noise;
* ``H > 0.5`` indicates persistence (trend-following).

This implementation uses the classical rescaled-range method on log-spaced
window sizes and returns the slope of ``log(R/S)`` versus ``log(n)`` fit by
ordinary least squares. The estimator is biased for short samples (see
Weron 2002); callers should treat values within roughly ``+/-0.05`` of the
random-walk benchmark as inconclusive.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pairs._exceptions import InputError, InsufficientDataError

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray

__all__ = ["hurst_exponent"]

_MIN_SAMPLES: int = 100


def _to_array(series: pd.Series | NDArray[np.float64] | list[float]) -> NDArray[np.float64]:
    """Coerce input to a clean 1-D ``float64`` numpy array.

    Drops non-finite values (NaN, +/-inf) because the rescaled-range formula
    requires finite cumulative deviations.
    """
    arr = np.asarray(series, dtype=np.float64).reshape(-1)
    finite = np.isfinite(arr)
    return arr[finite]


def _rs_for_window(values: NDArray[np.float64], window: int) -> float:
    """Compute the average rescaled range across non-overlapping windows.

    For each chunk of ``window`` consecutive observations the function:

    1. Subtracts the chunk mean to obtain centred deviations.
    2. Forms the cumulative deviation series ``Z``.
    3. Computes the range ``R = max(Z) - min(Z)`` and the standard deviation
       ``S = std(values, ddof=1)``.
    4. Aggregates ``R / S`` across chunks via the arithmetic mean.

    Chunks with ``S == 0`` (constant slice) contribute nothing -- they are
    discarded before averaging.
    """
    n = values.size
    n_chunks = n // window
    rs_values: list[float] = []
    for i in range(n_chunks):
        chunk = values[i * window : (i + 1) * window]
        mean = float(chunk.mean())
        centred = chunk - mean
        cumulative = np.cumsum(centred)
        rng = float(cumulative.max() - cumulative.min())
        std = float(chunk.std(ddof=1))
        if std == 0.0 or not np.isfinite(std):
            continue
        rs_values.append(rng / std)
    if not rs_values:
        return float("nan")
    return float(np.mean(rs_values))


def hurst_exponent(
    series: pd.Series | NDArray[np.float64] | list[float],
    *,
    min_lag: int = 10,
    max_lag: int | None = None,
    is_increments: bool = True,
) -> float:
    """Estimate the Hurst exponent of ``series`` via rescaled-range analysis.

    Parameters
    ----------
    series
        One-dimensional sequence of observations. Non-finite values are
        silently dropped before estimation.
    min_lag
        Smallest window size considered. Must satisfy ``min_lag >= 2``.
    max_lag
        Largest window size considered. Defaults to ``len(series) // 2`` so
        each window provides at least two non-overlapping chunks.
    is_increments
        Whether ``series`` is already an innovation/increment process such as
        returns or spread innovations. When ``False`` the function takes first
        differences via :func:`numpy.diff` before computing R/S, which is the
        correct treatment for level series (prices, random-walk paths) whose
        rescaled range grows like ``n`` rather than ``n ** H``. Defaults to
        ``True`` because most callers pass stationary spreads or returns.

    Returns
    -------
    float
        Estimated Hurst exponent. Returns ``float("nan")`` if every window
        produced a degenerate (zero-variance) sample.

    Raises
    ------
    InsufficientDataError
        If fewer than ``100`` finite observations remain.
    InputError
        If ``min_lag`` is ``< 2`` or ``max_lag <= min_lag`` after defaulting.
    """
    if min_lag < 2:
        msg = f"min_lag must be >= 2; got {min_lag}"
        raise InputError(msg)
    values = _to_array(series)
    if not is_increments:
        values = np.diff(values)
    if values.size < _MIN_SAMPLES:
        msg = f"need at least {_MIN_SAMPLES} finite observations; got {values.size}"
        raise InsufficientDataError(msg)
    upper = max_lag if max_lag is not None else max(min_lag + 1, values.size // 2)
    if upper <= min_lag:
        msg = f"max_lag ({upper}) must exceed min_lag ({min_lag})"
        raise InputError(msg)

    n_points = max(4, int(np.log2(max(upper / min_lag, 2.0))) + 1)
    lags = np.unique(np.round(np.geomspace(min_lag, upper, num=n_points)).astype(int))
    lags = lags[lags >= min_lag]
    if lags.size < 2:
        msg = "need at least two distinct lag sizes; widen [min_lag, max_lag]"
        raise InputError(msg)

    log_lags: list[float] = []
    log_rs: list[float] = []
    for lag in lags:
        rs = _rs_for_window(values, int(lag))
        if not np.isfinite(rs) or rs <= 0.0:
            continue
        log_lags.append(float(np.log(lag)))
        log_rs.append(float(np.log(rs)))

    if len(log_lags) < 2:
        return float("nan")
    slope, _ = np.polyfit(log_lags, log_rs, 1)
    return float(slope)
