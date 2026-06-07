"""Heteroskedasticity- and autocorrelation-consistent standard errors.

Implements the Newey-West (1987) long-run variance with Bartlett
weights, optionally with the Andrews (1991) data-dependent lag selector.
The estimator returns a standard error of the *sample mean* so callers
can build t-statistics for Sharpe ratios or other averaged metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pairs._exceptions import InputError

__all__ = ["andrews_lag", "newey_west_se"]


def andrews_lag(t: int) -> int:
    """Return the Andrews (1991) automatic lag truncation.

    Uses the rule of thumb ``ceil(4 * (T/100)**(2/9))`` which is the
    plug-in formula favoured by Newey-West for general autocovariance
    structures.

    Parameters
    ----------
    t : int
        Sample size; must be at least one.

    Returns
    -------
    int
        Non-negative lag truncation; never less than zero.
    """
    if t <= 0:
        raise InputError(f"t must be positive; got {t}")
    return int(np.ceil(4.0 * (t / 100.0) ** (2.0 / 9.0)))


def _coerce_array(returns: pd.Series | NDArray[np.float64]) -> NDArray[np.float64]:
    if isinstance(returns, pd.Series):
        arr = returns.to_numpy(dtype=float, copy=False)
    else:
        arr = np.asarray(returns, dtype=float)
    if arr.ndim != 1:
        raise InputError("returns must be one-dimensional")
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        raise InputError("need at least two finite observations")
    return arr


def newey_west_se(
    returns: pd.Series | NDArray[np.float64],
    *,
    lag: int | None = None,
) -> float:
    """Newey-West HAC standard error of the sample mean.

    Parameters
    ----------
    returns : pandas.Series or numpy.ndarray
        Realised returns. Non-finite values are dropped.
    lag : int, optional
        Bartlett lag truncation. ``None`` selects the Andrews rule via
        :func:`andrews_lag`.

    Returns
    -------
    float
        Standard error of the sample mean, ``sqrt(omega_hat / T)`` where
        ``omega_hat`` is the Bartlett-weighted long-run variance.
    """
    arr = _coerce_array(returns)
    t = arr.size
    if lag is None:
        lag = andrews_lag(t)
    if lag < 0:
        raise InputError(f"lag must be non-negative; got {lag}")
    centred = arr - arr.mean()
    gamma0 = float(np.dot(centred, centred) / t)
    omega = gamma0
    max_lag = min(lag, t - 1)
    for h in range(1, max_lag + 1):
        weight = 1.0 - h / (lag + 1.0)
        gamma_h = float(np.dot(centred[h:], centred[:-h]) / t)
        omega += 2.0 * weight * gamma_h
    omega = max(omega, 0.0)
    return float(np.sqrt(omega / t))
