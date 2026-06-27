"""Heteroskedasticity- and autocorrelation-consistent standard errors.

Implements the Newey-West (1987) long-run variance with Bartlett
weights, optionally with the Andrews (1991) data-dependent lag selector.
The estimator returns a standard error of the *sample mean* so callers
can build t-statistics for Sharpe ratios or other averaged metrics.

MIGRATED TO QUANTCORE: the numeric Newey-West kernel and the Andrews lag rule
are now sourced from ``quantcore`` (github.com/FatihHekim0glu/quantcore @v0.1.0),
whose ``newey_west_se`` / ``andrews_lag`` are byte-identical to the kernel that
lived here (parity confirmed to the last ULP). This module keeps its OWN input
validation so the public contract is unchanged: the local ``ValidationError``
type (no shared ancestry with ``quantcore.ValidationError``), the
``pd.Series | NDArray`` annotation, and the EXACT error messages callers and
tests rely on are all preserved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from quantcore import andrews_lag as _qc_andrews_lag
from quantcore import newey_west_se as _qc_newey_west_se

from wsb_sentiment._exceptions import ValidationError as InputError

__all__ = ["andrews_lag", "newey_west_se"]


def andrews_lag(t: int) -> int:
    """Return the Andrews (1991) automatic lag truncation.

    Uses the rule of thumb ``ceil(4 * (T/100)**(2/9))`` which is the
    plug-in formula favoured by Newey-West for general autocovariance
    structures.

    MIGRATED: the local positivity check (with this package's message) is kept,
    then the byte-identical :func:`quantcore.andrews_lag` rule is applied.

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
    return _qc_andrews_lag(t)


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
    return np.asarray(arr, dtype=np.float64)


def newey_west_se(
    returns: pd.Series | NDArray[np.float64],
    *,
    lag: int | None = None,
) -> float:
    """Newey-West HAC standard error of the sample mean.

    MIGRATED: the local coercion/validation (this package's ``ValidationError``
    and messages) is kept, then the byte-identical numeric kernel is delegated to
    :func:`quantcore.newey_west_se`. With ``lag=None`` the Andrews rule is applied
    locally (so the local positivity message is preserved) and the resulting
    truncation is passed through explicitly, keeping the result identical.

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
    return _qc_newey_west_se(arr, lag=lag)
