"""Heteroskedasticity- and autocorrelation-consistent standard errors.

Implements the Newey-West (1987) long-run variance with Bartlett weights,
optionally with the Andrews (1991) data-dependent lag selector. The estimator
returns a standard error of the *sample mean* so callers can build t-statistics
for the system-vs-buy-hold per-bar net-return differential (the Diebold-Mariano
denominator).

Drift-elimination: the HAC kernels are now thin re-exports of the shared
``quantcore`` package (``quantcore.newey_west_se`` / ``quantcore.andrews_lag``),
whose Bartlett-weighted long-run variance is numerically identical to this repo's
original implementation (both seeded from the same pairs-trading reference, parity
to 1e-10). The local public names and the pandas-or-ndarray input contract are
kept so every call site is unchanged; the only adaptation is translating
``quantcore.ValidationError`` to this repo's
:class:`algosystem._exceptions.ValidationError` (with the IDENTICAL message
string). Importing this module has no side effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from quantcore import ValidationError as _QuantCoreValidationError
from quantcore import andrews_lag as _qc_andrews_lag
from quantcore import newey_west_se as _qc_newey_west_se

from algosystem._exceptions import ValidationError

__all__ = ["andrews_lag", "newey_west_se"]


def andrews_lag(t: int) -> int:
    """Return the Andrews (1991) automatic lag truncation.

    Uses the rule of thumb ``ceil(4 * (T/100)**(2/9))`` which is the plug-in
    formula favoured by Newey-West for general autocovariance structures.

    Parameters
    ----------
    t : int
        Sample size; must be at least one.

    Returns
    -------
    int
        Non-negative lag truncation; never less than zero.

    Raises
    ------
    ValidationError
        If ``t <= 0``.
    """
    try:
        return _qc_andrews_lag(t)
    except _QuantCoreValidationError as exc:
        raise ValidationError(str(exc)) from exc


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

    Raises
    ------
    ValidationError
        If there are fewer than two finite observations or ``lag < 0``.
    """
    # Coerce the pandas-or-ndarray input to a float64 ndarray up front (exactly what
    # the quantcore kernel does internally), so the pandas Series branch of the local
    # input contract satisfies quantcore's ndarray-or-sequence signature with no
    # change in numeric behaviour.
    arr: NDArray[np.float64] = np.asarray(returns, dtype=np.float64)
    try:
        return _qc_newey_west_se(arr, lag=lag)
    except _QuantCoreValidationError as exc:
        raise ValidationError(str(exc)) from exc
