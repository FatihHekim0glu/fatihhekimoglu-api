"""Heteroskedasticity- and autocorrelation-consistent (HAC) long-run variance.

The Newey-West (1987) Bartlett-weighted long-run variance, with the Andrews
(1991) data-dependent lag selector. Two return conventions are exposed
EXPLICITLY because the repos disagreed on which one they meant (see DRIFT_REPORT
§4):

- :func:`newey_west_lrv` returns the long-run variance ``omega_hat`` itself
  (the volforecast convention);
- :func:`newey_west_se` returns the standard error of the **sample mean**,
  ``sqrt(omega_hat / T)`` (the fed-causal / algo-system convention, "parity to
  the reference holds to 1e-10"), which is the Diebold-Mariano denominator.

At lag 0 the estimator reduces to the iid variance ``gamma_0`` (the
``omega_hat`` with no autocovariance terms), asserted in the property suite.
Importing this module has no side effects.
"""

from __future__ import annotations

import numpy as np

from quantcore._exceptions import ValidationError
from quantcore._typing import FloatArray, Series1D

__all__ = ["andrews_lag", "newey_west_lrv", "newey_west_se"]


def andrews_lag(t: int) -> int:
    """Andrews (1991) automatic Bartlett lag truncation ``ceil(4*(T/100)**(2/9))``.

    The plug-in formula favoured by Newey-West for general autocovariance
    structures.

    Parameters
    ----------
    t:
        Sample size (must be positive).

    Returns
    -------
    int
        The non-negative lag truncation (never less than zero).

    Raises
    ------
    ValidationError
        If ``t <= 0``.
    """
    if t <= 0:
        raise ValidationError(f"andrews_lag: t must be positive, got {t}.")
    return int(np.ceil(4.0 * (t / 100.0) ** (2.0 / 9.0)))


def _coerce_finite(x: Series1D) -> FloatArray:
    """Coerce ``x`` to a 1-D float64 array, dropping non-finite entries."""
    arr = np.asarray(x, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        raise ValidationError("HAC estimator needs at least two finite observations.")
    return np.asarray(arr, dtype=np.float64)


def newey_west_lrv(x: Series1D, *, lag: int | None = None) -> float:
    r"""Newey-West (1987) Bartlett long-run variance ``omega_hat`` of a 1-D series.

    .. math::

        \widehat{\omega} = \gamma_0 + 2\sum_{h=1}^{L}
                            \left(1 - \tfrac{h}{L+1}\right)\gamma_h,

    with :math:`\gamma_h = \tfrac{1}{T}\sum_t (x_t - \bar{x})(x_{t-h} - \bar{x})`
    and the Andrews automatic lag ``L = ceil(4 (T/100)^{2/9})`` when ``lag`` is
    ``None``. At ``lag=0`` this reduces to :math:`\gamma_0`, the iid variance
    (with the ``1/T`` divisor). The result is floored at ``0``.

    Parameters
    ----------
    x:
        A 1-D series (e.g. a Diebold-Mariano loss/performance differential).
        Non-finite entries are dropped.
    lag:
        Bartlett truncation lag; ``None`` selects the Andrews rule.

    Returns
    -------
    float
        The (non-negative) long-run variance estimate ``omega_hat``.

    Raises
    ------
    ValidationError
        If ``x`` has fewer than two finite observations or ``lag < 0``.
    """
    arr = _coerce_finite(x)
    t = arr.size
    truncation = andrews_lag(t) if lag is None else int(lag)
    if truncation < 0:
        raise ValidationError(f"newey_west_lrv: lag must be non-negative, got {truncation}.")
    centred = arr - arr.mean()
    gamma0 = float(np.dot(centred, centred) / t)
    omega = gamma0
    max_lag = min(truncation, t - 1)
    for h in range(1, max_lag + 1):
        weight = 1.0 - h / (truncation + 1.0)
        gamma_h = float(np.dot(centred[h:], centred[:-h]) / t)
        omega += 2.0 * weight * gamma_h
    return max(omega, 0.0)


def newey_west_se(x: Series1D, *, lag: int | None = None) -> float:
    """Newey-West HAC standard error of the sample mean: ``sqrt(omega_hat / T)``.

    The Diebold-Mariano denominator. Equals :func:`newey_west_lrv` divided by the
    sample size and square-rooted. With ``lag=0`` and no autocorrelation this is
    the usual ``sqrt(var / T)`` standard error of the mean.

    Parameters
    ----------
    x:
        A 1-D series (e.g. a DM differential). Non-finite entries are dropped.
    lag:
        Bartlett truncation lag; ``None`` selects the Andrews rule.

    Returns
    -------
    float
        The HAC standard error of the sample mean.

    Raises
    ------
    ValidationError
        If ``x`` has fewer than two finite observations or ``lag < 0``.
    """
    arr = _coerce_finite(x)
    omega = newey_west_lrv(arr, lag=lag)
    return float(np.sqrt(omega / arr.size))
