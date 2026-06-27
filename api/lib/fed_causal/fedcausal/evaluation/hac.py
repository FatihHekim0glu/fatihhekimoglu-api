"""Heteroskedasticity- and autocorrelation-consistent standard errors.

Implements the Newey-West (1987) long-run variance with Bartlett
weights, optionally with the Andrews (1991) data-dependent lag selector.
The estimator returns a standard error of the *sample mean* so callers
can build t-statistics for cumulative abnormal returns or other averaged
metrics.

PARTIALLY MIGRATED TO ``quantcore``. The Bartlett-weighted long-run-variance
KERNEL now lives in the shared, torch-free ``quantcore`` package
(:func:`quantcore.hac.newey_west_lrv`), the single source of truth for the
portfolio's honest-statistics primitives. :func:`newey_west_se` here delegates
its numeric core to that kernel (parity to 1e-10) while KEEPING ``fedcausal``'s
own input validation, because the contract differs from quantcore's in two
behaviour-load-bearing ways:

- ``_coerce_array`` REJECTS a non-1-D input with ``ValidationError`` (quantcore's
  coercion silently flattens it instead), and
- the validation messages (``"returns must be one-dimensional"``, ``"need at
  least two finite observations"``, ``"lag must be non-negative; ..."``, ``"t must
  be positive; ..."``) are ``fedcausal``'s own and are preserved verbatim.

Only the numeric kernel — never the validation surface — is shared, so the public
behaviour (including the exception type and every message) is unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from quantcore.hac import newey_west_lrv as _qc_newey_west_lrv

from fedcausal._exceptions import ValidationError

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
        raise ValidationError(f"t must be positive; got {t}")
    return int(np.ceil(4.0 * (t / 100.0) ** (2.0 / 9.0)))


def _coerce_array(returns: pd.Series | NDArray[np.float64]) -> NDArray[np.float64]:
    if isinstance(returns, pd.Series):
        arr = returns.to_numpy(dtype=float, copy=False)
    else:
        arr = np.asarray(returns, dtype=float)
    if arr.ndim != 1:
        raise ValidationError("returns must be one-dimensional")
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        raise ValidationError("need at least two finite observations")
    # Boolean-masked indexing widens to ``Any`` under the numpy stubs; the dtype
    # is fixed (float64) by construction above, so narrow it back explicitly.
    return np.asarray(arr, dtype=np.float64)


def newey_west_se(
    returns: pd.Series | NDArray[np.float64],
    *,
    lag: int | None = None,
) -> float:
    """Newey-West HAC standard error of the sample mean.

    Parameters
    ----------
    returns : pandas.Series or numpy.ndarray
        Realised observations (e.g. per-event CARs or a daily series). Non-finite
        values are dropped.
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
        raise ValidationError(f"lag must be non-negative; got {lag}")
    # Bartlett-weighted long-run variance kernel (shared with quantcore; the
    # coerced array is already 1-D + finite so quantcore's idempotent coercion is
    # a no-op and the result is byte-identical to the former local loop).
    omega = _qc_newey_west_lrv(arr, lag=lag)
    return float(np.sqrt(omega / t))
