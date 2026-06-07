"""Memmel (2003) closed-form test of Sharpe-ratio equality.

The test addresses the standard Jobson-Korkie (1981) statistic for the
difference of two Sharpe ratios estimated on correlated return streams,
correcting the algebra error that Memmel identified. The null is
``H_0: SR_a = SR_b``.

Reference: Memmel, C. (2003), "Performance Hypothesis Testing with the
Sharpe Ratio", *Finance Letters*, 1, 21-23.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from pairs._exceptions import InputError

from .results import MemmelResult

__all__ = ["memmel_test"]


def _coerce(returns: pd.Series | NDArray[np.float64], name: str) -> NDArray[np.float64]:
    if isinstance(returns, pd.Series):
        arr = returns.to_numpy(dtype=float, copy=False)
    else:
        arr = np.asarray(returns, dtype=float)
    if arr.ndim != 1:
        raise InputError(f"{name} must be one-dimensional")
    return arr


def memmel_test(
    returns_a: pd.Series | NDArray[np.float64],
    returns_b: pd.Series | NDArray[np.float64],
) -> MemmelResult:
    """Test ``SR_a == SR_b`` on correlated return streams.

    Parameters
    ----------
    returns_a, returns_b : pandas.Series or numpy.ndarray
        Aligned return series of the same length.

    Returns
    -------
    MemmelResult
        Both Sharpe ratios, the z statistic, the two-sided p-value, the
        correlation and the effective sample size.
    """
    a = _coerce(returns_a, "returns_a")
    b = _coerce(returns_b, "returns_b")
    if a.size != b.size:
        raise InputError("returns_a and returns_b must have the same length")
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    t = a.size
    if t < 4:
        raise InputError("need at least four paired observations")
    mu_a = float(a.mean())
    mu_b = float(b.mean())
    sigma_a = float(a.std(ddof=1))
    sigma_b = float(b.std(ddof=1))
    if sigma_a <= 0.0 or sigma_b <= 0.0:
        raise InputError("standard deviations must be positive")
    sr_a = mu_a / sigma_a
    sr_b = mu_b / sigma_b
    cov = float(np.cov(a, b, ddof=1)[0, 1])
    rho = cov / (sigma_a * sigma_b)
    rho = float(np.clip(rho, -1.0, 1.0))
    theta = (1.0 / t) * (
        2.0 * (1.0 - rho) + 0.5 * (sr_a * sr_a + sr_b * sr_b - 2.0 * sr_a * sr_b * rho * rho)
    )
    if theta <= 0.0:
        raise InputError("Memmel variance term is non-positive")
    z = (sr_a - sr_b) / np.sqrt(theta)
    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
    p_value = float(np.clip(p_value, 0.0, 1.0))
    return MemmelResult(
        sr_a=float(sr_a),
        sr_b=float(sr_b),
        z_stat=float(z),
        p_value=p_value,
        n_obs=int(t),
        correlation=float(rho),
    )
