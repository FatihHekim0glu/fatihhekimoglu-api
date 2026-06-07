"""Estimators of the effective number of independent trials.

When a screen examines ``N`` strategies whose returns are correlated, the
effective sample size for multiplicity adjustments is smaller than ``N``.
Two cheap closed-form estimates are supported:

* **PCA / participation ratio.** Eigendecompose the correlation matrix and
  return ``(sum lambda_i)^2 / sum(lambda_i^2)``. Equals ``N`` for perfectly
  independent strategies and collapses to ``1`` when every strategy is a
  scalar multiple of one factor.
* **Average correlation.** Use the closed-form ``N / (1 + (N - 1) * rho_bar)``
  derived from a one-factor model where every off-diagonal correlation
  equals ``rho_bar``.

Both estimates are clipped to ``[1, N]`` so that downstream code can use the
result as an integer ceiling for Bonferroni-style adjustments without
worrying about boundary effects.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from pairs._exceptions import InputError, InsufficientDataError

__all__ = ["estimate_effective_n"]


def _correlation_matrix(returns: pd.DataFrame) -> np.ndarray:
    """Return the Pearson correlation matrix as a clean ``float64`` array.

    Constant columns produce ``NaN`` rows in ``DataFrame.corr``; we replace
    those with the identity to keep the eigendecomposition well-defined and
    to reflect that a constant strategy contributes a single independent
    "direction" (itself).
    """
    # to_numpy() can return a read-only view under pandas 3.0 copy-on-write;
    # copy=True guarantees a writable array for the NaN-fixup below.
    corr = returns.corr().to_numpy(dtype=float, copy=True)
    n = corr.shape[0]
    bad = ~np.isfinite(corr)
    if bad.any():
        corr[bad] = 0.0
        diag_idx = np.arange(n)
        corr[diag_idx, diag_idx] = 1.0
    return corr


def estimate_effective_n(
    trial_returns: pd.DataFrame,
    *,
    method: Literal["pca", "avg_corr"] = "pca",
) -> float:
    """Estimate the effective number of independent trials.

    Parameters
    ----------
    trial_returns
        DataFrame of returns whose columns index distinct trials/strategies
        and whose rows index time. Must contain at least two rows so the
        correlation matrix is defined.
    method
        ``"pca"`` for the participation ratio derived from correlation
        eigenvalues; ``"avg_corr"`` for the one-factor approximation
        ``N / (1 + (N - 1) * rho_bar)``.

    Returns
    -------
    float
        Effective number of trials, always in ``[1, N]`` where ``N`` is the
        number of columns. A single-column input always returns ``1.0``.

    Raises
    ------
    InsufficientDataError
        If the input has fewer than two rows.
    InputError
        If the method is unknown or the input is not a DataFrame.
    """
    if not isinstance(trial_returns, pd.DataFrame):
        msg = "trial_returns must be a pandas DataFrame"
        raise InputError(msg)
    n_cols = trial_returns.shape[1]
    if n_cols == 0:
        msg = "trial_returns has zero columns"
        raise InputError(msg)
    if n_cols == 1:
        return 1.0
    if trial_returns.shape[0] < 2:
        msg = "need at least 2 rows to estimate effective N"
        raise InsufficientDataError(msg)

    if method == "pca":
        corr = _correlation_matrix(trial_returns)
        eigvals = np.linalg.eigvalsh(corr)
        eigvals = np.clip(eigvals, 0.0, None)
        denom = float(np.sum(eigvals**2))
        if denom <= 0.0:
            return 1.0
        n_eff = (float(np.sum(eigvals)) ** 2) / denom
    elif method == "avg_corr":
        corr = _correlation_matrix(trial_returns)
        n = corr.shape[0]
        off_diag = corr[~np.eye(n, dtype=bool)]
        rho_bar = float(np.mean(off_diag)) if off_diag.size else 0.0
        rho_bar = max(rho_bar, -1.0 / (n - 1) + 1e-12)
        n_eff = n / (1.0 + (n - 1) * rho_bar)
    else:
        msg = f"unknown method {method!r}; expected 'pca' or 'avg_corr'"
        raise InputError(msg)

    return float(np.clip(n_eff, 1.0, float(n_cols)))
