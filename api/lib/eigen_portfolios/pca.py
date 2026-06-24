"""PCA eigendecomposition of a daily-returns panel.

We work on the correlation matrix (Pearson, equivalently the covariance of
z-scored returns) so that eigenvalues are dimensionless and sum to ``N``. That
matches the canonical form used by Plerou et al. (1999) and Bouchaud (2001) for
random-matrix-theory comparisons.

Pipeline
--------
1. Drop columns whose return series has zero variance (e.g. a stopped ticker).
2. Z-score each surviving column (column-wise demean, divide by std).
3. Form the ``N × N`` correlation matrix ``C = (Z^T Z) / T``.
4. Symmetric eigendecomposition with ``np.linalg.eigh`` - sorted descending.
5. Factor returns: ``F = Z @ V`` where ``V`` is the orthonormal eigenvector
   matrix. ``F[:, k]`` is the time series of the ``k``-th principal portfolio.

The function is numpy-pure and deterministic - no I/O, no randomness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Minimum std below which a column is considered constant and dropped. Picked
# to be small enough that legitimate low-vol names survive while pre-IPO /
# post-delist zero-rows get filtered.
_MIN_STD = 1e-12


@dataclass(frozen=True)
class EigenResult:
    """Bundled output of :func:`eigen_decompose`.

    Attributes
    ----------
    eigvals
        1-D array of length ``N`` sorted descending. Sum equals ``N`` for
        correlation-matrix PCA (numerical tolerance).
    eigvecs
        ``N × N`` DataFrame; column ``k`` is the ``k``-th eigenvector (loadings
        across tickers). Rows are indexed by ticker; columns are integer
        component ranks ``0..N-1``.
    factor_returns
        ``T × N`` DataFrame; column ``k`` is ``Z @ V[:, k]``, i.e. the time
        series of the ``k``-th eigen-portfolio's z-score returns.
    explained_var
        ``eigvals / eigvals.sum()`` - share of total variance per component.
    tickers
        The surviving ticker order after the zero-variance filter.
    """

    eigvals: np.ndarray
    eigvecs: pd.DataFrame
    factor_returns: pd.DataFrame
    explained_var: np.ndarray
    tickers: list[str]

    @property
    def n(self) -> int:
        return len(self.tickers)


def eigen_decompose(returns: pd.DataFrame) -> EigenResult:
    """Run PCA on the correlation matrix of a ``T × N`` returns panel.

    Parameters
    ----------
    returns
        DataFrame indexed by time with one column per ticker. NaN rows should
        be dropped beforehand - we do not impute.

    Returns
    -------
    EigenResult
        See class docs. ``eigvals`` is sorted descending and eigenvectors are
        orthonormal (verified by the test suite).

    Raises
    ------
    ValueError
        If the panel has fewer than 2 columns after dropping zero-variance
        ones, or zero rows.
    """
    if returns.empty:
        raise ValueError("returns panel is empty")

    # 1. Drop zero-variance columns (constant or all-NaN-then-filled rows).
    stds = returns.std(axis=0, ddof=1)
    keep_mask = stds > _MIN_STD
    if int(keep_mask.sum()) < 2:
        raise ValueError(
            "at least 2 tickers with non-zero variance are required for PCA"
        )
    panel = returns.loc[:, keep_mask].copy()
    tickers = list(panel.columns)

    # 2. Z-score column-wise (demean + scale by sample std).
    means = panel.mean(axis=0)
    stds = panel.std(axis=0, ddof=1)
    z = (panel - means) / stds  # T × N

    z_mat = z.to_numpy(dtype=float)
    t_obs, n_assets = z_mat.shape
    if t_obs < 2:
        raise ValueError("at least 2 observations are required for PCA")

    # 3. Correlation matrix. Using Z^T Z / (T-1) matches np.corrcoef when the
    # input is sample-z-scored (the diagonal will be ~1 by construction).
    corr = (z_mat.T @ z_mat) / (t_obs - 1)
    # Symmetrise to suppress floating-point asymmetry feeding eigh.
    corr = (corr + corr.T) / 2.0

    # 4. Symmetric eigendecomposition. np.linalg.eigh returns ascending; flip.
    eigvals_asc, eigvecs_asc = np.linalg.eigh(corr)
    order = np.argsort(eigvals_asc)[::-1]
    eigvals = np.ascontiguousarray(eigvals_asc[order])
    eigvecs_mat = np.ascontiguousarray(eigvecs_asc[:, order])

    # 5. Factor returns: T × N matrix of z-score-projected portfolio returns.
    factor_mat = z_mat @ eigvecs_mat

    component_labels = [f"PC{k + 1}" for k in range(n_assets)]
    eigvecs = pd.DataFrame(eigvecs_mat, index=tickers, columns=component_labels)
    factor_returns = pd.DataFrame(
        factor_mat, index=panel.index, columns=component_labels
    )

    total = float(eigvals.sum())
    if total <= 0:
        explained_var = np.zeros_like(eigvals)
    else:
        explained_var = eigvals / total

    return EigenResult(
        eigvals=eigvals,
        eigvecs=eigvecs,
        factor_returns=factor_returns,
        explained_var=explained_var,
        tickers=tickers,
    )
