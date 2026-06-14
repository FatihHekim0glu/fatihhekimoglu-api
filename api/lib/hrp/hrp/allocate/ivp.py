"""Inverse-variance portfolio (IVP) allocation.

The IVP is the diagonal-only special case of minimum-variance: it ignores all
off-diagonal covariances and weights each asset inversely to its own variance. In
the HRP horse race it is the natural "no clustering, no off-diagonal information"
baseline that isolates what the tree structure adds.

Importing this module has no side effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hrp._exceptions import ValidationError
from hrp._typing import MatrixLike

# quantcore-candidate: mirrors markowitz-optimizer:src/markowitz/allocate/ivp.py


def ivp_weights(cov: MatrixLike) -> pd.Series:
    r"""Inverse-variance portfolio weights.

    Uses only the diagonal of ``cov``:
    :math:`w_i = \dfrac{1 / \sigma_i^2}{\sum_j 1 / \sigma_j^2}`, where
    :math:`\sigma_i^2 = \Sigma_{ii}`. The result lies on the simplex (sums to
    one, all non-negative) and requires no matrix inversion, so it is robust to
    off-diagonal singularity.

    Parameters
    ----------
    cov:
        An ``N x N`` covariance matrix (DataFrame or ndarray). Only the diagonal
        is used.

    Returns
    -------
    pandas.Series
        Inverse-variance weights labelled by asset.

    Raises
    ------
    ValidationError
        If ``cov`` is not square or has a non-positive diagonal entry.
    """
    frame = cov if isinstance(cov, pd.DataFrame) else pd.DataFrame(cov)
    n_rows, n_cols = frame.shape
    if n_rows != n_cols:
        raise ValidationError(
            f"ivp_weights: cov must be square, got shape {(n_rows, n_cols)}."
        )

    variances = np.asarray(np.diag(frame.to_numpy(dtype="float64")), dtype="float64")
    if not np.all(np.isfinite(variances)):
        raise ValidationError("ivp_weights: cov diagonal contains non-finite entries.")
    if np.any(variances <= 0.0):
        raise ValidationError(
            "ivp_weights: cov has a non-positive diagonal (variance) entry."
        )

    inv_var = 1.0 / variances
    weights = inv_var / inv_var.sum()
    return pd.Series(weights, index=frame.columns, dtype="float64")
