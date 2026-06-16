"""Marchenko-Pastur random-matrix-theory covariance denoising.

Eigenvalues of an empirical correlation matrix that fall below the
Marchenko-Pastur upper edge are statistically indistinguishable from noise; this
module clips (flattens) them while preserving the trace, leaving the
signal-carrying eigenvalues intact. Exposed as an optional ``rmt_denoise`` step
that can sit between covariance estimation and clustering.

Importing this module has no side effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockclusters._exceptions import ValidationError
from stockclusters._typing import MatrixLike

# quantcore-candidate: mirrors eigen-portfolios:src/eigenportfolios/rmt.py


def marchenko_pastur_clip(
    cov: MatrixLike,
    *,
    n_obs: int,
    n_assets: int | None = None,
) -> pd.DataFrame:
    r"""Denoise a covariance matrix by clipping sub-edge eigenvalues.

    The procedure converts ``cov`` to a correlation matrix, eigendecomposes it,
    and identifies the Marchenko-Pastur upper edge
    :math:`\lambda_+ = \sigma^2 (1 + \sqrt{q})^2` with aspect ratio
    :math:`q = N / T` (NOTE the convention: ``q = n_assets / n_obs``). All
    eigenvalues at or below :math:`\lambda_+` are replaced by their common
    average (so the trace, and hence total variance, is preserved); the rest are
    kept. The cleaned correlation matrix is then rescaled back to a covariance
    matrix using the original asset standard deviations.

    NO-LOOKAHEAD REQUIREMENT: the cutoff :math:`\lambda_+` depends only on the
    in-sample ``(n_obs, n_assets)`` and the in-sample eigenvalues; it must be
    future-perturbation-invariant (asserted in the property suite).

    Parameters
    ----------
    cov:
        An ``N x N`` covariance matrix (DataFrame or ndarray).
    n_obs:
        The number of in-sample observations ``T`` used to estimate ``cov``.
    n_assets:
        The number of assets ``N``. If ``None``, inferred from ``cov``'s shape.

    Returns
    -------
    pandas.DataFrame
        The denoised ``N x N`` covariance matrix, symmetric and labelled by
        asset.

    Raises
    ------
    ValidationError
        If ``cov`` is not square or ``n_obs`` is non-positive.
    """
    if isinstance(cov, pd.DataFrame):
        labels = cov.index
        cov_arr = cov.to_numpy(dtype=np.float64)
    else:
        cov_arr = np.asarray(cov, dtype=np.float64)
        labels = pd.RangeIndex(cov_arr.shape[0]) if cov_arr.ndim == 2 else pd.RangeIndex(0)

    if cov_arr.ndim != 2 or cov_arr.shape[0] != cov_arr.shape[1]:
        raise ValidationError(f"cov must be a square 2-D matrix, got shape {cov_arr.shape}.")
    if int(n_obs) <= 0:
        raise ValidationError(f"n_obs must be positive, got {n_obs}.")

    n = cov_arr.shape[0]
    if n_assets is None:
        n_assets = n

    # Decompose covariance into standard deviations and a correlation matrix.
    std = np.sqrt(np.diag(cov_arr))
    # Guard against zero/negative diagonal variances when forming the correlation.
    safe_std = np.where(std > 0.0, std, 1.0)
    inv_std = 1.0 / safe_std
    corr = cov_arr * np.outer(inv_std, inv_std)
    # Symmetrize before eigendecomposition (eigh assumes a symmetric input).
    corr = 0.5 * (corr + corr.T)

    eigvals, eigvecs = np.linalg.eigh(corr)

    # Marchenko-Pastur upper edge. For a correlation matrix the average eigenvalue
    # (population variance per series) is sigma^2 = 1. Convention: q = N / T.
    q = float(n_assets) / float(n_obs)
    lambda_plus = (1.0 + np.sqrt(q)) ** 2

    # Clip: every eigenvalue at or below the noise edge is replaced by their common
    # average so the trace (total variance) is preserved; signal eigenvalues are kept.
    noise_mask = eigvals <= lambda_plus
    n_noise = int(noise_mask.sum())
    cleaned = eigvals.copy()
    if n_noise > 0:
        cleaned[noise_mask] = float(eigvals[noise_mask].sum()) / n_noise

    # Reconstruct the denoised correlation matrix and re-impose a unit diagonal
    # (rescaling each row/column so the diagonal is exactly 1, as a correlation).
    corr_clean = (eigvecs * cleaned) @ eigvecs.T
    corr_clean = 0.5 * (corr_clean + corr_clean.T)
    diag = np.sqrt(np.diag(corr_clean))
    safe_diag = np.where(diag > 0.0, diag, 1.0)
    norm = 1.0 / safe_diag
    corr_clean = corr_clean * np.outer(norm, norm)
    np.fill_diagonal(corr_clean, 1.0)

    # Rescale the cleaned correlation back to a covariance using the original stds.
    cov_clean = corr_clean * np.outer(std, std)
    cov_clean = 0.5 * (cov_clean + cov_clean.T)

    return pd.DataFrame(cov_clean, index=labels, columns=labels)
