"""Correlation-based distance metrics for HRP tree clustering.

Stage 1 of HRP turns a correlation matrix into a distance matrix suitable for
agglomerative clustering. de Prado uses a *two-step* distance: first the
correlation distance :math:`d_{ij} = \\sqrt{0.5(1 - \\rho_{ij})}`, then a
second-order Euclidean distance over the columns of that distance matrix.

Importing this module has no side effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hrp._exceptions import ValidationError
from hrp._typing import MatrixLike
from hrp._validation import ensure_dataframe

# quantcore-candidate: new code (the genuinely novel HRP stage); parity oracle =
# PyPortfolioOpt / mlfinlab (dev-only).


def _ensure_square(matrix: MatrixLike, *, name: str) -> pd.DataFrame:
    """Coerce ``matrix`` to a square, finite DataFrame with aligned labels.

    Local guardrail shared by the two distance kernels. quantcore-candidate:
    mirrors the square-matrix coercion pattern used across the estimators.
    """
    frame = ensure_dataframe(matrix, name=name)
    n_rows, n_cols = frame.shape
    if n_rows != n_cols:
        raise ValidationError(f"{name} must be square, got shape {frame.shape}.")
    # Give an ndarray input symmetric integer labels so the result is labelled
    # consistently along both axes.
    if not frame.index.equals(frame.columns):
        frame.index = frame.columns
    return frame


def correl_dist(corr: MatrixLike) -> pd.DataFrame:
    r"""Correlation distance :math:`d_{ij} = \sqrt{0.5\,(1 - \rho_{ij})}`.

    HONESTY REQUIREMENT: the metric is :math:`\sqrt{0.5(1 - \rho)}`, NOT
    :math:`1 - \rho`. The square-root form is a proper metric on the unit sphere
    (it satisfies the triangle inequality), maps :math:`\rho = +1 \to d = 0`,
    :math:`\rho = 0 \to d = \tfrac{1}{\sqrt{2}}`, and :math:`\rho = -1 \to d = 1`.
    Using ``1 - rho`` is the classic HRP footgun and is explicitly rejected here.

    Parameters
    ----------
    corr:
        An ``N x N`` correlation matrix with unit diagonal and entries in
        ``[-1, 1]`` (DataFrame or ndarray).

    Returns
    -------
    pandas.DataFrame
        An ``N x N`` symmetric distance matrix with a zero diagonal, labelled by
        asset.

    Raises
    ------
    ValidationError
        If ``corr`` is not square or contains entries outside ``[-1, 1]``.
    """
    frame = _ensure_square(corr, name="corr")

    values = frame.to_numpy()
    # HONESTY REQUIREMENT: domain check - a correlation matrix must live in
    # [-1, 1]. A small tolerance absorbs floating-point drift from estimation.
    tol = 1e-8
    if float(np.nanmin(values)) < -1.0 - tol or float(np.nanmax(values)) > 1.0 + tol:
        raise ValidationError("corr contains entries outside [-1, 1].")

    # Clip to the valid domain so 1 - rho never goes slightly negative under the
    # sqrt due to rounding, then apply the proper metric d = sqrt(0.5 * (1 - rho)).
    # HONESTY REQUIREMENT: this is sqrt(0.5 * (1 - rho)), NOT 1 - rho.
    clipped = np.clip(values, -1.0, 1.0)
    dist = np.sqrt(0.5 * (1.0 - clipped))

    # Enforce an exactly zero diagonal and exact symmetry (rho symmetric =>
    # distance symmetric, but kill residual asymmetry from input noise).
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)

    return pd.DataFrame(dist, index=frame.index, columns=frame.columns)


def euclidean_codistance(dist: MatrixLike) -> pd.DataFrame:
    r"""Second-order Euclidean distance over the columns of a distance matrix.

    Given the correlation-distance matrix ``dist`` from :func:`correl_dist`, this
    computes the pairwise Euclidean distance between its *columns*:

    .. math::

        \tilde{d}_{ij} = \sqrt{\sum_{k} (d_{ki} - d_{kj})^2}.

    This is the second distance de Prado feeds into linkage: it measures how
    similarly two assets relate to the *entire* universe, not just to each
    other, which stabilizes the resulting dendrogram.

    Parameters
    ----------
    dist:
        An ``N x N`` correlation-distance matrix (output of :func:`correl_dist`).

    Returns
    -------
    pandas.DataFrame
        An ``N x N`` symmetric Euclidean co-distance matrix with a zero
        diagonal, labelled by asset.

    Raises
    ------
    ValidationError
        If ``dist`` is not square.
    """
    frame = _ensure_square(dist, name="dist")

    values = frame.to_numpy()
    # Euclidean distance between columns i and j over the rows (the full
    # universe): tilde_d_ij = sqrt(sum_k (d_ki - d_kj)^2). Vectorized via the
    # identity ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b applied column-wise.
    gram = values.T @ values
    sq_norms = np.diag(gram)
    sq_dist = sq_norms[:, None] + sq_norms[None, :] - 2.0 * gram
    # Numerical floor: tiny negative values from cancellation -> 0 before sqrt.
    sq_dist = np.maximum(sq_dist, 0.0)
    codist = np.sqrt(sq_dist)

    codist = 0.5 * (codist + codist.T)
    np.fill_diagonal(codist, 0.0)

    return pd.DataFrame(codist, index=frame.index, columns=frame.columns)
