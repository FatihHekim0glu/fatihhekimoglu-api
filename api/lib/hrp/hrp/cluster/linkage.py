"""Agglomerative linkage for HRP tree clustering.

Stage 1 of HRP finishes by running SciPy hierarchical/agglomerative clustering
on the (second-order) distance matrix. The paper default is **single** linkage;
``ward``/``complete``/``average`` are exposed only as configurable ablations.

Importing this module has no side effects.
"""

from __future__ import annotations

import numpy as np

from hrp._exceptions import ValidationError
from hrp._typing import MatrixLike
from hrp._validation import ensure_dataframe

# quantcore-candidate: new code (HRP stage); parity oracle = scipy single-linkage
# reference leaf order (dev-only).

#: Linkage methods accepted by :func:`linkage_matrix`. ``single`` is the paper
#: default; the others are ablations.
VALID_METHODS: frozenset[str] = frozenset({"single", "ward", "complete", "average"})


def linkage_matrix(dist: MatrixLike, *, method: str = "single") -> np.ndarray:
    r"""Compute a SciPy linkage matrix from a square distance matrix.

    Condenses the square, symmetric distance matrix ``dist`` to the upper-
    triangular vector form expected by ``scipy.cluster.hierarchy.linkage`` and
    runs agglomerative clustering with the requested ``method``.

    HONESTY REQUIREMENT: the validated default is **single** linkage (de Prado
    2016). Silently defaulting to ``ward`` or ``average`` changes the dendrogram
    and the resulting weights; the parity test catches such drift.

    Parameters
    ----------
    dist:
        An ``N x N`` symmetric distance matrix with a zero diagonal.
    method:
        One of :data:`VALID_METHODS`. Defaults to ``"single"``.

    Returns
    -------
    numpy.ndarray
        The ``(N - 1) x 4`` SciPy linkage matrix. Row ``k`` records the two
        clusters merged at step ``k``, their merge distance, and the size of the
        new cluster.

    Raises
    ------
    ValidationError
        If ``dist`` is not square/symmetric, or ``method`` is not in
        :data:`VALID_METHODS`.
    """
    # HONESTY REQUIREMENT: the validated default is single linkage (de Prado
    # 2016). Reject any unrecognized method rather than silently substituting.
    if method not in VALID_METHODS:
        raise ValidationError(
            f"method must be one of {sorted(VALID_METHODS)}, got {method!r}."
        )

    frame = ensure_dataframe(dist, name="dist")
    n_rows, n_cols = frame.shape
    if n_rows != n_cols:
        raise ValidationError(f"dist must be square, got shape {frame.shape}.")

    values = frame.to_numpy(dtype=np.float64)
    if n_rows < 2:
        raise ValidationError("dist must contain at least two assets to cluster.")

    # Symmetry check: linkage requires a symmetric distance matrix with a zero
    # diagonal. Validate before condensing so a malformed input fails loudly.
    if not np.allclose(values, values.T, rtol=1e-7, atol=1e-9):
        raise ValidationError("dist must be symmetric.")
    if not np.allclose(np.diag(values), 0.0, atol=1e-9):
        raise ValidationError("dist must have a zero diagonal.")

    # Lazily import scipy at call time (no import-time side effects). Condense
    # the symmetric matrix to the upper-triangular vector form linkage expects;
    # force exact symmetry/zero-diagonal first so squareform does not complain
    # about residual floating-point asymmetry.
    from scipy.cluster.hierarchy import linkage as _scipy_linkage
    from scipy.spatial.distance import squareform

    symmetric = 0.5 * (values + values.T)
    np.fill_diagonal(symmetric, 0.0)
    condensed = squareform(symmetric, checks=False)

    link = _scipy_linkage(condensed, method=method)
    return np.asarray(link, dtype=np.float64)
