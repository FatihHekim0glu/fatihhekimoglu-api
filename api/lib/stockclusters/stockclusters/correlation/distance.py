r"""Mantegna correlation distance and network helpers.

The Mantegna (1999) distance

.. math::

    d_{ij} = \sqrt{2\,(1 - \rho_{ij})}

is a TRUE metric on the unit sphere: it satisfies non-negativity, symmetry, the
identity of indiscernibles (zero diagonal), and the triangle inequality. It is the
distance used to build correlation-based minimum spanning trees and the
subdominant ultrametric (single-linkage hierarchy).

HONESTY REQUIREMENT: the metric is :math:`\sqrt{2(1 - \rho)}`, NOT ``1 - rho`` and
NOT ``1 - |rho|``. The metric axioms are asserted with a Hypothesis property test.

Importing this module has no side effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockclusters._exceptions import ValidationError
from stockclusters._typing import MatrixLike
from stockclusters._validation import ensure_dataframe

__all__ = [
    "mantegna_distance",
    "minimum_spanning_tree",
    "subdominant_ultrametric",
]


def _ensure_square(matrix: MatrixLike, *, name: str) -> pd.DataFrame:
    """Coerce ``matrix`` to a square DataFrame with matching index/columns."""
    frame = ensure_dataframe(matrix, name=name)
    n_rows, n_cols = frame.shape
    if n_rows != n_cols:
        raise ValidationError(f"{name} must be square, got shape {frame.shape}.")
    if not frame.index.equals(frame.columns):
        frame.index = frame.columns
    return frame


def mantegna_distance(corr: MatrixLike) -> pd.DataFrame:
    r"""Mantegna distance :math:`d_{ij} = \sqrt{2(1 - \rho_{ij})}`.

    Maps a correlation matrix to a true-metric distance matrix:
    :math:`\rho = +1 \to d = 0`, :math:`\rho = 0 \to d = \sqrt{2}`,
    :math:`\rho = -1 \to d = 2`.

    HONESTY REQUIREMENT: this is :math:`\sqrt{2(1 - \rho)}`, the Mantegna metric —
    NOT ``1 - rho`` (not a metric) and NOT ``1 - |rho|`` (collapses anti-correlated
    pairs). The triangle inequality and the other metric axioms are property-tested.

    Parameters
    ----------
    corr:
        An ``N x N`` correlation matrix with unit diagonal and entries in
        ``[-1, 1]``.

    Returns
    -------
    pandas.DataFrame
        An ``N x N`` symmetric distance matrix with an exactly zero diagonal,
        labelled by asset.

    Raises
    ------
    ValidationError
        If ``corr`` is not square or contains entries outside ``[-1, 1]``.
    """
    frame = _ensure_square(corr, name="corr")

    values = frame.to_numpy(dtype=np.float64)
    # Domain check: a correlation matrix lives in [-1, 1]; a small tolerance
    # absorbs floating-point drift from estimation.
    tol = 1e-8
    if float(np.nanmin(values)) < -1.0 - tol or float(np.nanmax(values)) > 1.0 + tol:
        raise ValidationError("corr contains entries outside [-1, 1].")

    # HONESTY REQUIREMENT: the Mantegna metric is sqrt(2 * (1 - rho)) — NOT
    # ``1 - rho`` (not a metric) and NOT ``1 - |rho|`` (collapses anti-correlation).
    # Clip first so 1 - rho never dips below zero under the sqrt from rounding.
    clipped = np.clip(values, -1.0, 1.0)
    dist = np.sqrt(2.0 * (1.0 - clipped))

    # Exact symmetry (rho symmetric => d symmetric) and an exactly zero diagonal.
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)

    return pd.DataFrame(dist, index=frame.index, columns=frame.columns)


def minimum_spanning_tree(dist: MatrixLike) -> pd.DataFrame:
    r"""Minimum spanning tree edge list of a distance matrix.

    Builds the MST of the fully-connected distance graph (Mantegna's asset graph).
    The MST is the backbone of the correlation network and the basis of the
    subdominant ultrametric (single linkage).

    Parameters
    ----------
    dist:
        An ``N x N`` symmetric distance matrix (output of
        :func:`mantegna_distance`).

    Returns
    -------
    pandas.DataFrame
        Edge list with columns ``["source", "target", "weight"]`` (``N - 1`` rows),
        sorted by ascending weight.

    Raises
    ------
    ValidationError
        If ``dist`` is not square/symmetric.
    """
    frame = _ensure_square(dist, name="dist")
    values = frame.to_numpy(dtype=np.float64)
    if not np.allclose(values, values.T, rtol=1e-7, atol=1e-9):
        raise ValidationError("dist must be symmetric.")

    labels = list(frame.columns.astype(str))
    n = len(labels)
    if n < 2:
        raise ValidationError("dist must contain at least two assets for an MST.")

    # Lazily import scipy at call time (no import-time side effects). The MST of
    # the dense distance graph is the correlation-network backbone.
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree as _mst

    symmetric = 0.5 * (values + values.T)
    tree = _mst(csr_matrix(symmetric)).toarray()

    sources: list[str] = []
    targets: list[str] = []
    weights: list[float] = []
    rows, cols = np.nonzero(tree)
    for i, j in zip(rows.tolist(), cols.tolist(), strict=True):
        sources.append(labels[i])
        targets.append(labels[j])
        weights.append(float(tree[i, j]))

    edges = pd.DataFrame({"source": sources, "target": targets, "weight": weights})
    edges = edges.sort_values("weight", kind="stable").reset_index(drop=True)
    return edges


def subdominant_ultrametric(dist: MatrixLike) -> pd.DataFrame:
    r"""Subdominant ultrametric (single-linkage cophenetic) distance matrix.

    The subdominant ultrametric is the largest ultrametric dominated by ``dist``;
    it equals the single-linkage cophenetic distance and underlies the MST-based
    hierarchical clustering of the correlation network.

    Parameters
    ----------
    dist:
        An ``N x N`` symmetric distance matrix (output of
        :func:`mantegna_distance`).

    Returns
    -------
    pandas.DataFrame
        The ``N x N`` ultrametric distance matrix, symmetric with a zero diagonal,
        labelled by asset.

    Raises
    ------
    ValidationError
        If ``dist`` is not square/symmetric.
    """
    frame = _ensure_square(dist, name="dist")
    values = frame.to_numpy(dtype=np.float64)
    if not np.allclose(values, values.T, rtol=1e-7, atol=1e-9):
        raise ValidationError("dist must be symmetric.")

    n = values.shape[0]
    if n < 2:
        raise ValidationError("dist must contain at least two assets.")

    # The subdominant ultrametric IS the single-linkage cophenetic distance: the
    # largest ultrametric dominated by ``dist``. Compute it via scipy single
    # linkage + cophenet (lazy import; no import-time side effects).
    from scipy.cluster.hierarchy import cophenet
    from scipy.cluster.hierarchy import linkage as _scipy_linkage
    from scipy.spatial.distance import squareform

    symmetric = 0.5 * (values + values.T)
    np.fill_diagonal(symmetric, 0.0)
    condensed = squareform(symmetric, checks=False)

    link = _scipy_linkage(condensed, method="single")
    copheneticv = cophenet(link)
    ultra = squareform(np.asarray(copheneticv, dtype=np.float64))
    np.fill_diagonal(ultra, 0.0)

    return pd.DataFrame(ultra, index=frame.index, columns=frame.columns)
