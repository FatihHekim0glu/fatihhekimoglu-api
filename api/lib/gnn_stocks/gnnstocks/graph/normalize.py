"""Symmetric adjacency normalization ``Â = D^{-1/2}(A + I)D^{-1/2}`` (Kipf-Welling).

The dense-adjacency GCN / GraphSAGE consume the SYMMETRICALLY NORMALIZED adjacency
with self-loops, exactly as in Kipf & Welling (2017):

.. math::

    \\hat{A} = \\tilde{D}^{-1/2}\\,\\tilde{A}\\,\\tilde{D}^{-1/2},
    \\qquad \\tilde{A} = A + I,
    \\quad \\tilde{D}_{ii} = \\sum_j \\tilde{A}_{ij}.

Adding the self-loop ``A + I`` keeps a node's own features in the aggregation; the
symmetric degree scaling keeps the spectral radius bounded so stacked layers stay
stable. The normalized adjacency is an ONNX INPUT (recomputed per rebalance from
the per-fold graph); the model weights are baked into the exported graph. This
function is validated against a hand-computed reference Â in the parity suite.

Importing this module has no side effects.
"""

from __future__ import annotations

import numpy as np

from gnnstocks._exceptions import ValidationError
from gnnstocks._typing import AdjacencyMatrix


def normalize_adjacency(adjacency: AdjacencyMatrix, *, add_self_loops: bool = True) -> AdjacencyMatrix:
    r"""Return the symmetric normalized adjacency ``Â = D^{-1/2}(A + I)D^{-1/2}``.

    Adds a self-loop (``A + I`` when ``add_self_loops``), computes the (self-looped)
    degree ``D``, and applies the symmetric scaling ``D^{-1/2} Ã D^{-1/2}``. Rows
    with zero degree are left at zero (their ``D^{-1/2}`` is floored to zero rather
    than dividing by zero), which only arises when self-loops are disabled on an
    isolated node. The result is symmetric whenever ``A`` is symmetric.

    Parameters
    ----------
    adjacency:
        A dense ``(n_nodes, n_nodes)`` adjacency (symmetric for the kNN/sector
        builders), with any diagonal (the self-loop is added here when requested).
    add_self_loops:
        If ``True`` (default), add ``I`` before normalizing (the GCN convention).

    Returns
    -------
    AdjacencyMatrix
        The dense ``(n_nodes, n_nodes)`` symmetric normalized adjacency Â.

    Raises
    ------
    ValidationError
        If ``adjacency`` is not a square 2-D matrix or contains non-finite values.
    """
    a = np.asarray(adjacency, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValidationError(
            f"normalize_adjacency: adjacency must be a square 2-D matrix, got shape {a.shape}."
        )
    if a.shape[0] == 0:
        raise ValidationError("normalize_adjacency: adjacency must be non-empty.")
    if not np.isfinite(a).all():
        raise ValidationError("normalize_adjacency: adjacency contains non-finite values.")

    n = a.shape[0]
    tilde = a + np.eye(n, dtype=np.float64) if add_self_loops else a.copy()

    degree = tilde.sum(axis=1)
    # D^{-1/2} with a zero floor for isolated nodes (avoids divide-by-zero); an
    # isolated node only occurs when self-loops are disabled.
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.where(degree > 0.0, 1.0 / np.sqrt(degree), 0.0)

    # Â = D^{-1/2} Ã D^{-1/2} via row/column scaling (a diagonal congruence).
    normalized: AdjacencyMatrix = d_inv_sqrt[:, np.newaxis] * tilde * d_inv_sqrt[np.newaxis, :]
    return normalized
