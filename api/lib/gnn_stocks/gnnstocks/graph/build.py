"""Per-fold adjacency builders (TRAIN-window only — the no-future-graph guard).

Three adjacency builders, all rebuilt PER FOLD from train-window data ONLY so no
future return ever touches a fold's graph (a Hypothesis test asserts perturbing
post-train returns leaves the adjacency byte-identical):

- :func:`corr_knn_adjacency` — a symmetric k-nearest-neighbour graph on the
  train-window return CORRELATION: each node keeps edges to its ``k`` most
  correlated peers, symmetrized by union so the adjacency is undirected;
- :func:`sector_adjacency` — a block-membership graph: nodes in the same sector
  block are connected (the DESCRIPTIVE structure the synthetic panel encodes);
- :func:`empty_adjacency` — the no-edges ablation (identity-only after the
  self-loop is added in normalization), isolating whether message passing helps.

Each builder returns a dense ``(n_nodes, n_nodes)`` symmetric adjacency with a zero
diagonal; the self-loop ``A + I`` and the degree normalization are applied later by
:func:`gnnstocks.graph.normalize.normalize_adjacency`. Importing this module has no
side effects.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from gnnstocks._exceptions import InsufficientDataError, ValidationError
from gnnstocks._typing import AdjacencyMatrix
from gnnstocks._validation import ensure_dataframe

# quantcore-candidate: correlation distance mirrors
# hrp-portfolio:cluster/distance.py (sqrt(0.5 * (1 - corr))), thresholded to k-NN.

#: Minimum train-window length for a defensible pairwise correlation estimate.
_MIN_CORR_OBS: int = 2


def corr_knn_adjacency(
    train_returns: pd.DataFrame,
    *,
    k: int = 8,
) -> AdjacencyMatrix:
    """Build a symmetric k-NN adjacency on the TRAIN-window return correlation.

    Computes the node-by-node return correlation over the TRAIN window ONLY, keeps
    each node's edges to its ``k`` most correlated peers, and symmetrizes by union
    (an edge survives if either endpoint selected it) so the adjacency is
    undirected with a zero diagonal. NO future data enters: a Hypothesis test
    asserts the result is byte-identical when post-train returns are perturbed.

    Parameters
    ----------
    train_returns:
        The TRAIN-window wide returns panel (rows = time, columns = node).
    k:
        Number of nearest neighbours each node connects to (``1 <= k < n_nodes``).

    Returns
    -------
    AdjacencyMatrix
        A dense ``(n_nodes, n_nodes)`` symmetric 0/1 adjacency, zero diagonal.

    Raises
    ------
    ValidationError
        If the panel is empty, has fewer than two nodes, or ``k`` is out of range.
    InsufficientDataError
        If the train window is too short to estimate a correlation.
    """
    frame = ensure_dataframe(train_returns, name="train_returns")
    n_nodes = int(frame.shape[1])
    if n_nodes < 2:
        raise ValidationError(
            f"corr_knn_adjacency: need at least 2 nodes to build a k-NN graph, got {n_nodes}."
        )
    if k < 1 or k >= n_nodes:
        raise ValidationError(
            f"corr_knn_adjacency: k must satisfy 1 <= k < n_nodes ({n_nodes}), got {k}."
        )
    n_obs = int(frame.shape[0])
    if n_obs < _MIN_CORR_OBS:
        raise InsufficientDataError(
            f"corr_knn_adjacency: train window has {n_obs} observation(s) but at least "
            f"{_MIN_CORR_OBS} are required to estimate a correlation."
        )

    # Pearson correlation over the TRAIN window ONLY. ``np.corrcoef`` divides by the
    # per-node standard deviation; a zero-variance (constant) node yields NaN, which
    # we treat as "uncorrelated with everything" (0.0) so the k-NN selection stays
    # well-defined and deterministic.
    values = frame.to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(values, rowvar=False)
    corr = np.asarray(corr, dtype=np.float64).reshape(n_nodes, n_nodes)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    # A node is never its own neighbour: mask the diagonal below any real value so
    # ``argsort`` never selects self. Symmetrize the (numerically) symmetric corr to
    # kill float asymmetry so the neighbour ranking is order-stable.
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, -np.inf)

    # Per row, the k most correlated peers (largest correlation). ``argsort`` is
    # ascending, so the last ``k`` indices are the top-k; this is fully determined by
    # the train window and independent of any post-train data.
    directed = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    neighbour_order = np.argsort(corr, axis=1, kind="stable")
    topk = neighbour_order[:, -k:]
    rows = np.repeat(np.arange(n_nodes), k)
    directed[rows, topk.reshape(-1)] = 1.0

    # Symmetrize by UNION: an undirected edge survives if EITHER endpoint selected
    # the other. ``maximum`` of the directed matrix and its transpose is the union.
    adjacency: AdjacencyMatrix = np.maximum(directed, directed.T)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def sector_adjacency(sector_labels: Sequence[int] | AdjacencyMatrix) -> AdjacencyMatrix:
    """Build a block-membership adjacency: same-sector nodes are connected.

    Two nodes share an edge iff they belong to the same sector block. This is the
    DESCRIPTIVE structure the synthetic block-factor panel encodes by construction;
    it uses ONLY the (static, known) sector labels, never any return data, so it is
    trivially leakage-free.

    Parameters
    ----------
    sector_labels:
        Per-node integer sector/block labels, in node (column) order.

    Returns
    -------
    AdjacencyMatrix
        A dense ``(n_nodes, n_nodes)`` symmetric 0/1 adjacency (block-diagonal up
        to node ordering), zero diagonal.

    Raises
    ------
    ValidationError
        If ``sector_labels`` is empty or non-integer.
    """
    labels = np.asarray(sector_labels)
    if labels.ndim != 1:
        raise ValidationError(
            f"sector_adjacency: sector_labels must be 1-dimensional, got ndim={labels.ndim}."
        )
    if labels.size == 0:
        raise ValidationError("sector_adjacency: sector_labels must be non-empty.")
    if not np.issubdtype(labels.dtype, np.integer):
        # Float labels are accepted only if they are exact integers (e.g. ``2.0``);
        # genuine non-integer labels (``2.5``) are a programming error.
        if not np.issubdtype(labels.dtype, np.floating) or not np.all(
            labels == np.floor(labels)
        ):
            raise ValidationError(
                f"sector_adjacency: sector_labels must be integer, got dtype {labels.dtype}."
            )
        labels = labels.astype(np.int64)

    # Two nodes are connected iff their labels match. The outer equality is the
    # block-membership matrix; zero the diagonal so a node has no self-edge (the
    # self-loop is added later in :func:`normalize_adjacency`).
    adjacency = (labels[:, np.newaxis] == labels[np.newaxis, :]).astype(np.float64)
    np.fill_diagonal(adjacency, 0.0)
    result: AdjacencyMatrix = adjacency
    return result


def empty_adjacency(n_nodes: int) -> AdjacencyMatrix:
    """Build the no-edges ablation adjacency (all zeros, zero diagonal).

    With no edges, the normalized adjacency reduces to the identity (only the
    self-loop ``A + I`` survives), so the GCN / GraphSAGE collapse to a per-node
    MLP — the ablation that isolates whether message passing adds anything over a
    node-only model. This is the leakage-free floor for the graph arm.

    Parameters
    ----------
    n_nodes:
        The node count.

    Returns
    -------
    AdjacencyMatrix
        A dense ``(n_nodes, n_nodes)`` all-zeros adjacency.

    Raises
    ------
    ValidationError
        If ``n_nodes < 1``.
    """
    if n_nodes < 1:
        raise ValidationError(f"empty_adjacency: n_nodes must be >= 1, got {n_nodes}.")
    adjacency: AdjacencyMatrix = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    return adjacency
