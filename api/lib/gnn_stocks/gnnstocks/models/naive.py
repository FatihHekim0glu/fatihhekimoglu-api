"""Cross-sectional naive baselines — the prediction FLOOR (torch-free numpy).

The honest floor for next-period cross-sectional return RANKING is a simple,
transparent signal computed live (pure numpy, no torch, no ONNX):

- :func:`cross_sectional_momentum` — rank nodes by their trailing momentum (a
  node's own recent cumulative return); the canonical cheap cross-sectional
  predictor that any GNN must beat to be worth the message passing;
- :func:`sector_mean_score` — score each node by its sector block's mean trailing
  return (a pure group-mean signal), the simplest "graph-aware" floor.

A GNN only "beats the baseline" if it lifts OOS rank-IC / long-short decile spread
above the BEST of these with a DM-significant margin AND a positive DSR. On the
synthetic block-factor null these baselines and the GNN both hug zero rank-IC.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from gnnstocks._exceptions import ValidationError
from gnnstocks._typing import FeatureMatrix, FloatArray


def _coerce_feature_matrix(features: FeatureMatrix, *, name: str = "features") -> FloatArray:
    """Coerce ``features`` to a finite 2-D float64 array and validate it.

    Centralizes the shape/dtype/finiteness guard the naive scorers share so both
    entry points reject empty, non-2-D, or non-finite feature matrices identically.

    Parameters
    ----------
    features:
        A ``(n_nodes, n_features)`` per-node feature matrix.
    name:
        Human-readable label used in error messages.

    Returns
    -------
    FloatArray
        A 2-D float64 copy of ``features``.

    Raises
    ------
    ValidationError
        If ``features`` is not 2-dimensional, is empty, or is non-finite.
    """
    arr = np.asarray(features, dtype="float64")
    if arr.ndim != 2:
        raise ValidationError(f"{name} must be 2-dimensional, got ndim={arr.ndim}.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValidationError(f"{name} must have at least one node and one feature column.")
    if not bool(np.isfinite(arr).all()):
        raise ValidationError(f"{name} must be finite (no NaN/inf).")
    return arr


def _resolve_column(n_features: int, momentum_col: int) -> int:
    """Resolve a possibly-negative ``momentum_col`` against ``n_features`` columns.

    Parameters
    ----------
    n_features:
        The feature-matrix width.
    momentum_col:
        The (possibly negative) momentum column index.

    Returns
    -------
    int
        The non-negative resolved column index.

    Raises
    ------
    ValidationError
        If ``momentum_col`` is out of range for ``n_features`` columns.
    """
    resolved = momentum_col + n_features if momentum_col < 0 else momentum_col
    if not 0 <= resolved < n_features:
        raise ValidationError(
            f"momentum_col {momentum_col} is out of range for a feature matrix with "
            f"{n_features} column(s)."
        )
    return resolved


@dataclass(frozen=True, slots=True)
class NaiveScores:
    """Immutable result of a cross-sectional naive scoring over the node cross-section.

    Attributes
    ----------
    name:
        Model identifier (``"naive"`` / ``"momentum"`` / ``"sector_mean"``).
    scores:
        The ``(n_nodes,)`` per-node cross-sectional score (higher = predicted to
        rank higher next period).
    n_nodes:
        Number of nodes scored.
    """

    name: str
    scores: FloatArray
    n_nodes: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        out: dict[str, Any] = asdict(self)
        out["scores"] = [float(x) for x in np.asarray(self.scores).ravel()]
        return out


def cross_sectional_momentum(
    features: FeatureMatrix,
    *,
    momentum_col: int = -1,
) -> NaiveScores:
    """Score each node by its trailing momentum feature (the cheap floor).

    Reads the momentum column out of the per-node feature matrix and uses it
    directly as the cross-sectional score (nodes with higher trailing momentum are
    predicted to rank higher next period). No graph, no training, no torch.

    Parameters
    ----------
    features:
        The ``(n_nodes, n_features)`` per-node feature matrix.
    momentum_col:
        Index of the momentum feature column (default the last column).

    Returns
    -------
    NaiveScores
        The frozen per-node momentum scores.

    Raises
    ------
    ValidationError
        If ``features`` is empty or ``momentum_col`` is out of range.
    """
    arr = _coerce_feature_matrix(features)
    col = _resolve_column(arr.shape[1], momentum_col)
    # The momentum feature IS the cross-sectional score: nodes with higher trailing
    # momentum are predicted to rank higher next period. A copy keeps the result
    # immutable and decoupled from the caller's array.
    scores: FloatArray = np.ascontiguousarray(arr[:, col], dtype="float64").copy()
    return NaiveScores(name="momentum", scores=scores, n_nodes=int(arr.shape[0]))


def sector_mean_score(
    features: FeatureMatrix,
    sector_labels: list[int],
    *,
    momentum_col: int = -1,
) -> NaiveScores:
    """Score each node by its sector block's mean trailing return (group-mean floor).

    Assigns every node the mean of the momentum feature over its sector block — the
    simplest "graph-aware" baseline (a one-hop mean over the sector graph) that the
    GraphSAGE mean-aggregator must beat to justify learned message passing.

    Parameters
    ----------
    features:
        The ``(n_nodes, n_features)`` per-node feature matrix.
    sector_labels:
        Per-node integer sector/block labels, in node order.
    momentum_col:
        Index of the momentum feature column (default the last column).

    Returns
    -------
    NaiveScores
        The frozen per-node sector-mean scores.

    Raises
    ------
    ValidationError
        If ``features`` is empty, labels misalign, or ``momentum_col`` is out of
        range.
    """
    arr = _coerce_feature_matrix(features)
    n_nodes = int(arr.shape[0])
    if len(sector_labels) != n_nodes:
        raise ValidationError(
            f"sector_labels has length {len(sector_labels)} but features has {n_nodes} node(s)."
        )
    col = _resolve_column(arr.shape[1], momentum_col)

    momentum = arr[:, col]
    labels = np.asarray(sector_labels, dtype="int64")
    # Each node's score is its sector block's MEAN momentum — a one-hop mean over
    # the (block) graph, the simplest graph-aware floor the mean-aggregator
    # GraphSAGE must beat. Group means are computed via a stable sum/count over the
    # unique labels so the result is deterministic and order-independent.
    scores = np.empty(n_nodes, dtype="float64")
    for label in np.unique(labels):
        mask = labels == label
        scores[mask] = float(momentum[mask].mean())
    return NaiveScores(name="sector_mean", scores=scores, n_nodes=n_nodes)
