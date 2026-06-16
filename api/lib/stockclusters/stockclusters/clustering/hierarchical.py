"""Agglomerative hierarchical clustering of the correlation network.

Runs ``scipy.cluster.hierarchy`` linkage (average / ward / single) on a Mantegna
distance matrix, then cuts the dendrogram at ``k`` clusters. The frozen
:class:`ClusterResult` is the canonical output bundle carried through the rest of
the pipeline (allocation, stability, evaluation).

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from stockclusters._exceptions import ValidationError
from stockclusters._typing import MatrixLike
from stockclusters._validation import ensure_dataframe

__all__ = ["ClusterResult", "cut_tree", "hierarchical_clusters"]

#: Linkage methods accepted by :func:`hierarchical_clusters`. ``average`` is the
#: reported default for correlation networks; ``ward``/``single`` are ablations.
VALID_METHODS: frozenset[str] = frozenset({"average", "ward", "single"})


def _ensure_square_dist(dist: MatrixLike) -> pd.DataFrame:
    """Coerce ``dist`` to a square, symmetric, zero-diagonal DataFrame."""
    frame = ensure_dataframe(dist, name="dist")
    n_rows, n_cols = frame.shape
    if n_rows != n_cols:
        raise ValidationError(f"dist must be square, got shape {frame.shape}.")
    if not frame.index.equals(frame.columns):
        frame.index = frame.columns
    values = frame.to_numpy(dtype=np.float64)
    if n_rows < 2:
        raise ValidationError("dist must contain at least two assets to cluster.")
    if not np.allclose(values, values.T, rtol=1e-7, atol=1e-9):
        raise ValidationError("dist must be symmetric.")
    return frame


def _silhouette_on_distance(labels: pd.Series, dist: pd.DataFrame) -> float:
    """Mean silhouette of ``labels`` using ``dist`` as a precomputed metric.

    Returns ``nan`` when the silhouette is undefined (fewer than 2 clusters or
    every point in its own cluster), so callers can serialize it to ``None``.
    """
    n_clusters = int(labels.nunique())
    n_samples = int(labels.shape[0])
    if n_clusters < 2 or n_clusters >= n_samples:
        return float("nan")
    from sklearn.metrics import silhouette_score

    aligned = dist.reindex(index=labels.index, columns=labels.index)
    return float(
        silhouette_score(
            aligned.to_numpy(dtype=np.float64),
            labels.to_numpy(),
            metric="precomputed",
        )
    )


@dataclass(frozen=True, slots=True)
class ClusterResult:
    """Immutable result of a clustering of the asset universe.

    Attributes
    ----------
    labels:
        Integer cluster label per asset, indexed by asset ticker.
    n_clusters:
        The number of distinct clusters (``labels.nunique()``).
    method:
        The clustering method used (e.g. ``"hierarchical:average"`` or
        ``"kmeans"``).
    linkage:
        The ``(N - 1) x 4`` SciPy linkage matrix when a hierarchical method was
        used; ``None`` for flat methods (e.g. K-means).
    ordered_assets:
        Asset tickers in dendrogram-leaf order (for the cluster-ordered heatmap);
        falls back to input order for flat methods.
    silhouette:
        Mean silhouette score of ``labels`` on the distance matrix.
    """

    labels: pd.Series
    n_clusters: int
    method: str
    ordered_assets: list[str]
    silhouette: float
    linkage: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result.

        ``labels`` is rendered as an ordered ``{ticker: cluster_id}`` mapping and
        ``linkage`` as a nested list (or ``None``) so the result crosses the API
        boundary without numpy/pandas types leaking through.
        """
        out = asdict(self)
        out["labels"] = {str(k): int(v) for k, v in self.labels.items()}
        out["linkage"] = None if self.linkage is None else np.asarray(self.linkage).tolist()
        out["silhouette"] = float(self.silhouette)
        return out


def hierarchical_clusters(
    dist: MatrixLike,
    *,
    n_clusters: int,
    method: str = "average",
) -> ClusterResult:
    r"""Agglomerative clustering of a distance matrix cut at ``n_clusters``.

    Condenses ``dist`` to vector form, runs ``scipy.cluster.hierarchy.linkage``
    with ``method``, and cuts the resulting tree into exactly ``n_clusters`` flat
    clusters.

    HONESTY REQUIREMENT: clustering fits on the supplied (train-window) distance
    matrix only; the resulting labels are frozen before being applied to any
    out-of-sample window.

    Parameters
    ----------
    dist:
        An ``N x N`` symmetric Mantegna distance matrix.
    n_clusters:
        The number of flat clusters to cut the dendrogram into (``1 <= k <= N``).
    method:
        One of :data:`VALID_METHODS`. Defaults to ``"average"``.

    Returns
    -------
    ClusterResult
        The frozen clustering bundle.

    Raises
    ------
    ValidationError
        If ``dist`` is not square/symmetric, ``method`` is unknown, or
        ``n_clusters`` is out of range.
    """
    if method not in VALID_METHODS:
        raise ValidationError(f"method must be one of {sorted(VALID_METHODS)}, got {method!r}.")
    frame = _ensure_square_dist(dist)
    labels_list = list(frame.columns.astype(str))
    n = len(labels_list)
    if not 1 <= int(n_clusters) <= n:
        raise ValidationError(f"n_clusters must satisfy 1 <= k <= {n}, got {n_clusters}.")

    # Lazy scipy import (no import-time side effects). Condense the symmetric
    # matrix to vector form and run agglomerative linkage with ``method``.
    from scipy.cluster.hierarchy import leaves_list
    from scipy.cluster.hierarchy import linkage as _scipy_linkage
    from scipy.spatial.distance import squareform

    values = frame.to_numpy(dtype=np.float64)
    symmetric = 0.5 * (values + values.T)
    np.fill_diagonal(symmetric, 0.0)
    condensed = squareform(symmetric, checks=False)
    link = np.asarray(_scipy_linkage(condensed, method=method), dtype=np.float64)

    labels = cut_tree(link, n_clusters=int(n_clusters), labels=labels_list)
    ordered = [labels_list[i] for i in leaves_list(link).tolist()]
    silhouette = _silhouette_on_distance(labels, frame)

    return ClusterResult(
        labels=labels,
        n_clusters=int(labels.nunique()),
        method=f"hierarchical:{method}",
        ordered_assets=ordered,
        silhouette=silhouette,
        linkage=link,
    )


def cut_tree(linkage: np.ndarray, *, n_clusters: int, labels: list[str]) -> pd.Series:
    r"""Cut a SciPy linkage matrix into ``n_clusters`` flat clusters.

    Thin wrapper over ``scipy.cluster.hierarchy.fcluster`` with
    ``criterion="maxclust"`` that returns a label Series indexed by asset.

    MONOTONICITY REQUIREMENT: ``n_clusters = N`` yields singletons and
    ``n_clusters = 1`` yields a single cluster; a higher cut never increases the
    cluster count (property-tested).

    Parameters
    ----------
    linkage:
        The ``(N - 1) x 4`` SciPy linkage matrix.
    n_clusters:
        The number of flat clusters to produce.
    labels:
        Asset tickers, in the original (pre-linkage) order, used to index the
        returned Series.

    Returns
    -------
    pandas.Series
        Integer cluster labels indexed by asset ticker.

    Raises
    ------
    ValidationError
        If ``n_clusters`` is out of range or ``labels`` length mismatches.
    """
    link = np.asarray(linkage, dtype=np.float64)
    n = link.shape[0] + 1
    if len(labels) != n:
        raise ValidationError(
            f"labels has length {len(labels)} but the linkage encodes {n} leaves."
        )
    if not 1 <= int(n_clusters) <= n:
        raise ValidationError(f"n_clusters must satisfy 1 <= k <= {n}, got {n_clusters}.")

    from scipy.cluster.hierarchy import fcluster

    flat = fcluster(link, t=int(n_clusters), criterion="maxclust")
    return pd.Series(np.asarray(flat, dtype=int), index=list(labels), name="cluster")
