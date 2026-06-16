"""Clustering of the correlation network.

Two families of clustering live here, both consuming the Mantegna distance / RMT
signal from :mod:`stockclusters.correlation`:

- :mod:`stockclusters.clustering.hierarchical` — SciPy agglomerative clustering
  (average / ward / single linkage) on the distance matrix.
- :mod:`stockclusters.clustering.embedding` — an RMT-signal eigenvector embedding
  of the correlation matrix.
- :mod:`stockclusters.clustering.kmeans` — K-means ON that embedding (not on the
  raw distances).
- :mod:`stockclusters.clustering.selection` — gap statistic vs a phase-randomized
  null with the Tibshirani 1-SE rule (silhouette + MST modularity reported as
  cross-checks); records ALL trials for the DSR trial count.

The frozen :class:`ClusterResult` is the common return bundle.

Importing this subpackage has no side effects.
"""

from __future__ import annotations

from stockclusters.clustering.embedding import rmt_signal_embedding
from stockclusters.clustering.hierarchical import (
    ClusterResult,
    cut_tree,
    hierarchical_clusters,
)
from stockclusters.clustering.kmeans import kmeans_clusters
from stockclusters.clustering.selection import GapResult, select_k_gap

__all__ = [
    "ClusterResult",
    "GapResult",
    "cut_tree",
    "hierarchical_clusters",
    "kmeans_clusters",
    "rmt_signal_embedding",
    "select_k_gap",
]
