"""K-means clustering on the RMT-signal embedding.

K-means runs on the eigenvector embedding from
:mod:`stockclusters.clustering.embedding`, NOT on the raw Mantegna distances
(K-means assumes Euclidean geometry, which the embedding provides and the distance
matrix does not). Results are returned as a :class:`ClusterResult`.

REPRODUCIBILITY: seeding flows through ``seed`` so labels and inertia are
deterministic. The parity oracle pins **inertia** (not label identity) against
``sklearn`` under a fixed explicit init.

Importing this module has no side effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockclusters._exceptions import ValidationError
from stockclusters._typing import MatrixLike
from stockclusters._validation import ensure_dataframe
from stockclusters.clustering.hierarchical import ClusterResult

__all__ = ["kmeans_clusters"]


def kmeans_clusters(
    embedding: MatrixLike,
    *,
    n_clusters: int,
    seed: int = 0,
    n_init: int = 10,
    init: np.ndarray | str = "k-means++",
) -> ClusterResult:
    r"""K-means clustering on an RMT-signal embedding.

    Runs ``sklearn.cluster.KMeans`` on the Euclidean ``embedding`` (rows = assets)
    and packages the result as a :class:`ClusterResult` with ``method="kmeans"``
    and ``linkage=None``.

    HONESTY REQUIREMENT: K-means runs on the embedding, never on the raw distance
    matrix. Determinism is seeded so a fixed ``seed`` reproduces the same inertia.

    Parameters
    ----------
    embedding:
        An ``N x d`` Euclidean embedding (output of
        :func:`stockclusters.clustering.embedding.rmt_signal_embedding`).
    n_clusters:
        The number of clusters ``k``.
    seed:
        Master seed for K-means initialization.
    n_init:
        Number of random initializations; the lowest-inertia run is kept. Forced
        to ``1`` when an explicit ``init`` array is supplied (sklearn requirement
        and the regime the inertia parity oracle pins).
    init:
        K-means initialization. Defaults to ``"k-means++"``; an ``(k, d)`` array
        gives a fixed explicit init (the parity-oracle regime, ``n_init=1``).

    Returns
    -------
    ClusterResult
        The frozen clustering bundle (``linkage=None``).

    Raises
    ------
    ValidationError
        If ``embedding`` is empty or ``n_clusters`` is out of range.
    """
    frame = ensure_dataframe(embedding, name="embedding")
    n_samples, _ = frame.shape
    assets = list(frame.index.astype(str))
    if not 1 <= int(n_clusters) <= n_samples:
        raise ValidationError(f"n_clusters must satisfy 1 <= k <= {n_samples}, got {n_clusters}.")

    x = frame.to_numpy(dtype=np.float64)

    from sklearn.cluster import KMeans

    # An explicit init array must run with n_init=1 (sklearn requirement) and pins
    # a deterministic inertia for the parity oracle.
    effective_n_init = 1 if isinstance(init, np.ndarray) else int(n_init)
    model = KMeans(
        n_clusters=int(n_clusters),
        init=init,
        n_init=effective_n_init,
        random_state=int(seed),
    )
    raw = model.fit_predict(x)

    labels = pd.Series(np.asarray(raw, dtype=int), index=assets, name="cluster")

    # Silhouette on the Euclidean embedding (defined only for 2..n-1 clusters).
    n_unique = int(labels.nunique())
    if 2 <= n_unique < n_samples:
        from sklearn.metrics import silhouette_score

        silhouette = float(silhouette_score(x, raw, metric="euclidean"))
    else:
        silhouette = float("nan")

    return ClusterResult(
        labels=labels,
        n_clusters=n_unique,
        method="kmeans",
        ordered_assets=assets,
        silhouette=silhouette,
        linkage=None,
        meta={"inertia": float(model.inertia_)},
    )
