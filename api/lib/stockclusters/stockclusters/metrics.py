"""Cluster-quality and post-hoc diagnostic metrics.

Internal-validity metrics (silhouette, cophenetic correlation, modularity) and the
post-hoc ARI-vs-GICS comparison.

HONESTY REQUIREMENT: GICS sector membership is used ONLY post-hoc, to report how
much the correlation clusters re-discover sectors (the expected ARI ~0.4-0.7). GICS
NEVER enters the distance computation or the ``k`` selection.

PARITY: ``silhouette_score`` matches ``sklearn.metrics`` to ``1e-10``.

Importing this module has no side effects.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from stockclusters._exceptions import ValidationError
from stockclusters._typing import MatrixLike

__all__ = [
    "ari_vs_gics",
    "cophenetic_correlation",
    "modularity",
    "silhouette_score",
]


def _as_labelled_matrix(mat: MatrixLike, *, name: str) -> tuple[np.ndarray, list[str]]:
    """Coerce a square matrix to ``(ndarray, labels)``, validating squareness."""
    if isinstance(mat, pd.DataFrame):
        labels = [str(c) for c in mat.columns]
        arr = mat.to_numpy(dtype=np.float64)
    else:
        arr = np.asarray(mat, dtype=np.float64)
        labels = [str(i) for i in range(arr.shape[0])] if arr.ndim == 2 else []
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValidationError(f"{name} must be a square 2-D matrix, got shape {arr.shape}.")
    return arr, labels


def silhouette_score(dist: MatrixLike, labels: pd.Series) -> float:
    r"""Mean silhouette score of a labeling on a precomputed distance matrix.

    Uses the silhouette definition with ``metric="precomputed"``: for each asset,
    ``(b - a) / max(a, b)`` where ``a`` is mean intra-cluster distance and ``b`` the
    mean nearest-other-cluster distance, averaged over assets.

    PARITY REQUIREMENT: matches ``sklearn.metrics.silhouette_score`` (precomputed)
    to ``1e-10``.

    Parameters
    ----------
    dist:
        An ``N x N`` symmetric distance matrix.
    labels:
        Integer cluster labels indexed by asset (aligned to ``dist``).

    Returns
    -------
    float
        The mean silhouette score in ``[-1, 1]``.

    Raises
    ------
    ValidationError
        If there are fewer than two clusters or labels/dist disagree.
    """
    arr, mat_labels = _as_labelled_matrix(dist, name="dist")
    if not isinstance(labels, pd.Series):
        raise ValidationError("labels must be a pandas Series indexed by asset.")
    n = arr.shape[0]
    if int(labels.shape[0]) != n:
        raise ValidationError(f"labels length ({labels.shape[0]}) must match dist size ({n}).")
    # Align labels to the matrix order when the matrix carries asset labels.
    if mat_labels and set(mat_labels) == set(str(i) for i in labels.index):
        label_arr = labels.reindex([*mat_labels]).to_numpy()
    else:
        label_arr = labels.to_numpy()

    unique = pd.unique(label_arr)
    n_clusters = len(unique)
    if n_clusters < 2:
        raise ValidationError(f"silhouette_score requires at least two clusters, got {n_clusters}.")
    if n_clusters >= n:
        # Every point its own cluster: silhouette is undefined per sklearn (a==0,
        # b across singletons); sklearn returns the mean over the per-sample values
        # which are all defined here only when at least one cluster has >1 member.
        raise ValidationError("silhouette_score requires 2 <= n_clusters <= n_samples - 1.")

    # Cluster index arrays for vectorized intra/inter distances.
    clusters: dict[object, np.ndarray] = {c: np.flatnonzero(label_arr == c) for c in unique}
    sizes = {c: idx.size for c, idx in clusters.items()}

    sil = np.empty(n, dtype=np.float64)
    for i in range(n):
        ci = label_arr[i]
        own = clusters[ci]
        if sizes[ci] > 1:
            # Mean distance to other members of own cluster (exclude self).
            a_i = float(arr[i, own].sum()) / (sizes[ci] - 1)
        else:
            # Singleton cluster: sklearn defines its silhouette as 0.
            sil[i] = 0.0
            continue
        b_i = np.inf
        for c, idx in clusters.items():
            if c == ci:
                continue
            b_c = float(arr[i, idx].mean())
            if b_c < b_i:
                b_i = b_c
        denom = max(a_i, b_i)
        sil[i] = 0.0 if denom == 0.0 else (b_i - a_i) / denom

    return float(sil.mean())


def cophenetic_correlation(linkage: np.ndarray, dist: MatrixLike) -> float:
    r"""Cophenetic correlation between a linkage tree and the original distances.

    Pearson correlation between the condensed original distances and the
    cophenetic distances implied by ``linkage`` — a measure of how faithfully the
    dendrogram preserves pairwise distances.

    PARITY REQUIREMENT: matches ``scipy.cluster.hierarchy.cophenet`` to ``1e-10``.

    Parameters
    ----------
    linkage:
        The ``(N - 1) x 4`` SciPy linkage matrix.
    dist:
        The ``N x N`` distance matrix the linkage was built from.

    Returns
    -------
    float
        The cophenetic correlation coefficient.

    Raises
    ------
    ValidationError
        If shapes are inconsistent.
    """
    # Lazy import: keep SciPy off this pure module's import path.
    from scipy.cluster.hierarchy import cophenet
    from scipy.spatial.distance import squareform

    link_arr = np.asarray(linkage, dtype=np.float64)
    arr, _ = _as_labelled_matrix(dist, name="dist")
    n = arr.shape[0]
    if link_arr.ndim != 2 or link_arr.shape[1] != 4 or link_arr.shape[0] != n - 1:
        raise ValidationError(
            f"linkage must be ({n - 1}, 4) for an {n}-asset distance matrix, got {link_arr.shape}."
        )

    # Condense the (symmetric, zero-diagonal) distance matrix to vector form. Force
    # an exact zero diagonal so squareform's checksum does not reject FP drift.
    sym = 0.5 * (arr + arr.T)
    np.fill_diagonal(sym, 0.0)
    condensed = squareform(sym, checks=False)

    coph_corr, _ = cophenet(link_arr, condensed)
    return float(coph_corr)


def modularity(labels: pd.Series, corr: MatrixLike) -> float:
    r"""Newman modularity of a labeling over a correlation-weighted graph.

    Treats the (thresholded, non-negative) correlation matrix as a weighted graph
    and computes the modularity ``Q`` of the cluster partition — the fraction of
    within-cluster edge weight minus its expectation under a degree-preserving
    null. Reported as an MST/network cross-check on the gap selection.

    Parameters
    ----------
    labels:
        Integer cluster labels indexed by asset.
    corr:
        The ``N x N`` correlation matrix, labelled by asset.

    Returns
    -------
    float
        The modularity ``Q`` (higher = stronger community structure).

    Raises
    ------
    ValidationError
        If ``labels`` and ``corr`` labels disagree.
    """
    if not isinstance(labels, pd.Series):
        raise ValidationError("labels must be a pandas Series indexed by asset.")
    arr, mat_labels = _as_labelled_matrix(corr, name="corr")
    n = arr.shape[0]
    if int(labels.shape[0]) != n:
        raise ValidationError(f"labels length ({labels.shape[0]}) must match corr size ({n}).")
    if mat_labels and set(mat_labels) == set(str(i) for i in labels.index):
        label_arr = labels.reindex([*mat_labels]).to_numpy()
    else:
        label_arr = labels.to_numpy()

    # Build the weighted adjacency: non-negative correlations as edge weights, no
    # self-loops. Negative correlations are thresholded to zero (Newman modularity
    # is defined for a non-negative weighted graph).
    weights = np.clip(0.5 * (arr + arr.T), 0.0, None)
    np.fill_diagonal(weights, 0.0)

    total = float(weights.sum())  # = 2m (each edge counted twice)
    if total <= 0.0:
        # No positive edges: a degenerate graph has zero community structure.
        return 0.0

    degree = weights.sum(axis=1)
    # Same-cluster indicator (delta_{c_i, c_j}).
    same = label_arr[:, None] == label_arr[None, :]
    actual = float(weights[same].sum())
    expected = float((np.outer(degree, degree) / total)[same].sum())
    return (actual - expected) / total


def ari_vs_gics(labels: pd.Series, gics: Mapping[str, str]) -> float:
    r"""Post-hoc Adjusted Rand Index between clusters and GICS sectors.

    HONESTY REQUIREMENT: this is a POST-HOC diagnostic only. GICS labels never
    enter the distance matrix or the ``k`` selection; this just measures how much
    the correlation clusters re-discover the sector taxonomy (expected
    ARI ~0.4-0.7).

    Parameters
    ----------
    labels:
        Integer cluster labels indexed by asset ticker.
    gics:
        A mapping ``{ticker: gics_sector}`` for (a superset of) the clustered
        assets.

    Returns
    -------
    float
        The Adjusted Rand Index between the clusters and the GICS sectors on their
        shared assets.

    Raises
    ------
    ValidationError
        If fewer than two assets have a GICS sector.
    """
    if not isinstance(labels, pd.Series):
        raise ValidationError("labels must be a pandas Series indexed by asset ticker.")

    # Restrict to assets that appear in BOTH the labeling and the GICS map, in the
    # label index order, so the two partitions are over the identical asset set.
    shared = [str(t) for t in labels.index if str(t) in gics]
    if len(shared) < 2:
        raise ValidationError(
            f"ari_vs_gics needs at least two assets with a GICS sector, got {len(shared)}."
        )

    cluster_labels = pd.Series([int(labels.loc[t]) for t in shared], index=shared, dtype=int)
    # Factorize the (string) GICS sectors to integer codes for the ARI.
    sectors = pd.Series([gics[t] for t in shared], index=shared)
    sector_codes = pd.Series(pd.factorize(sectors)[0], index=shared, dtype=int)

    # Reuse the library's parity-tested ARI (matches sklearn to 1e-12).
    from stockclusters.stability.ari import adjusted_rand_index

    return adjusted_rand_index(cluster_labels, sector_codes)
