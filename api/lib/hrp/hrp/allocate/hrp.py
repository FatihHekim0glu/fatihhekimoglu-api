"""Hierarchical Risk Parity allocation (Lopez de Prado, 2016).

Stage 3 of HRP: recursive bisection. ``getRecBipart`` walks top-down through the
quasi-diagonalized asset order, splitting each contiguous block in two and
allocating between the two halves inversely to their cluster variances, where the
intra-cluster weights used to compute each cluster's variance are themselves
**inverse-variance** weights (``getClusterVar``).

:func:`hrp_allocate` is the public entry point: returns -> covariance ->
correlation -> distance -> linkage -> quasi-diag -> recursive bisection ->
:class:`HRPResult`.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from hrp._exceptions import ValidationError
from hrp._typing import MatrixLike, ReturnsLike
from hrp.cluster.distance import correl_dist, euclidean_codistance
from hrp.cluster.linkage import linkage_matrix
from hrp.cluster.quasidiag import get_quasi_diag
from hrp.estimators.covariance import ledoit_wolf_cov

# quantcore-candidate: new code (HRP stage); parity oracle = PyPortfolioOpt +
# mlfinlab/Riskfolio HRP (dev-only).


@dataclass(frozen=True, slots=True)
class HRPResult:
    """Immutable result of an HRP allocation.

    Attributes
    ----------
    weights:
        Final portfolio weights on the simplex (sum to 1, all non-negative),
        labelled by asset in the original input order.
    ordered_assets:
        Asset labels in the quasi-diagonal (dendrogram leaf) order.
    link:
        The ``(N - 1) x 4`` SciPy linkage matrix used for clustering.
    quasidiag_order:
        The integer leaf order (a permutation of ``0 .. N-1``) returned by
        :func:`hrp.cluster.quasidiag.get_quasi_diag`.
    """

    weights: pd.Series
    ordered_assets: list[str]
    link: np.ndarray
    quasidiag_order: list[int]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result.

        ``weights`` is rendered as an ordered ``{asset: weight}`` mapping and the
        ``link`` matrix as a nested list, so the result crosses the API boundary
        without numpy/pandas types leaking through.
        """
        out = asdict(self)
        out["weights"] = {str(k): float(v) for k, v in self.weights.items()}
        out["link"] = np.asarray(self.link).tolist()
        return out


def get_cluster_var(cov: MatrixLike, cluster_items: list[int]) -> float:
    r"""Variance of a cluster under inverse-variance intra-cluster weights.

    Slices ``cov`` to the assets in ``cluster_items``, forms the
    inverse-variance portfolio over that sub-covariance
    (:math:`w_i \propto 1 / \sigma_i^2`, normalized to sum to one), and returns
    the resulting portfolio variance :math:`w^\top \Sigma_{\text{sub}}\, w`.

    HONESTY REQUIREMENT: the intra-cluster weights are **inverse-variance**, NOT
    equal weights. Using equal weights here is a common HRP bug and is rejected;
    a golden test pins the inverse-variance behaviour.

    Parameters
    ----------
    cov:
        The full ``N x N`` covariance matrix.
    cluster_items:
        Integer positions (into ``cov``) of the assets in this cluster.

    Returns
    -------
    float
        The cluster's portfolio variance under inverse-variance weighting.

    Raises
    ------
    ValidationError
        If ``cluster_items`` is empty or indexes outside ``cov``.
    """
    cov_arr = np.asarray(
        cov.to_numpy() if isinstance(cov, pd.DataFrame) else cov, dtype="float64"
    )
    if cov_arr.ndim != 2 or cov_arr.shape[0] != cov_arr.shape[1]:
        raise ValidationError(
            f"get_cluster_var: cov must be square, got shape {cov_arr.shape}."
        )
    n = cov_arr.shape[0]

    items = list(cluster_items)
    if len(items) == 0:
        raise ValidationError("get_cluster_var: cluster_items must be non-empty.")
    if any((not (0 <= i < n)) for i in items):
        raise ValidationError(
            f"get_cluster_var: cluster_items index outside cov of size {n}."
        )

    # Slice to the cluster's sub-covariance (positional indexing).
    sub = cov_arr[np.ix_(items, items)]

    # HONESTY REQUIREMENT: intra-cluster weights are INVERSE-VARIANCE, not equal.
    # quantcore-candidate: local inverse-variance portfolio (mirrors ivp_weights).
    diag = np.diag(sub).astype("float64")
    if np.any(diag <= 0.0) or not np.all(np.isfinite(diag)):
        raise ValidationError(
            "get_cluster_var: sub-covariance has a non-positive diagonal entry."
        )
    inv_var = 1.0 / diag
    weights = inv_var / inv_var.sum()

    cluster_var = float(weights @ sub @ weights)
    return cluster_var


def get_rec_bipart(cov: MatrixLike, sort_ix: list[int]) -> pd.Series:
    r"""Recursive bisection allocation over a quasi-diagonal asset order.

    Implements de Prado's ``getRecBipart``. Begins with unit weight on every
    asset, then repeatedly splits each contiguous cluster (in the
    ``sort_ix`` leaf order) into two halves and rescales the two halves by the
    factor

    .. math::

        \alpha = 1 - \frac{V_1}{V_1 + V_2},

    where :math:`V_1, V_2` are the two sub-clusters' variances from
    :func:`get_cluster_var`. The left half is multiplied by :math:`\alpha` and
    the right half by :math:`1 - \alpha`, so more capital flows to the
    lower-variance cluster. Recursion continues until every cluster is a single
    asset.

    Parameters
    ----------
    cov:
        The full ``N x N`` covariance matrix.
    sort_ix:
        The quasi-diagonal leaf order (a permutation of ``0 .. N-1``).

    Returns
    -------
    pandas.Series
        Weights labelled by the integer positions in ``sort_ix``, summing to one
        and all non-negative.

    Raises
    ------
    ValidationError
        If ``sort_ix`` is not a valid permutation of ``cov``'s indices.
    """
    cov_arr = np.asarray(
        cov.to_numpy() if isinstance(cov, pd.DataFrame) else cov, dtype="float64"
    )
    if cov_arr.ndim != 2 or cov_arr.shape[0] != cov_arr.shape[1]:
        raise ValidationError(
            f"get_rec_bipart: cov must be square, got shape {cov_arr.shape}."
        )
    n = cov_arr.shape[0]

    order = [int(i) for i in sort_ix]
    if sorted(order) != list(range(n)):
        raise ValidationError(
            "get_rec_bipart: sort_ix must be a permutation of cov's indices "
            f"(0 .. {n - 1})."
        )

    # de Prado's getRecBipart: start with unit weight everywhere, repeatedly
    # bisect each contiguous cluster (in leaf order) and rescale the two halves
    # inversely to their cluster variances.
    weights = pd.Series(1.0, index=order, dtype="float64")
    clusters: list[list[int]] = [order]

    while clusters:
        # Bisect every cluster with more than one item.
        clusters = [
            cluster[half_start:half_end]
            for cluster in clusters
            for half_start, half_end in (
                (0, len(cluster) // 2),
                (len(cluster) // 2, len(cluster)),
            )
            if len(cluster) > 1
        ]
        # Process pairs of (left, right) sub-clusters.
        for i in range(0, len(clusters), 2):
            left = clusters[i]
            right = clusters[i + 1]
            var_left = get_cluster_var(cov_arr, left)
            var_right = get_cluster_var(cov_arr, right)
            # alpha = 1 - V_left / (V_left + V_right); more capital to lower var.
            alpha = 1.0 - var_left / (var_left + var_right)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha

    return weights


def hrp_allocate(
    returns: ReturnsLike,
    *,
    cov: MatrixLike | None = None,
    linkage_method: str = "single",
) -> HRPResult:
    r"""Full Hierarchical Risk Parity allocation.

    Pipeline: estimate (or accept) a covariance matrix, convert it to a
    correlation matrix, build the two-step correlation distance
    (:func:`hrp.cluster.distance.correl_dist` then
    :func:`hrp.cluster.distance.euclidean_codistance`), cluster with the chosen
    linkage (:func:`hrp.cluster.linkage.linkage_matrix`), recover the leaf order
    (:func:`hrp.cluster.quasidiag.get_quasi_diag`), and allocate by recursive
    bisection (:func:`get_rec_bipart`).

    ROBUSTNESS CLAIM (encoded as a regression test): HRP never inverts the full
    covariance, so it must return valid simplex weights even on a
    block-perfectly-correlated (singular) covariance on which Markowitz CLA fails.

    Validated against PyPortfolioOpt and a second reference to ``1e-7`` on
    identical shrunk covariance + single linkage.

    Parameters
    ----------
    returns:
        A wide panel of asset returns (rows = time, columns = asset). Used to
        estimate the covariance when ``cov`` is not supplied, and to recover
        asset labels.
    cov:
        Optional pre-computed covariance matrix. When ``None``, the covariance is
        estimated from ``returns`` (the shared estimator is injected by the
        caller / orchestrator). Supplying ``cov`` is how the horse race feeds the
        identical covariance to every allocator.
    linkage_method:
        Linkage method for clustering. Defaults to ``"single"`` (paper default).

    Returns
    -------
    HRPResult
        The frozen result bundle (weights, leaf order, linkage matrix, order).

    Raises
    ------
    ValidationError
        If inputs are malformed.
    InsufficientDataError
        If there are too few observations to estimate the covariance.
    """
    # --- Resolve the covariance matrix and asset labels -------------------
    if cov is None:
        # The shared default estimator (Ledoit-Wolf) is used when the caller
        # does not inject a covariance; supplying ``cov`` is how the horse race
        # feeds the identical covariance to every allocator.
        cov_df = ledoit_wolf_cov(returns)
    elif isinstance(cov, pd.DataFrame):
        cov_df = cov.astype("float64").copy()
    else:
        cov_arr = np.asarray(cov, dtype="float64")
        if cov_arr.ndim != 2 or cov_arr.shape[0] != cov_arr.shape[1]:
            raise ValidationError(
                f"hrp_allocate: cov must be square, got shape {cov_arr.shape}."
            )
        # Recover labels from the returns columns when cov is unlabelled.
        if isinstance(returns, pd.DataFrame) and len(returns.columns) == cov_arr.shape[0]:
            labels = list(returns.columns)
        else:
            labels = list(range(cov_arr.shape[0]))
        cov_df = pd.DataFrame(cov_arr, index=labels, columns=labels)

    n_rows, n_cols = cov_df.shape
    if n_rows != n_cols:
        raise ValidationError(
            f"hrp_allocate: cov must be square, got shape {(n_rows, n_cols)}."
        )

    assets = list(cov_df.columns)

    # --- Stage 1: covariance -> correlation -> two-step distance ----------
    std = np.sqrt(np.diag(cov_df.to_numpy(dtype="float64")))
    if np.any(std <= 0.0) or not np.all(np.isfinite(std)):
        raise ValidationError(
            "hrp_allocate: covariance has a non-positive diagonal (variance) entry."
        )
    # quantcore-candidate: local cov -> corr (D^-1 Sigma D^-1, clipped to [-1, 1]).
    inv_std = 1.0 / std
    corr_arr = cov_df.to_numpy(dtype="float64") * np.outer(inv_std, inv_std)
    corr_arr = np.clip(corr_arr, -1.0, 1.0)
    np.fill_diagonal(corr_arr, 1.0)
    corr_df = pd.DataFrame(corr_arr, index=assets, columns=assets)

    dist = correl_dist(corr_df)
    codist = euclidean_codistance(dist)

    # --- Stage 1 (linkage) + Stage 2 (quasi-diagonalization) --------------
    link = linkage_matrix(codist, method=linkage_method)
    sort_ix = get_quasi_diag(link)

    # --- Stage 3: recursive bisection -------------------------------------
    # Weights are keyed by positional index into cov; map back to labels and
    # restore the original input asset order.
    pos_weights = get_rec_bipart(cov_df, sort_ix)
    label_weights = pd.Series(
        {assets[pos]: float(w) for pos, w in pos_weights.items()}, dtype="float64"
    )
    weights = label_weights.reindex(assets)

    ordered_assets = [assets[i] for i in sort_ix]

    return HRPResult(
        weights=weights,
        ordered_assets=[str(a) for a in ordered_assets],
        link=np.asarray(link),
        quasidiag_order=[int(i) for i in sort_ix],
        meta={"linkage_method": linkage_method, "n_assets": len(assets)},
    )
