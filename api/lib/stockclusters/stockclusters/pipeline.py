"""End-to-end clustering pipeline — the single high-level entrypoint.

This module wires the three module groups (``correlation`` -> ``clustering`` ->
``stability`` / ``allocation`` / ``evaluation``) into one coherent call so the
hosted FastAPI router (and the CLI/Streamlit demo) can stay a thin wrapper:

    result = run_cluster_analysis(returns, params)
    figures = assemble_figures(result)

:func:`run_cluster_analysis` returns a frozen :class:`ClusterAnalysis` bundle that
always carries the :class:`~stockclusters.clustering.hierarchical.ClusterResult`,
the correlation / Mantegna-distance / MST artefacts, the gap-selection record, and
post-hoc metrics; it OPTIONALLY carries the
:class:`~stockclusters.allocation.schemes.DiversificationResult` (the honest 1/N
horse race) and the :class:`~stockclusters.stability.resample.StabilityResult`
(rolling-window ARI) when those analyses are requested.

The headline verdict is a PURE function of the diversification inference outputs
(:func:`~stockclusters.evaluation.verdict.derive_clustering_verdict`); it cannot
read "clusters beat 1/N" while the Memmel-JK test is insignificant or the deflated
Sharpe is non-positive.

HONESTY / LEAKAGE DISCIPLINE — what is fit on what
    - The DISPLAY cluster map (the ``ClusterResult`` carried on the bundle) is fit
      on the FULL supplied panel: correlation -> (RMT denoise) -> Mantegna distance
      -> gap-``k`` -> labels. This full-panel map is DESCRIPTIVE — it drives the
      heatmap/dendrogram/MST/embedding figures and the post-hoc ARI-vs-GICS
      diagnostic ONLY. It is NOT a backtest and is NEVER applied out-of-sample.
    - The diversification horse race is a TRUE walk-forward backtest with NO
      look-ahead: it does NOT reuse the full-panel labels. Inside each walk-forward
      TRAIN window it RE-FITS clusters (correlation -> RMT -> distance -> cut, on
      that window's data alone) and applies those train-only labels to the next OOS
      window with purge + embargo + ``shift(1)``. Post-cutoff returns therefore
      cannot shape the in-sample clusters of any OOS window. This reuses the exact
      per-window pattern of
      :func:`~stockclusters.stability.resample._default_clusterer`.
    - The DSR ``n_trials`` is the FULL product of every swept axis
      (``#clustering-families x #k-candidates x #weighting-schemes x
      #denoise-settings x #cost-grid-points``). The clustering-family axis is 2 when
      ``method="both"`` (both hierarchical and kmeans are fit and the best display
      map kept). Under-counting manufactures false significance, so the count is
      assembled explicitly in :func:`_dsr_trial_count` and guarded by the
      regression suite.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from stockclusters._typing import ReturnsLike
    from stockclusters.allocation.schemes import DiversificationResult
    from stockclusters.clustering.hierarchical import ClusterResult
    from stockclusters.clustering.selection import GapResult
    from stockclusters.evaluation.verdict import ClusteringVerdict
    from stockclusters.stability.resample import StabilityResult

__all__ = [
    "ClusterAnalysis",
    "ClusterAnalysisParams",
    "assemble_figures",
    "run_cluster_analysis",
]

#: Weighting schemes swept inside the diversification horse race
#: (1/N is the baseline; cluster-EW and stripped-HRP are the cluster-aware arms).
_N_WEIGHTING_SCHEMES = 2

#: Denoise settings *compared on OOS*. The pipeline runs a single denoise setting,
#: so this axis is 1 unless a caller explicitly sweeps it.
_N_DENOISE_SETTINGS = 1


@dataclass(frozen=True, slots=True)
class ClusterAnalysisParams:
    """Immutable parameter bundle for :func:`run_cluster_analysis`.

    Mirrors the hosted ``POST /tools/stock-clusters/run`` request so the router can
    construct this directly from the validated Pydantic model.
    """

    method: str = "both"
    linkage: str = "average"
    n_clusters: int | None = None
    k_min: int = 2
    k_max: int = 20
    denoise: bool = True
    distance: str = "mantegna"
    run_diversification: bool = False
    run_stability: bool = False
    cost_bps: float = 5.0
    embargo_days: int = 5
    train_window: int = 504
    gap_b: int = 20
    rebalance: str = "monthly"
    seed: int = 0


@dataclass(frozen=True, slots=True)
class ClusterAnalysis:
    """Immutable end-to-end result of :func:`run_cluster_analysis`.

    Attributes
    ----------
    cluster_result:
        The frozen clustering bundle (labels, dendrogram order, linkage,
        silhouette).
    correlation:
        The (optionally RMT-denoised) correlation matrix the clustering was fit on.
    distance:
        The Mantegna distance matrix.
    mst_edges:
        The minimum-spanning-tree edge list of ``distance``.
    embedding:
        The RMT-signal eigenvector embedding (``None`` when it could not be built).
    gap_result:
        The gap-statistic selection record when ``k`` was chosen automatically;
        ``None`` for a fixed ``k``.
    n_clusters:
        The number of clusters in the DISPLAY labeling.
    selection_method:
        ``"fixed"`` or the gap selection rule (e.g. ``"tibshirani_1se"``). When
        ``method="both"`` a ``"+both(won=...)"`` suffix records that both clustering
        families ran and which one supplied the canonical display map.
    silhouette, modularity:
        Post-hoc cluster-quality scalars.
    ari_vs_gics:
        Post-hoc ARI between clusters and GICS sectors (``None`` when no GICS map
        was supplied). GICS NEVER enters the distance or ``k``-selection.
    diversification:
        The honest 1/N-vs-cluster horse race (clusters RE-FIT per walk-forward
        train window — never the full-panel display labels), or ``None`` when not
        requested.
    stability:
        The rolling-window stability record, or ``None`` when not requested.
    verdict:
        The headline verdict (pure function of the diversification inference), or
        ``None`` when no diversification ran.
    n_trials:
        The FULL DSR trial count used to deflate the diversification Sharpe.
    data_source:
        Provenance tag passed through from the caller (e.g. ``"polygon"``).
    """

    cluster_result: ClusterResult
    correlation: pd.DataFrame
    distance: pd.DataFrame
    mst_edges: pd.DataFrame
    embedding: pd.DataFrame | None
    gap_result: GapResult | None
    n_clusters: int
    selection_method: str
    silhouette: float
    modularity: float
    ari_vs_gics: float | None
    diversification: DiversificationResult | None = None
    stability: StabilityResult | None = None
    verdict: ClusteringVerdict | None = None
    n_trials: int = 1
    data_source: str = "synthetic"
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable summary ``dict`` of this analysis.

        Heavy matrices (correlation/distance/MST/embedding) are intentionally
        omitted here — they belong in :func:`assemble_figures`. This is the scalar
        + cluster-membership summary the API ``summary`` block is built from.
        """
        clusters: dict[str, list[str]] = {}
        labels = self.cluster_result.labels
        for cid in sorted({int(v) for v in labels.to_numpy()}):
            members = [str(a) for a in labels.index[labels == cid]]
            clusters[str(cid)] = members
        out: dict[str, Any] = {
            "n_assets": int(labels.shape[0]),
            "n_clusters": int(self.n_clusters),
            "selection_method": str(self.selection_method),
            "silhouette": _finite_or_none(self.silhouette),
            "modularity": _finite_or_none(self.modularity),
            "ari_vs_gics": _finite_or_none(self.ari_vs_gics),
            "clusters": clusters,
            "n_trials": int(self.n_trials),
            "data_source": str(self.data_source),
            "verdict": None if self.verdict is None else str(self.verdict.value),
        }
        if self.gap_result is not None:
            out["gap_k"] = int(self.gap_result.k_selected)
        if self.diversification is not None:
            out["diversification"] = self.diversification.to_dict()
        if self.stability is not None:
            out["stability_ari_mean"] = _finite_or_none(self.stability.ari_mean)
        return out


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #


def _finite_or_none(value: float | None) -> float | None:
    """Map ``None`` / NaN / Inf to ``None``; otherwise return ``float(value)``."""
    import math

    if value is None:
        return None
    f = float(value)
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _renormalize_to_correlation(denoised: pd.DataFrame) -> pd.DataFrame:
    """Rescale an RMT-denoised covariance back to a unit-diagonal correlation."""
    import numpy as np
    import pandas as pd

    arr = denoised.to_numpy(dtype="float64")
    std = np.sqrt(np.diag(arr))
    safe = np.where(std > 0.0, std, 1.0)
    corr_arr = arr / np.outer(safe, safe)
    np.fill_diagonal(corr_arr, 1.0)
    corr_arr = np.clip(corr_arr, -1.0, 1.0)
    return pd.DataFrame(corr_arr, index=denoised.index, columns=denoised.columns)


def _dsr_trial_count(
    *,
    n_linkages: int,
    n_k_candidates: int,
    n_weighting_schemes: int = _N_WEIGHTING_SCHEMES,
    n_denoise_settings: int = _N_DENOISE_SETTINGS,
    n_cost_points: int = 1,
) -> int:
    r"""The FULL DSR trial count = product of every swept axis.

    ``n_trials = #clustering-families x #k-candidates x #weighting-schemes x
    #denoise-settings x #cost-grid-points``. The first factor (``n_linkages``)
    counts the clustering FAMILIES actually fit and compared (2 when
    ``method="both"`` runs hierarchical AND kmeans, else 1). Each factor is floored
    at ``1`` so the product is always ``>= 1``; the regression suite asserts the
    returned count is never *less* than the product of the axes the pipeline
    actually swept (under-counting manufactures false significance).
    """
    factors = (
        max(1, int(n_linkages)),
        max(1, int(n_k_candidates)),
        max(1, int(n_weighting_schemes)),
        max(1, int(n_denoise_settings)),
        max(1, int(n_cost_points)),
    )
    product = 1
    for f in factors:
        product *= f
    return product


def _cluster_universe(
    returns: pd.DataFrame,
    params: ClusterAnalysisParams,
) -> tuple[ClusterResult, pd.DataFrame, pd.DataFrame, GapResult | None, str]:
    """Fit correlation -> (denoise) -> Mantegna distance -> select ``k`` -> cluster.

    Returns ``(cluster_result, correlation, distance, gap_result, selection_method)``.

    All fitting here is on the supplied (FULL) panel; this DISPLAY map drives the
    descriptive figures (heatmap/dendrogram/MST/embedding) and the post-hoc
    ARI-vs-GICS diagnostic ONLY. The diversification horse race never reuses these
    labels OOS — it re-fits clusters inside each walk-forward train window (see
    :func:`_run_diversification`).

    ``method`` selects the clustering family:

    - ``"hierarchical"`` — agglomerative linkage on the Mantegna distance.
    - ``"kmeans"`` — K-means on the RMT-signal embedding.
    - ``"both"`` (default) — run BOTH and keep the higher-silhouette map as the
      canonical DISPLAY labeling; ``selection_method`` records that both ran and
      which family won.
    """

    from stockclusters.clustering.embedding import rmt_signal_embedding
    from stockclusters.clustering.hierarchical import hierarchical_clusters
    from stockclusters.clustering.kmeans import kmeans_clusters
    from stockclusters.clustering.selection import select_k_gap
    from stockclusters.correlation.distance import mantegna_distance
    from stockclusters.correlation.estimate import correlation_matrix
    from stockclusters.correlation.rmt import marchenko_pastur_clip

    n_obs = int(returns.shape[0])
    n_assets = int(returns.shape[1])
    corr: pd.DataFrame = correlation_matrix(returns)
    if params.denoise:
        denoised = marchenko_pastur_clip(corr, n_obs=n_obs)
        corr = _renormalize_to_correlation(denoised)

    dist = mantegna_distance(corr)

    gap_result: GapResult | None = None
    if params.n_clusters is not None and int(params.n_clusters) > 0:
        k = int(params.n_clusters)
        selection_method = "fixed"
    else:
        k_max = min(int(params.k_max), n_assets - 1)
        k_min = max(2, min(int(params.k_min), k_max))
        gap_result = select_k_gap(
            returns,
            dist,
            k_min=k_min,
            k_max=k_max,
            method=params.linkage,
            n_references=int(params.gap_b),
            seed=int(params.seed),
        )
        k = int(gap_result.k_selected)
        selection_method = gap_result.selection_rule

    def _fit_hierarchical() -> ClusterResult:
        return hierarchical_clusters(dist, n_clusters=k, method=params.linkage)

    def _fit_kmeans() -> ClusterResult:
        embedding = rmt_signal_embedding(corr, n_obs=n_obs)
        return kmeans_clusters(embedding, n_clusters=k, seed=int(params.seed))

    def _sil(value: float) -> float:
        import math

        return value if math.isfinite(value) else float("-inf")

    if params.method == "kmeans":
        result = _fit_kmeans()
    elif params.method == "both":
        # Run BOTH families and keep the higher-silhouette map as the canonical
        # DISPLAY labeling. Both are real trials the DSR multiplicity must count.
        hier = _fit_hierarchical()
        kmn = _fit_kmeans()
        if _sil(kmn.silhouette) > _sil(hier.silhouette):
            result, won = kmn, "kmeans"
        else:
            result, won = hier, "hierarchical"
        selection_method = f"{selection_method}+both(won={won})"
    else:
        result = _fit_hierarchical()

    return result, corr, dist, gap_result, selection_method


# --------------------------------------------------------------------------- #
# Public entrypoint                                                            #
# --------------------------------------------------------------------------- #


def run_cluster_analysis(
    returns: ReturnsLike,
    params: ClusterAnalysisParams | None = None,
    *,
    gics: dict[str, str] | None = None,
    data_source: str = "synthetic",
) -> ClusterAnalysis:
    """Run the full clustering pipeline and return a frozen :class:`ClusterAnalysis`.

    Pipeline::

        # DISPLAY map (full-panel, descriptive — NOT a backtest):
        correlation -> (RMT denoise) -> Mantegna distance -> gap/fixed k
        -> cluster -> MST + embedding + post-hoc metrics
        -> [optional] rolling stability (adjacent-window ARI; per-window re-fit)
        # Backtest (TRAIN-ONLY cluster re-fit per walk-forward window):
        -> [optional] 1/N vs cluster-aware OOS horse race (Memmel-JK + DSR)
        -> [optional] pure-function headline verdict

    Parameters
    ----------
    returns:
        A wide panel of asset returns (rows = time, columns = asset). The DISPLAY
        cluster map is fit on this FULL panel for descriptive figures and the
        post-hoc ARI-vs-GICS diagnostic ONLY; the diversification horse race never
        reuses it OOS — it re-fits clusters inside each walk-forward train window.
    params:
        The :class:`ClusterAnalysisParams` bundle (defaults to library defaults).
    gics:
        Optional ``{ticker: gics_sector}`` map for the POST-HOC ARI-vs-GICS metric
        ONLY. GICS NEVER enters the distance or ``k``-selection.
    data_source:
        Provenance tag passed through to the result.

    Returns
    -------
    ClusterAnalysis
        The frozen end-to-end bundle.
    """

    from stockclusters._validation import ensure_dataframe
    from stockclusters.correlation.distance import minimum_spanning_tree
    from stockclusters.metrics import ari_vs_gics, modularity, silhouette_score

    if params is None:
        params = ClusterAnalysisParams()

    panel = ensure_dataframe(returns, name="returns")
    panel = panel.dropna(axis=1, how="all").dropna(axis=0, how="any")

    result, corr, dist, gap_result, selection_method = _cluster_universe(panel, params)

    mst_edges = minimum_spanning_tree(dist)
    try:
        from stockclusters.clustering.embedding import rmt_signal_embedding

        embedding: pd.DataFrame | None = rmt_signal_embedding(corr, n_obs=int(panel.shape[0]))
    except Exception:
        embedding = None

    silhouette = silhouette_score(dist, result.labels)
    modularity_score = modularity(result.labels, corr)
    ari_gics: float | None = None
    if gics:
        ari_gics = ari_vs_gics(result.labels, gics)

    # --- DSR trial count: the FULL product of every swept axis ----------------
    # Axes actually swept for the reported diversification result:
    #   * clustering families: 2 when method=="both" (hierarchical AND kmeans both
    #     run and the best DISPLAY map is selected by silhouette), else 1;
    #   * k-candidates: the gap selector evaluates every k in the candidate grid;
    #   * weighting schemes: the OOS race selects the best of cluster-EW /
    #     stripped-HRP (folded in by _dsr_trial_count's default);
    #   * denoise settings / cost-grid points: the pipeline reports a single
    #     setting of each, so both axes are 1 here.
    n_k = 1 if gap_result is None else len(gap_result.k_candidates)
    n_families = 2 if params.method == "both" else 1
    n_trials = _dsr_trial_count(n_linkages=n_families, n_k_candidates=n_k)

    diversification = None
    verdict = None
    stability = None

    if params.run_stability:
        stability = _run_stability(panel, result, params)

    if params.run_diversification:
        diversification, verdict = _run_diversification(panel, result, params, n_trials)

    return ClusterAnalysis(
        cluster_result=result,
        correlation=corr,
        distance=dist,
        mst_edges=mst_edges,
        embedding=embedding,
        gap_result=gap_result,
        n_clusters=int(result.n_clusters),
        selection_method=selection_method,
        silhouette=float(silhouette),
        modularity=float(modularity_score),
        ari_vs_gics=ari_gics,
        diversification=diversification,
        stability=stability,
        verdict=verdict,
        n_trials=int(n_trials),
        data_source=str(data_source),
        meta={
            "n_obs": int(panel.shape[0]),
            "n_assets": int(panel.shape[1]),
            "denoise": bool(params.denoise),
            "linkage": str(params.linkage),
            "method": str(params.method),
        },
    )


def _run_stability(
    returns: pd.DataFrame,
    result: ClusterResult,
    params: ClusterAnalysisParams,
) -> StabilityResult:
    """Run the rolling-window stability analysis at the frozen ``k``."""
    from stockclusters.stability.resample import rolling_stability

    return rolling_stability(
        returns,
        n_clusters=int(result.n_clusters),
        train_window=int(params.train_window),
        method=params.linkage,
        denoise=bool(params.denoise),
    )


def _run_diversification(
    returns: pd.DataFrame,
    result: ClusterResult,
    params: ClusterAnalysisParams,
    n_trials: int,
) -> tuple[DiversificationResult, ClusteringVerdict]:
    """Run the honest horse race (TRAIN-ONLY cluster re-fit) + derive the verdict.

    The OOS horse race does NOT reuse ``result.labels`` (which were fit on the FULL
    panel — fine for display, leaky as a backtest). Instead it builds a per-window
    clusterer that RE-FITS the correlation -> RMT -> distance -> cut pipeline inside
    each walk-forward train window, applying those train-only labels to the next OOS
    window with purge + embargo + ``shift(1)``. The clustering FAMILY (kmeans vs
    hierarchical) matches the canonical display map; ``k`` is pinned to
    ``result.n_clusters``.
    """
    from stockclusters.allocation.schemes import (
        _default_window_clusterer,
        run_diversification,
    )
    from stockclusters.evaluation.verdict import derive_clustering_verdict

    # The family that drives the OOS race matches the canonical display labeling:
    # for method=="both" the winner is recorded in result.method.
    family = "kmeans" if str(result.method).startswith("kmeans") else "hierarchical"
    window_clusterer = _default_window_clusterer(
        n_clusters=int(result.n_clusters),
        method=params.linkage,
        denoise=bool(params.denoise),
        family=family,
        seed=int(params.seed),
    )

    diversification = run_diversification(
        returns,
        result.labels,
        lookback_window=int(params.train_window),
        n_trials=int(n_trials),
        cost_bps=float(params.cost_bps),
        rebalance=params.rebalance,
        embargo=int(params.embargo_days),
        purge=int(params.embargo_days),
        clusterer=window_clusterer,
    )
    verdict = derive_clustering_verdict(
        diversification.memmel_jk_pvalue,
        diversification.deflated_sharpe,
        diversification.sharpe_diff_vs_1overN,
    )
    return diversification, verdict


# --------------------------------------------------------------------------- #
# Figure assembly                                                             #
# --------------------------------------------------------------------------- #


def assemble_figures(analysis: ClusterAnalysis) -> dict[str, dict[str, Any] | None]:
    """Assemble all Plotly figures for a :class:`ClusterAnalysis` (lazy plotly).

    Returns a mapping with keys ``heatmap_figure``, ``dendrogram_figure``,
    ``mst_figure``, ``embedding_figure``, ``equity_curve_figure`` and
    ``stability_figure``. Figures that cannot be produced (no diversification /
    stability run, or a missing artefact) are explicit ``None`` so the API layer
    serializes ``null`` rather than ``undefined``/``NaN``.
    """
    from stockclusters.plots import (
        cluster_heatmap_figure,
        dendrogram_figure,
        embedding_scatter_figure,
        mst_network_figure,
        oos_equity_figure,
        stability_figure,
    )

    labels = analysis.cluster_result.labels
    figures: dict[str, dict[str, Any] | None] = {
        "heatmap_figure": cluster_heatmap_figure(
            analysis.correlation, analysis.cluster_result.ordered_assets, labels
        ),
        "mst_figure": mst_network_figure(analysis.mst_edges, labels),
        "dendrogram_figure": None,
        "embedding_figure": None,
        "equity_curve_figure": None,
        "stability_figure": None,
    }

    if analysis.cluster_result.linkage is not None:
        figures["dendrogram_figure"] = dendrogram_figure(
            analysis.cluster_result.linkage, [str(a) for a in labels.index]
        )

    if analysis.embedding is not None:
        figures["embedding_figure"] = embedding_scatter_figure(analysis.embedding, labels)

    if analysis.diversification is not None:
        curves = analysis.diversification.meta.get("oos_curves")
        if isinstance(curves, dict) and curves:
            figures["equity_curve_figure"] = oos_equity_figure(curves)

    if analysis.stability is not None:
        figures["stability_figure"] = stability_figure(
            analysis.stability.ari_series, analysis.stability.window_dates
        )

    return figures
