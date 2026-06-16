"""stock-clusters — correlation-clustering of equity universes (pure, typed).

Maps the diversification skeleton of an equity universe by clustering its
RMT-denoised correlation matrix under the Mantegna metric, and honestly tests
whether cluster-aware allocation beats naive 1/N out-of-sample after costs (it
usually does not). The clustering science is layered on top of reused, tested
machinery (RMT denoising, walk-forward purge/embargo, DSR, Jobson-Korkie-Memmel)
carried over from the HRP tool.

The package has ZERO import-time side effects and ZERO UI coupling: the same
functions back a local CLI/Streamlit demo and a hosted FastAPI tool unchanged.

Public API is curated below; see :data:`__all__`.
"""

from __future__ import annotations

# --- Reused core (verbatim-ported from HRP) --------------------------------- #
from stockclusters._constants import EPS, PERIODS_PER_YEAR, TRADING_DAYS
from stockclusters._exceptions import (
    HRPError,
    InsufficientDataError,
    SingularCovarianceError,
    ValidationError,
)
from stockclusters._manifest import RunManifest, config_hash
from stockclusters._rng import make_rng, spawn_substreams
from stockclusters._validation import (
    align_inner,
    ensure_dataframe,
    ensure_series,
    validate_min_obs,
)

# --- Allocation: cluster-aware schemes + diversification horse race --------- #
from stockclusters.allocation.schemes import (
    DiversificationResult,
    cluster_equal_weight,
    one_over_n_weights,
    stripped_hrp_weights,
)

# --- Reused backtest + estimators ------------------------------------------- #
from stockclusters.backtest.costs import FixedBpsCost
from stockclusters.backtest.stats import (
    annualized_vol,
    max_drawdown,
    sharpe_ratio,
    turnover,
)
from stockclusters.backtest.walk_forward import BacktestResult, walk_forward_backtest

# --- Clustering: hierarchical, embedding, kmeans, k-selection --------------- #
from stockclusters.clustering.embedding import rmt_signal_embedding
from stockclusters.clustering.hierarchical import (
    ClusterResult,
    cut_tree,
    hierarchical_clusters,
)
from stockclusters.clustering.kmeans import kmeans_clusters
from stockclusters.clustering.selection import GapResult, phase_randomize, select_k_gap

# --- Correlation: estimation, RMT, Mantegna distance ------------------------ #
from stockclusters.correlation.distance import (
    mantegna_distance,
    minimum_spanning_tree,
    subdominant_ultrametric,
)
from stockclusters.correlation.estimate import correlation_matrix, log_returns
from stockclusters.correlation.rmt import marchenko_pastur_clip

# --- Reused data layer ------------------------------------------------------ #
from stockclusters.data import compute_returns, get_prices, get_risk_free
from stockclusters.estimators.covariance import ledoit_wolf_cov, oas_cov, sample_cov

# --- Reused evaluation + clustering verdict --------------------------------- #
from stockclusters.evaluation.comparison import (
    ComparisonResult,
    block_bootstrap_sharpe_gap,
    jobson_korkie_memmel,
)
from stockclusters.evaluation.dsr import deflated_sharpe_ratio, probabilistic_sharpe_ratio
from stockclusters.evaluation.verdict import ClusteringVerdict, derive_clustering_verdict

# --- Metrics + plots -------------------------------------------------------- #
from stockclusters.metrics import (
    ari_vs_gics,
    cophenetic_correlation,
    modularity,
    silhouette_score,
)
from stockclusters.pipeline import (
    ClusterAnalysis,
    ClusterAnalysisParams,
    assemble_figures,
    run_cluster_analysis,
)
from stockclusters.plots import (
    cluster_heatmap_figure,
    dendrogram_figure,
    embedding_scatter_figure,
    mst_network_figure,
    oos_equity_figure,
    stability_figure,
)

# --- Stability: rolling re-fit, ARI, alignment ------------------------------ #
from stockclusters.stability.align import align_labels, births_and_deaths
from stockclusters.stability.ari import adjacent_window_ari, adjusted_rand_index
from stockclusters.stability.resample import StabilityResult, rolling_stability

__version__ = "0.1.0"

__all__ = [
    # constants
    "EPS",
    "PERIODS_PER_YEAR",
    "TRADING_DAYS",
    # backtest
    "BacktestResult",
    # pipeline
    "ClusterAnalysis",
    "ClusterAnalysisParams",
    # clustering
    "ClusterResult",
    # evaluation
    "ClusteringVerdict",
    "ComparisonResult",
    # allocation
    "DiversificationResult",
    "FixedBpsCost",
    "GapResult",
    # exceptions
    "HRPError",
    "InsufficientDataError",
    # reproducibility
    "RunManifest",
    "SingularCovarianceError",
    # stability
    "StabilityResult",
    "ValidationError",
    # version
    "__version__",
    "adjacent_window_ari",
    "adjusted_rand_index",
    # validation
    "align_inner",
    "align_labels",
    "annualized_vol",
    # metrics
    "ari_vs_gics",
    # pipeline
    "assemble_figures",
    "births_and_deaths",
    "block_bootstrap_sharpe_gap",
    "cluster_equal_weight",
    # plots
    "cluster_heatmap_figure",
    # data
    "compute_returns",
    "config_hash",
    "cophenetic_correlation",
    # correlation
    "correlation_matrix",
    "cut_tree",
    "deflated_sharpe_ratio",
    "dendrogram_figure",
    "derive_clustering_verdict",
    "embedding_scatter_figure",
    "ensure_dataframe",
    "ensure_series",
    "get_prices",
    "get_risk_free",
    "hierarchical_clusters",
    "jobson_korkie_memmel",
    "kmeans_clusters",
    # estimators
    "ledoit_wolf_cov",
    "log_returns",
    "make_rng",
    "mantegna_distance",
    "marchenko_pastur_clip",
    "max_drawdown",
    "minimum_spanning_tree",
    "modularity",
    "mst_network_figure",
    "oas_cov",
    "one_over_n_weights",
    "oos_equity_figure",
    "phase_randomize",
    "probabilistic_sharpe_ratio",
    "rmt_signal_embedding",
    "rolling_stability",
    "run_cluster_analysis",
    "sample_cov",
    "select_k_gap",
    "sharpe_ratio",
    "silhouette_score",
    "spawn_substreams",
    "stability_figure",
    "stripped_hrp_weights",
    "subdominant_ultrametric",
    "turnover",
    "validate_min_obs",
    "walk_forward_backtest",
]
