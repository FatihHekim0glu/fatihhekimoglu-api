"""gnn-stocks — a leakage-free graph-neural-network cross-section benchmark (honest NULL).

Builds a point-in-time stock-relationship graph from rolling return correlations
and pits a dense-adjacency Graph Convolutional Network and a mean-aggregator
GraphSAGE (plain torch, NO torch-geometric) against a per-node cross-sectional
ridge and a cross-sectional-momentum baseline for next-period return RANKING. The
comparison is leakage-free by construction — a point-in-time index-membership
universe (no static current-membership node set), a per-fold graph + feature scaler
fit on the TRAIN window only, and a purged (>= feature lookback + horizon - 1),
embargoed walk-forward — and judged honestly in RANKING space with rank-IC
(Spearman), a long-short decile spread net of costs, Diebold-Mariano, and a
Deflated-Sharpe correction. NO baseline-free in-sample R²/accuracy.

The documented, literature-consistent headline: a GCN / GraphSAGE over a
point-in-time stock-relationship graph RECOVERS sector/cluster structure
(DESCRIPTIVE) but does NOT reliably beat the ridge / momentum baselines on
out-of-sample rank-IC or long-short decile spread after costs (Diebold-Mariano
insignificant, Deflated Sharpe ~ 0). The deliverable is the rigorous, PIT,
leakage-free comparison, not a profit claim. The PURE ``gnn_beats_baseline``
verdict is ``False`` unless a GNN beats the best baseline with a DM-significant
margin AND a positive DSR net of costs.

IMPORT PURITY: this package has ZERO import-time side effects and imports NO heavy
dependency at module load. torch (``models.gcn`` / ``models.graphsage`` /
``train``), onnxruntime (``models.onnx_runtime`` / ``serve``), sklearn
(``models.ridge``), and plotly (``plots``) are imported LAZILY inside their
functions, so ``import gnnstocks`` never imports torch, onnxruntime, sklearn, or an
inference engine. The same functions back the Typer CLI and the hosted FastAPI tool.

Public API is curated below; see :data:`__all__`.
"""

from __future__ import annotations

from gnnstocks._constants import EPS, PERIODS_PER_YEAR, TRADING_DAYS
from gnnstocks._exceptions import (
    ArtifactError,
    GnnStocksError,
    InsufficientDataError,
    ValidationError,
)
from gnnstocks._manifest import RunManifest, config_hash
from gnnstocks._rng import make_rng, spawn_substreams
from gnnstocks._validation import (
    align_inner,
    ensure_dataframe,
    ensure_series,
    validate_min_obs,
)
from gnnstocks.costs import FixedBpsCost
from gnnstocks.data.loaders import (
    DataSource,
    forward_returns,
    load_pit_panel,
    node_features,
    pit_universe_as_of,
    synthetic_universe_map,
)
from gnnstocks.data.synthetic import (
    DEFAULT_N_BLOCKS,
    DEFAULT_N_NODES,
    BlockFactorPanel,
    block_factor_panel,
    pure_noise_panel,
)
from gnnstocks.evaluation.diebold_mariano import diebold_mariano, dm_favours_model
from gnnstocks.evaluation.dsr import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from gnnstocks.evaluation.metrics import (
    CrossSectionalMetrics,
    andrews_lag,
    cross_sectional_metrics,
    hac_standard_error,
    long_short_spread,
    rank_ic,
)
from gnnstocks.evaluation.verdict import Verdict, VerdictResult, derive_verdict
from gnnstocks.graph.build import corr_knn_adjacency, empty_adjacency, sector_adjacency
from gnnstocks.graph.normalize import normalize_adjacency
from gnnstocks.models.gcn import GcnConfig
from gnnstocks.models.graphsage import GraphSageConfig
from gnnstocks.models.naive import NaiveScores, cross_sectional_momentum, sector_mean_score
from gnnstocks.models.onnx_runtime import OnnxGraphModel, default_artifact_path
from gnnstocks.models.ridge import RidgeConfig, RidgeScores
from gnnstocks.serve import (
    GnnStocksRun,
    GnnStocksSummary,
    run_analysis,
    run_compare,
    score_from_onnx,
)
from gnnstocks.train import TrainResult, train_pipeline
from gnnstocks.walk_forward import Fold, WalkForwardResult, make_folds, required_purge

__version__ = "0.1.0"

__all__ = [  # noqa: RUF022 - grouped by domain for readability, not alphabetized
    # version
    "__version__",
    # constants
    "EPS",
    "PERIODS_PER_YEAR",
    "TRADING_DAYS",
    # exceptions
    "ArtifactError",
    "GnnStocksError",
    "InsufficientDataError",
    "ValidationError",
    # reproducibility
    "RunManifest",
    "config_hash",
    "make_rng",
    "spawn_substreams",
    # validation
    "align_inner",
    "ensure_dataframe",
    "ensure_series",
    "validate_min_obs",
    # data
    "BlockFactorPanel",
    "DEFAULT_N_BLOCKS",
    "DEFAULT_N_NODES",
    "DataSource",
    "block_factor_panel",
    "forward_returns",
    "load_pit_panel",
    "node_features",
    "pit_universe_as_of",
    "pure_noise_panel",
    "synthetic_universe_map",
    # graph
    "corr_knn_adjacency",
    "empty_adjacency",
    "normalize_adjacency",
    "sector_adjacency",
    # walk-forward + costs
    "FixedBpsCost",
    "Fold",
    "WalkForwardResult",
    "make_folds",
    "required_purge",
    # models (baselines + configs + lazy ONNX serve wrapper; torch stays lazy)
    "GcnConfig",
    "GraphSageConfig",
    "NaiveScores",
    "OnnxGraphModel",
    "RidgeConfig",
    "RidgeScores",
    "cross_sectional_momentum",
    "default_artifact_path",
    "sector_mean_score",
    # train + serve entrypoints (the backend calls run_analysis / score_from_onnx)
    "GnnStocksRun",
    "GnnStocksSummary",
    "TrainResult",
    "run_analysis",
    "run_compare",
    "score_from_onnx",
    "train_pipeline",
    # evaluation
    "CrossSectionalMetrics",
    "Verdict",
    "VerdictResult",
    "andrews_lag",
    "cross_sectional_metrics",
    "deflated_sharpe_ratio",
    "derive_verdict",
    "diebold_mariano",
    "dm_favours_model",
    "hac_standard_error",
    "long_short_spread",
    "probabilistic_sharpe_ratio",
    "rank_ic",
]
