"""LendingClub default classifier — a pure, typed compute library.

A leakage-free, calibrated XGBoost classifier that *ranks* a loan application's
probability of default at origination. It is benchmarked honestly out-of-time
(temporal vintage split, never random K-fold) against a base-rate predictor and
an L2-logistic baseline, with a DeLong test on the AUC gap.

HONEST-NULL DISCIPLINE: the headline is ROC-AUC / PR-AUC / Brier — NEVER accuracy
and NEVER profit/ROI. The model ranks risk; it does not predict which individuals
default. Two limits are stated up front: (1) accepted-loans-only selection bias;
(2) ``int_rate`` / ``grade`` are LendingClub's own risk model, so their importance
is partly circular. The shipped demo artifact is trained on a SYNTHETIC LC-schema
panel so the tool is reproducible without the proprietary Kaggle dump; expected
real-data ROC-AUC (~0.70) requires the Kaggle CSV via ``train --data``.

The package has ZERO import-time side effects and ZERO UI coupling: the same
functions back the CLI and a hosted FastAPI tool unchanged.

Public API is curated below; see :data:`__all__`.
"""

from __future__ import annotations

from lendingclub_default._constants import (
    DEFAULT_STATUSES,
    EPS,
    IN_PROGRESS_STATUSES,
    N_RISK_DECILES,
    PAID_STATUSES,
    VALID_GRADES,
    VALID_TERMS,
)
from lendingclub_default._exceptions import (
    ArtifactError,
    InsufficientDataError,
    LeakageError,
    LendingClubDefaultError,
    TemporalSplitError,
    ValidationError,
)
from lendingclub_default._manifest import RunManifest, config_hash
from lendingclub_default._rng import make_rng, spawn_substreams
from lendingclub_default._validation import (
    align_inner,
    ensure_dataframe,
    ensure_series,
    validate_min_obs,
)
from lendingclub_default.data.labels import LabelResult, build_labels
from lendingclub_default.data.leakage import (
    LEAKAGE_COLS,
    assert_no_leakage,
    drop_leakage,
)
from lendingclub_default.data.load import APPLICATION_COLUMNS, coerce_dtypes, load_panel
from lendingclub_default.data.split import (
    TemporalSplit,
    assert_temporal_order,
    temporal_split,
)
from lendingclub_default.data.synthetic import (
    SyntheticConfig,
    generate_synthetic_panel,
)
from lendingclub_default.evaluation.calibration import (
    ReliabilityCurve,
    reliability_curve,
)
from lendingclub_default.evaluation.delong import DeLongResult, delong_auc_test
from lendingclub_default.evaluation.metrics import (
    MetricBundle,
    brier_score,
    compute_metrics,
    ks_statistic,
    log_loss,
    pr_auc,
    roc_auc,
)
from lendingclub_default.evaluation.threshold import (
    CostMatrix,
    ThresholdSweep,
    threshold_sweep,
)
from lendingclub_default.features.pipeline import (
    FeatureSpec,
    build_feature_pipeline,
    out_of_fold_target_encode,
)
from lendingclub_default.models.baselines import BaseRatePredictor, fit_logistic
from lendingclub_default.models.calibrate import (
    CalibratedModel,
    CalibrationResult,
    fit_calibration,
)
from lendingclub_default.models.reason_codes import (
    ReasonCode,
    reason_codes_from_logit,
    shap_reason_codes,
)
from lendingclub_default.models.xgb import (
    XGBConfig,
    fit_xgb,
    save_booster,
)
from lendingclub_default.plots import (
    calibration_figure,
    figure_to_dict,
    pr_figure,
    roc_figure,
    score_distribution_figure,
)
from lendingclub_default.train import (
    ScoredArtifacts,
    TrainArtifacts,
    load_artifacts,
    load_booster,
    train,
)

__version__ = "0.1.0"

# Curated public API, kept isort-sorted (ruff RUF022) for a stable surface.
__all__ = [
    "APPLICATION_COLUMNS",
    "DEFAULT_STATUSES",
    "EPS",
    "IN_PROGRESS_STATUSES",
    "LEAKAGE_COLS",
    "N_RISK_DECILES",
    "PAID_STATUSES",
    "VALID_GRADES",
    "VALID_TERMS",
    "ArtifactError",
    "BaseRatePredictor",
    "CalibratedModel",
    "CalibrationResult",
    "CostMatrix",
    "DeLongResult",
    "FeatureSpec",
    "InsufficientDataError",
    "LabelResult",
    "LeakageError",
    "LendingClubDefaultError",
    "MetricBundle",
    "ReasonCode",
    "ReliabilityCurve",
    "RunManifest",
    "ScoredArtifacts",
    "SyntheticConfig",
    "TemporalSplit",
    "TemporalSplitError",
    "ThresholdSweep",
    "TrainArtifacts",
    "ValidationError",
    "XGBConfig",
    "__version__",
    "align_inner",
    "assert_no_leakage",
    "assert_temporal_order",
    "brier_score",
    "build_feature_pipeline",
    "build_labels",
    "calibration_figure",
    "coerce_dtypes",
    "compute_metrics",
    "config_hash",
    "delong_auc_test",
    "drop_leakage",
    "ensure_dataframe",
    "ensure_series",
    "figure_to_dict",
    "fit_calibration",
    "fit_logistic",
    "fit_xgb",
    "generate_synthetic_panel",
    "ks_statistic",
    "load_artifacts",
    "load_booster",
    "load_panel",
    "log_loss",
    "make_rng",
    "out_of_fold_target_encode",
    "pr_auc",
    "pr_figure",
    "reason_codes_from_logit",
    "reliability_curve",
    "roc_auc",
    "roc_figure",
    "save_booster",
    "score_distribution_figure",
    "shap_reason_codes",
    "spawn_substreams",
    "temporal_split",
    "threshold_sweep",
    "train",
    "validate_min_obs",
]
