"""lstm-forecast — a leakage-free LSTM next-day RETURN forecaster (honest NULL).

The explicit redemption of a leaky LSTM stock-price-prediction repo: predict the
next-day LOG-RETURN (never the price level), validate with a purged, embargoed,
per-fold-scaled walk-forward, and honestly test against a random-walk /
persistence baseline. The documented, literature-backed result is that the LSTM
does NOT beat naive persistence — reported as the deliverable, with no profit
claim and NO price-level R².

IMPORT PURITY: this package has ZERO import-time side effects and imports NO heavy
dependency at module load. TensorFlow (``models.lstm``, ``train`` export) and
onnxruntime (``models.onnx_runtime``) are imported LAZILY inside their functions,
so ``import lstmforecast`` never imports TF or an inference engine. The same
functions back a local demo, the Typer CLI, and the hosted FastAPI tool.

Public API is curated below; see :data:`__all__`.
"""

from __future__ import annotations

from lstmforecast._constants import EPS, PERIODS_PER_YEAR, TRADING_DAYS
from lstmforecast._exceptions import (
    ArtifactError,
    InsufficientDataError,
    LstmForecastError,
    ValidationError,
)
from lstmforecast._manifest import RunManifest, config_hash
from lstmforecast._rng import make_rng, spawn_substreams
from lstmforecast._validation import (
    ensure_dataframe,
    ensure_monotonic_index,
    ensure_series,
    validate_min_obs,
)
from lstmforecast.data import (
    load_prices,
    pure_noise_returns,
    random_walk_prices,
    to_log_returns,
    trend_plus_noise_prices,
)
from lstmforecast.evaluation.dsr import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from lstmforecast.evaluation.metrics import (
    ForecastMetrics,
    diebold_mariano,
    directional_accuracy,
    forecast_metrics,
    hac_standard_error,
    mae,
    mase_vs_persistence,
    rmse,
)
from lstmforecast.evaluation.verdict import Verdict, VerdictResult, derive_verdict
from lstmforecast.features.engineer import FeatureSpec, engineer_features
from lstmforecast.features.sequences import (
    create_sequences,
    fit_scaler,
    scale_sequences,
)
from lstmforecast.models.baselines import PersistenceForecaster, persistence_returns
from lstmforecast.models.lstm import LstmConfig
from lstmforecast.models.onnx_runtime import OnnxForecaster, default_artifact_path
from lstmforecast.serve import (
    ForecastRun,
    ForecastSummary,
    forecast_from_onnx,
    run_forecast,
)
from lstmforecast.train import TrainResult, train_pipeline
from lstmforecast.walkforward.costs import FixedBpsCost
from lstmforecast.walkforward.engine import (
    Fold,
    FoldResult,
    WalkForwardConfig,
    WalkForwardResult,
    make_folds,
    run_walk_forward,
)

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
    "InsufficientDataError",
    "LstmForecastError",
    "ValidationError",
    # reproducibility
    "RunManifest",
    "config_hash",
    "make_rng",
    "spawn_substreams",
    # validation
    "ensure_dataframe",
    "ensure_monotonic_index",
    "ensure_series",
    "validate_min_obs",
    # data
    "load_prices",
    "pure_noise_returns",
    "random_walk_prices",
    "to_log_returns",
    "trend_plus_noise_prices",
    # features
    "FeatureSpec",
    "create_sequences",
    "engineer_features",
    "fit_scaler",
    "scale_sequences",
    # models (baseline + config + lazy ONNX serve wrapper; TF stays lazy)
    "LstmConfig",
    "OnnxForecaster",
    "PersistenceForecaster",
    "default_artifact_path",
    "persistence_returns",
    # train + serve entrypoints (the backend calls run_forecast / forecast_from_onnx)
    "ForecastRun",
    "ForecastSummary",
    "TrainResult",
    "forecast_from_onnx",
    "run_forecast",
    "train_pipeline",
    # walk-forward
    "FixedBpsCost",
    "Fold",
    "FoldResult",
    "WalkForwardConfig",
    "WalkForwardResult",
    "make_folds",
    "run_walk_forward",
    # evaluation
    "ForecastMetrics",
    "Verdict",
    "VerdictResult",
    "deflated_sharpe_ratio",
    "derive_verdict",
    "diebold_mariano",
    "directional_accuracy",
    "forecast_metrics",
    "hac_standard_error",
    "mae",
    "mase_vs_persistence",
    "probabilistic_sharpe_ratio",
    "rmse",
]
