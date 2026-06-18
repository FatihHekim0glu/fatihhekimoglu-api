"""mvts-forecast — a leakage-free multivariate-transformer forecast benchmark (honest NULL).

Benchmarks a PatchTST-style encoder and a simplified interpretable
variable-selection transformer against an LSTM, a per-series ARIMA, and a naive
random-walk baseline on a multivariate financial time series. The comparison is
leakage-free by construction — RevIN instance-norm from the input window only, a
standardizer fitted on the TRAIN fold only, and a purged (>= ``look_back``),
embargoed walk-forward — and judged honestly in RETURN space with Diebold-Mariano
and Deflated-Sharpe gates. NO price-level R².

The documented, literature-consistent headline: on noisy daily returns the deep
models do NOT reliably beat the naive baseline OOS on directional accuracy or
risk-adjusted PnL after costs (DM insignificant, DSR ~ 0). The deliverable is the
rigorous comparison, not a profit claim. The PURE ``deep_beats_naive`` verdict is
``False`` unless a deep model beats naive with a DM-significant margin AND a
positive DSR.

IMPORT PURITY: this package has ZERO import-time side effects and imports NO heavy
dependency at module load. torch (``models.lstm`` / ``models.patchtst`` /
``models.transformer_vs`` / ``train``), onnxruntime (``models.onnx_runtime``),
statsmodels (``models.arima``), and plotly (``plots``) are imported LAZILY inside
their functions, so ``import mvtsforecast`` never imports torch, onnxruntime, or an
inference engine. The same functions back the Typer CLI and the hosted FastAPI
tool.

Public API is curated below; see :data:`__all__`.
"""

from __future__ import annotations

from mvtsforecast._constants import EPS, PERIODS_PER_YEAR, TRADING_DAYS
from mvtsforecast._exceptions import (
    ArtifactError,
    InsufficientDataError,
    MvtsForecastError,
    ValidationError,
)
from mvtsforecast._manifest import RunManifest, config_hash
from mvtsforecast._rng import make_rng, spawn_substreams
from mvtsforecast._validation import (
    align_inner,
    ensure_dataframe,
    ensure_series,
    validate_min_obs,
)
from mvtsforecast.data.loaders import (
    DataSource,
    load_macro,
    load_prices,
    returns_from_prices,
)
from mvtsforecast.data.synthetic import (
    DEFAULT_BASKET,
    random_walk_panel,
    synthetic_panel,
    weak_factor_panel,
)
from mvtsforecast.evaluation.diebold_mariano import diebold_mariano, dm_favours_model
from mvtsforecast.evaluation.dsr import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from mvtsforecast.evaluation.metrics import (
    ForecastMetrics,
    andrews_lag,
    directional_accuracy,
    forecast_metrics,
    hac_standard_error,
    mae,
    mase_vs_naive,
    net_pnl_sharpe,
    rmse,
)
from mvtsforecast.evaluation.verdict import Verdict, VerdictResult, derive_verdict
from mvtsforecast.models.arima import ArimaResult
from mvtsforecast.models.lstm import LstmConfig
from mvtsforecast.models.naive import NaiveResult, naive_forecast, persistence_returns
from mvtsforecast.models.onnx_runtime import OnnxForecaster, default_artifact_path
from mvtsforecast.models.patchtst import PatchTSTConfig
from mvtsforecast.models.transformer_vs import TransformerVSConfig
from mvtsforecast.serve import (
    ForecastRun,
    ForecastSummary,
    forecast_from_onnx,
    run_compare,
    run_forecast,
)
from mvtsforecast.train import TrainResult, train_pipeline
from mvtsforecast.windowing.costs import FixedBpsCost
from mvtsforecast.windowing.revin import RevInStats, revin_denormalize, revin_normalize
from mvtsforecast.windowing.windows import (
    Fold,
    Standardizer,
    WindowSpec,
    assert_no_target_leakage,
    fit_standardizer,
    make_folds,
    make_windows,
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
    "MvtsForecastError",
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
    "DEFAULT_BASKET",
    "DataSource",
    "load_macro",
    "load_prices",
    "random_walk_panel",
    "returns_from_prices",
    "synthetic_panel",
    "weak_factor_panel",
    # windowing
    "FixedBpsCost",
    "Fold",
    "RevInStats",
    "Standardizer",
    "WindowSpec",
    "assert_no_target_leakage",
    "fit_standardizer",
    "make_folds",
    "make_windows",
    "revin_denormalize",
    "revin_normalize",
    # models (baselines + configs + lazy ONNX serve wrapper; torch stays lazy)
    "ArimaResult",
    "LstmConfig",
    "NaiveResult",
    "OnnxForecaster",
    "PatchTSTConfig",
    "TransformerVSConfig",
    "default_artifact_path",
    "naive_forecast",
    "persistence_returns",
    # train + serve entrypoints (the backend calls run_forecast / forecast_from_onnx)
    "ForecastRun",
    "ForecastSummary",
    "TrainResult",
    "forecast_from_onnx",
    "run_compare",
    "run_forecast",
    "train_pipeline",
    # evaluation
    "ForecastMetrics",
    "Verdict",
    "VerdictResult",
    "andrews_lag",
    "deflated_sharpe_ratio",
    "derive_verdict",
    "diebold_mariano",
    "directional_accuracy",
    "dm_favours_model",
    "forecast_metrics",
    "hac_standard_error",
    "mae",
    "mase_vs_naive",
    "net_pnl_sharpe",
    "probabilistic_sharpe_ratio",
    "rmse",
]
