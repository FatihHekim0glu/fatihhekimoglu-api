"""volforecast - GARCH vs ML for realized-volatility forecasting (honest null).

A pure, typed compute library that forecasts the h-day-ahead realized volatility
of an index and honestly tests whether XGBoost (or a research-only LSTM) beats a
well-specified GARCH(1,1) / HAR-RV out-of-sample. Evaluation is QLIKE (robust to
the noisy RV proxy) plus Diebold-Mariano and Hansen-SPA significance, so the
``best_model`` / ``ml_beats_garch`` verdict is a PURE function of the evidence.

Honest headline (Hansen & Lunde 2005): GARCH(1,1)/HAR-RV are HARD to beat - ML
wins only marginally on OOS QLIKE, if at all, and the LSTM rarely justifies its
cost. The default run is on a synthetic GARCH(1,1)-like series, so the null holds
by construction.

The package has ZERO import-time side effects and ZERO UI coupling: ``import
volforecast`` pulls in NO ``arch`` / ``xgboost`` / TensorFlow (all heavy fitters
are imported lazily inside their functions; the research-only LSTM is never
re-exported here). The same functions back the CLI and the hosted FastAPI tool
unchanged.

Public API is curated below; see :data:`__all__`.
"""

from __future__ import annotations

from volforecast._constants import (
    EPS,
    HAR_DAILY_WINDOW,
    HAR_MONTHLY_WINDOW,
    HAR_WEEKLY_WINDOW,
    PERIODS_PER_YEAR,
    RISKMETRICS_LAMBDA,
    SUPPORTED_HORIZONS,
    TRADING_DAYS,
)
from volforecast._exceptions import (
    ConvergenceError,
    InsufficientDataError,
    ValidationError,
    VolForecastError,
)
from volforecast._manifest import RunManifest, config_hash
from volforecast._rng import make_rng, spawn_substreams
from volforecast._validation import (
    align_inner,
    ensure_dataframe,
    ensure_series,
    validate_min_obs,
)
from volforecast.backtest.costs import FixedBpsCost
from volforecast.backtest.overlay import OverlayResult, vol_target_overlay
from volforecast.baselines import (
    HARRVModel,
    ewma_vol_forecast,
    fit_har_rv,
    random_walk_vol_forecast,
)
from volforecast.data import (
    DataSource,
    generate_garch_ohlc,
    get_ohlc,
    log_returns,
)
from volforecast.evaluation.dsr import (
    deflated_sharpe_ratio,
    effective_n_trials,
    expected_sharpe_variance,
    probabilistic_sharpe_ratio,
    variance_of_trial_sharpes,
)
from volforecast.evaluation.qlike import mse, qlike, qlike_loss_series
from volforecast.evaluation.tests import (
    DMResult,
    SPAResult,
    diebold_mariano,
    hansen_spa,
    newey_west_lrv,
)
from volforecast.evaluation.verdict import (
    BestModelClass,
    Verdict,
    derive_verdict,
)
from volforecast.features.har import HARFeatures, build_har_features, har_components
from volforecast.garch.models import (
    GARCHFit,
    fit_garch,
    forecast_garch_vol,
    garch_11_log_likelihood,
)
from volforecast.ml.xgb import XGBForecaster, fit_xgb
from volforecast.pipeline import (
    VolForecastRun,
    VolForecastSummary,
    build_vol_forecast_figures,
    run_vol_forecast,
)
from volforecast.plots import qlike_bar_figure, rv_forecast_figure
from volforecast.realized.estimators import (
    close_to_close_rv,
    forward_rv_target,
    garman_klass_rv,
    parkinson_rv,
    realized_volatility,
)
from volforecast.walkforward.engine import (
    WalkForwardConfig,
    WalkForwardResult,
    run_walk_forward,
)

__version__ = "0.1.0"

__all__ = [
    # version
    "__version__",
    # constants
    "EPS",
    "HAR_DAILY_WINDOW",
    "HAR_MONTHLY_WINDOW",
    "HAR_WEEKLY_WINDOW",
    "PERIODS_PER_YEAR",
    "RISKMETRICS_LAMBDA",
    "SUPPORTED_HORIZONS",
    "TRADING_DAYS",
    # exceptions
    "ConvergenceError",
    "InsufficientDataError",
    "ValidationError",
    "VolForecastError",
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
    "DataSource",
    "generate_garch_ohlc",
    "get_ohlc",
    "log_returns",
    # realized vol
    "close_to_close_rv",
    "forward_rv_target",
    "garman_klass_rv",
    "parkinson_rv",
    "realized_volatility",
    # features
    "HARFeatures",
    "build_har_features",
    "har_components",
    # baselines
    "HARRVModel",
    "ewma_vol_forecast",
    "fit_har_rv",
    "random_walk_vol_forecast",
    # garch
    "GARCHFit",
    "fit_garch",
    "forecast_garch_vol",
    "garch_11_log_likelihood",
    # ml (XGBoost only; LSTM is research-only and NOT re-exported)
    "XGBForecaster",
    "fit_xgb",
    # walk-forward
    "WalkForwardConfig",
    "WalkForwardResult",
    "run_walk_forward",
    # public horse-race entrypoint (serve path; NO LSTM/TF)
    "VolForecastRun",
    "VolForecastSummary",
    "build_vol_forecast_figures",
    "run_vol_forecast",
    # evaluation
    "BestModelClass",
    "DMResult",
    "SPAResult",
    "Verdict",
    "deflated_sharpe_ratio",
    "derive_verdict",
    "diebold_mariano",
    "effective_n_trials",
    "expected_sharpe_variance",
    "hansen_spa",
    "mse",
    "newey_west_lrv",
    "probabilistic_sharpe_ratio",
    "qlike",
    "qlike_loss_series",
    "variance_of_trial_sharpes",
    # backtest overlay
    "FixedBpsCost",
    "OverlayResult",
    "vol_target_overlay",
    # plots
    "qlike_bar_figure",
    "rv_forecast_figure",
]
