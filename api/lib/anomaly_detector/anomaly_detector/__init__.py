"""Market anomaly detector — a pure, typed compute library.

Flags anomalous trading days in a liquid ETF with two INDEPENDENT unsupervised
detectors — an Isolation Forest (Liu, Ting & Zhou 2008) and a PCA
reconstruction-error autoencoder (Sakurada & Yairi 2014, no torch) — under a
strictly causal walk-forward refit. The scaler, both models, and ALL thresholds
are fitted on the TRAIN slice only and then score the disjoint out-of-sample
slice (full-sample-leakage guard; ``.shift(1)`` flag chokepoint).

The headline is DESCRIPTIVE, not predictive: the detectors agree on a small core
of known macro-stress dates, but their day-level agreement is MODEST and their
precision against a transparent ``|z-return| > 3`` proxy label is LOW. There is
no ground-truth anomaly label, so NO alpha or tradability is claimed.

The package has ZERO import-time side effects and ZERO UI coupling: the same
functions back a local CLI and a hosted FastAPI tool unchanged.

Public API is curated below; see :data:`__all__`.
"""

from __future__ import annotations

from anomaly_detector._constants import EPS, PERIODS_PER_YEAR, TRADING_DAYS
from anomaly_detector._exceptions import (
    AnomalyDetectorError,
    InsufficientDataError,
    NotFittedError,
    ValidationError,
)
from anomaly_detector._manifest import RunManifest, config_hash
from anomaly_detector._rng import make_rng, spawn_substreams
from anomaly_detector._validation import (
    align_inner,
    ensure_dataframe,
    ensure_series,
    validate_min_obs,
)
from anomaly_detector.backtest.walk_forward import BacktestResult, walk_forward_backtest
from anomaly_detector.data import (
    InjectedSeries,
    compute_returns,
    generate_injected_series,
    load_prices,
)
from anomaly_detector.detectors import (
    AnomalyResult,
    IsolationForestDetector,
    PCAAutoencoderDetector,
)
from anomaly_detector.evaluation.agreement import (
    AgreementResult,
    compute_agreement,
    jaccard_index,
    proxy_precision_recall,
    regime_alignment,
)
from anomaly_detector.evaluation.dsr import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from anomaly_detector.evaluation.overlay import OverlayResult, fade_the_anomaly_overlay
from anomaly_detector.features.engineer import (
    FEATURE_NAMES,
    causal_return_zscore,
    engineer_features,
)
from anomaly_detector.plots import price_anomaly_figure, score_threshold_figure
from anomaly_detector.scan import ScanResult, run_anomaly_scan

__version__ = "0.1.0"

__all__ = [
    "EPS",
    "FEATURE_NAMES",
    "PERIODS_PER_YEAR",
    "TRADING_DAYS",
    "AgreementResult",
    "AnomalyDetectorError",
    "AnomalyResult",
    "BacktestResult",
    "InjectedSeries",
    "InsufficientDataError",
    "IsolationForestDetector",
    "NotFittedError",
    "OverlayResult",
    "PCAAutoencoderDetector",
    "RunManifest",
    "ScanResult",
    "ValidationError",
    "__version__",
    "align_inner",
    "causal_return_zscore",
    "compute_agreement",
    "compute_returns",
    "config_hash",
    "deflated_sharpe_ratio",
    "engineer_features",
    "ensure_dataframe",
    "ensure_series",
    "fade_the_anomaly_overlay",
    "generate_injected_series",
    "jaccard_index",
    "load_prices",
    "make_rng",
    "price_anomaly_figure",
    "probabilistic_sharpe_ratio",
    "proxy_precision_recall",
    "regime_alignment",
    "run_anomaly_scan",
    "score_threshold_figure",
    "spawn_substreams",
    "validate_min_obs",
    "walk_forward_backtest",
]
