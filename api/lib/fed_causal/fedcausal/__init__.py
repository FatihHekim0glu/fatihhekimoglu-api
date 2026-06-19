"""fed-causal — causal inference on Fed announcements (a torch-free econometrics tool).

A leakage-free event-study + difference-in-differences stack for FOMC
announcements: estimation-window-only market-model abnormal returns, cumulative
abnormal returns (CAR), the Boehmer-Musumeci-Poulsen standardized test,
HAC/clustered standard errors, placebo-date randomization (the PRIMARY
significance source), a rate-sensitivity DiD, a multiple-testing correction over
the full spec grid, and a PURE ``fed_effect_is_tradable`` verdict.

Honest headline: around FOMC announcements, CARs show a transient average move,
but after placebo-date randomization + HAC SEs + a multiple-testing correction
the effect is NOT a robust, exploitable cross-sectional alpha — it is honest
cross-sectional heterogeneity (rate-sensitive names move more, mechanically), not
tradable net of costs.

The package has ZERO import-time side effects and ZERO UI coupling: the same
functions back the CLI and the hosted FastAPI tool unchanged. No torch, no ONNX,
no sklearn — pure numpy/scipy/statsmodels.

Public API is curated below; see :data:`__all__`.
"""

from __future__ import annotations

from fedcausal._constants import (
    DEFAULT_ALPHA,
    DEFAULT_COST_BPS,
    DEFAULT_ESTIMATION_GAP,
    DEFAULT_ESTIMATION_WINDOW,
    EPS,
    MAX_EVENT_HALF_WIDTH,
    MAX_PLACEBO_DRAWS,
    PERIODS_PER_YEAR,
    TRADING_DAYS,
)
from fedcausal._exceptions import (
    EventCalendarError,
    FedCausalError,
    InsufficientDataError,
    ValidationError,
    WindowOverlapError,
)
from fedcausal._manifest import RunManifest, config_hash
from fedcausal._rng import make_rng, spawn_substreams
from fedcausal._validation import (
    align_inner,
    ensure_dataframe,
    ensure_series,
    validate_alpha,
    validate_min_obs,
)
from fedcausal.data.loaders import load_event_panel
from fedcausal.data.synthetic import (
    SyntheticPanel,
    pure_noise_panel,
    rate_sensitive_panel,
    synthetic_event_panel,
)
from fedcausal.did.heterogeneity import (
    HeterogeneitySpread,
    describe_heterogeneity,
    heterogeneity_spread,
)
from fedcausal.did.model import DiDResult, build_did_panel, estimate_did
from fedcausal.evaluation.hac import andrews_lag, newey_west_se
from fedcausal.evaluation.multiple_testing import (
    MultipleTestingResult,
    benjamini_hochberg,
    romano_wolf,
)
from fedcausal.evaluation.verdict import (
    FedVerdict,
    VerdictInputs,
    derive_verdict,
    fed_effect_is_tradable,
)
from fedcausal.events.calendar import (
    FOMCEvent,
    classify_surprise,
    load_fomc_calendar,
)
from fedcausal.events.windows import (
    EventWindows,
    assert_no_overlap,
    build_all_windows,
    build_windows,
)
from fedcausal.eventstudy.abnormal import (
    CARResult,
    MarketModel,
    abnormal_returns,
    cumulative_abnormal_returns,
    fit_market_model,
    stack_event_cars,
)
from fedcausal.eventstudy.placebo import (
    PlaceboResult,
    placebo_distribution,
    sample_placebo_dates,
)
from fedcausal.eventstudy.tests import (
    CARTestResult,
    bmp_statistic,
    cross_sectional_t,
    hac_car_test,
    run_car_tests,
)
from fedcausal.serve import AnalysisResult, run_analysis

__version__ = "0.1.0"

__all__ = [
    # constants
    "DEFAULT_ALPHA",
    "DEFAULT_COST_BPS",
    "DEFAULT_ESTIMATION_GAP",
    "DEFAULT_ESTIMATION_WINDOW",
    "EPS",
    "MAX_EVENT_HALF_WIDTH",
    "MAX_PLACEBO_DRAWS",
    "PERIODS_PER_YEAR",
    "TRADING_DAYS",
    # serve
    "AnalysisResult",
    # eventstudy
    "CARResult",
    "CARTestResult",
    # did
    "DiDResult",
    # exceptions
    "EventCalendarError",
    # events
    "EventWindows",
    "FOMCEvent",
    "FedCausalError",
    # evaluation
    "FedVerdict",
    "HeterogeneitySpread",
    "InsufficientDataError",
    "MarketModel",
    "MultipleTestingResult",
    "PlaceboResult",
    # reproducibility
    "RunManifest",
    # data
    "SyntheticPanel",
    "ValidationError",
    "VerdictInputs",
    "WindowOverlapError",
    # version
    "__version__",
    "abnormal_returns",
    # validation
    "align_inner",
    "andrews_lag",
    "assert_no_overlap",
    "benjamini_hochberg",
    "bmp_statistic",
    "build_all_windows",
    "build_did_panel",
    "build_windows",
    "classify_surprise",
    "config_hash",
    "cross_sectional_t",
    "cumulative_abnormal_returns",
    "derive_verdict",
    "describe_heterogeneity",
    "ensure_dataframe",
    "ensure_series",
    "estimate_did",
    "fed_effect_is_tradable",
    "fit_market_model",
    "hac_car_test",
    "heterogeneity_spread",
    "load_event_panel",
    "load_fomc_calendar",
    "make_rng",
    "newey_west_se",
    "placebo_distribution",
    "pure_noise_panel",
    "rate_sensitive_panel",
    "romano_wolf",
    "run_analysis",
    "run_car_tests",
    "sample_placebo_dates",
    "spawn_substreams",
    "stack_event_cars",
    "synthetic_event_panel",
    "validate_alpha",
    "validate_min_obs",
]
