"""Event-study core: abnormal returns / CAR, significance tests, placebo null.

Importing this subpackage has no side effects.
"""

from __future__ import annotations

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

__all__ = [
    "CARResult",
    "CARTestResult",
    "MarketModel",
    "PlaceboResult",
    "abnormal_returns",
    "bmp_statistic",
    "cross_sectional_t",
    "cumulative_abnormal_returns",
    "fit_market_model",
    "hac_car_test",
    "placebo_distribution",
    "run_car_tests",
    "sample_placebo_dates",
    "stack_event_cars",
]
