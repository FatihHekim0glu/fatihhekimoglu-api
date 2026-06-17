"""No-lookahead walk-forward engine, cost models, and performance statistics.

Importing this subpackage has no side effects.
"""

from __future__ import annotations

from anomaly_detector.backtest.costs import FixedBpsCost
from anomaly_detector.backtest.stats import (
    annualized_vol,
    max_drawdown,
    sharpe_ratio,
    turnover,
)
from anomaly_detector.backtest.walk_forward import BacktestResult, walk_forward_backtest

__all__ = [
    "BacktestResult",
    "FixedBpsCost",
    "annualized_vol",
    "max_drawdown",
    "sharpe_ratio",
    "turnover",
    "walk_forward_backtest",
]
