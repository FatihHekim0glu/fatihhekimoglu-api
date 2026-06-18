"""Leakage-guarded walk-forward backtest engine + cost model + statistics.

The same purged/embargoed walk-forward machinery as the HRP infra, reused to
backtest the tone-tertile long-short spread on the point-in-time panel. Importing
this package has no side effects.
"""

from __future__ import annotations

from edgar_nlp.backtest.costs import FixedBpsCost
from edgar_nlp.backtest.stats import (
    annualized_vol,
    max_drawdown,
    sharpe_ratio,
    turnover,
)
from edgar_nlp.backtest.walk_forward import BacktestResult, walk_forward_backtest

__all__ = [
    "BacktestResult",
    "FixedBpsCost",
    "annualized_vol",
    "max_drawdown",
    "sharpe_ratio",
    "turnover",
    "walk_forward_backtest",
]
