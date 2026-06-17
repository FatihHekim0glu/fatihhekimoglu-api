"""No-lookahead walk-forward engine, cost models, and performance statistics.

The reusable walk-forward driver, cost model, and scalar statistics are vendored
from hrp-portfolio; :mod:`wsb_sentiment.backtest.engine` adds the
sentiment-signal-specific anchored backtest (positions vs buy-and-hold and an
attention-only baseline, with a per-side bps cost grid).

Importing this subpackage has no side effects.
"""

from __future__ import annotations

from wsb_sentiment.backtest.costs import FixedBpsCost
from wsb_sentiment.backtest.engine import (
    SignalBacktestResult,
    attention_only_positions,
    run_signal_backtest,
)
from wsb_sentiment.backtest.runner import (
    SentimentBacktestRun,
    build_sentiment_figures,
    run_sentiment_backtest,
)
from wsb_sentiment.backtest.stats import (
    annualized_vol,
    max_drawdown,
    sharpe_ratio,
    turnover,
)
from wsb_sentiment.backtest.walk_forward import (
    BacktestResult,
    walk_forward_backtest,
)

__all__ = [
    "BacktestResult",
    "FixedBpsCost",
    "SentimentBacktestRun",
    "SignalBacktestResult",
    "annualized_vol",
    "attention_only_positions",
    "build_sentiment_figures",
    "max_drawdown",
    "run_sentiment_backtest",
    "run_signal_backtest",
    "sharpe_ratio",
    "turnover",
    "walk_forward_backtest",
]
