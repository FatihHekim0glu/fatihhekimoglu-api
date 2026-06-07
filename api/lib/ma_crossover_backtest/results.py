"""Result dataclasses shared across the engine.

These are the contract types every module produces or consumes. They sit at the
bottom of the dependency graph: results.py imports nothing from this project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Output of a single backtest run.

    Attributes
    ----------
    equity : pd.Series
        Equity curve, indexed by date, starting at ``initial_cash``.
    positions : pd.Series
        Realised position held over each bar, in {0, 1} for long/flat.
        Already shifted by one bar — this is what was held *into* bar t.
    daily_returns : pd.Series
        Net daily returns of the strategy (gross minus cost on turnover).
    trades : pd.DataFrame
        Per-trade ledger with columns: entry_date, exit_date, entry_price,
        exit_price, bars_held, gross_return, net_return.
    gross_returns : pd.Series
        Daily returns before transaction costs. Useful for attribution.
    costs : pd.Series
        Daily cost drag (fraction of equity). Sums to total cost paid.
    """

    equity: pd.Series
    positions: pd.Series
    daily_returns: pd.Series
    trades: pd.DataFrame
    gross_returns: pd.Series
    costs: pd.Series


@dataclass(frozen=True, slots=True)
class MetricsTable:
    """Computed performance metrics for one return series."""

    total_return: float
    cagr: float
    annual_vol: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration_days: int
    n_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_holding_period_days: float
    annualised_turnover: float


@dataclass(frozen=True, slots=True)
class BenchmarkComparison:
    """Statistical comparison of strategy vs buy-and-hold."""

    alpha_annual: float
    alpha_t_stat: float
    alpha_p_value: float
    beta: float
    beta_se: float
    information_ratio: float
    tracking_error_annual: float
    active_return_annual: float
    sharpe_diff: float
    sharpe_diff_p_value: float
    n_observations: int
    hac_lags: int


@dataclass(frozen=True, slots=True)
class DeflatedSharpeResult:
    """Output of the Deflated Sharpe Ratio computation."""

    observed_sharpe: float
    expected_max_sharpe_under_null: float
    deflated_sharpe: float
    probabilistic_sharpe: float
    n_trials: int
    n_effective_trials: int
    skew: float
    kurtosis: float
    can_reject_null: bool


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """One out-of-sample fold from the walk-forward run."""

    fold_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    selected_fast: int
    selected_slow: int
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    out_of_sample_return: float
    test_equity: pd.Series
    test_positions: pd.Series


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    """Aggregate walk-forward result for one ticker."""

    ticker: str
    folds: list[WalkForwardFold] = field(default_factory=list)
    concatenated_equity: pd.Series | None = None
    concatenated_returns: pd.Series | None = None
