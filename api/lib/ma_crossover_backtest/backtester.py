"""Vectorised backtest engine.

The whole computation is one pass of pandas/numpy:

    signal     = generate_signal(close, config)         # uses data <= bar t
    position   = signal.shift(1).fillna(0)              # held over bar t+1
    bar_return = close.pct_change()                     # asset return on bar t
    gross_r_t  = position_t * bar_return_t              # earned by strategy
    cost_t     = |Delta position_t| * cost_per_side     # drag on changes
    net_r_t    = gross_r_t - cost_t
    equity_t   = product(1 + net_r) * initial_cash

There is no Python-level loop over bars in the hot path. The trade extraction
loops over entries (typically <100 per backtest) for readability - vectorising
the ledger build is not worth the obscurity at this scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ma_backtester.config import StrategyConfig
from ma_backtester.costs import CostModel, zero_cost_model
from ma_backtester.results import BacktestResult
from ma_backtester.strategy import (
    buy_and_hold_position,
    generate_signal,
    signal_to_position,
)


def _extract_trades(
    positions: pd.Series,
    close: pd.Series,
    net_returns: pd.Series,
    gross_returns: pd.Series,
) -> pd.DataFrame:
    delta = positions.diff().fillna(positions.iloc[0])
    entry_idxs = np.where(delta.to_numpy() > 0)[0]
    exit_idxs = np.where(delta.to_numpy() < 0)[0]

    rows: list[dict[str, object]] = []
    n = len(positions)

    for entry_idx in entry_idxs:
        later_exits = exit_idxs[exit_idxs > entry_idx]
        exit_idx = int(later_exits[0]) if later_exits.size else n
        entry_px_idx = max(0, int(entry_idx) - 1)
        exit_px_idx = exit_idx - 1

        gross_window = gross_returns.iloc[entry_idx:exit_idx]
        net_window = net_returns.iloc[entry_idx:exit_idx]

        rows.append(
            {
                "entry_date": close.index[entry_px_idx],
                "exit_date": close.index[exit_px_idx],
                "entry_price": float(close.iloc[entry_px_idx]),
                "exit_price": float(close.iloc[exit_px_idx]),
                "bars_held": int(exit_idx - entry_idx),
                "gross_return": float((1.0 + gross_window).prod() - 1.0),
                "net_return": float((1.0 + net_window).prod() - 1.0),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "entry_date",
                "exit_date",
                "entry_price",
                "exit_price",
                "bars_held",
                "gross_return",
                "net_return",
            ]
        )
    return pd.DataFrame(rows)


def run_backtest(
    *,
    close: pd.Series,
    strategy_config: StrategyConfig,
    cost_model: CostModel | None = None,
    initial_cash: float = 100_000.0,
) -> BacktestResult:
    """Run a single-asset, long/flat MA-crossover backtest.

    Parameters
    ----------
    close : pd.Series
        Adjusted close prices indexed by trading date.
    strategy_config : StrategyConfig
        ``fast_window`` / ``slow_window`` for the SMA pair.
    cost_model : CostModel | None
        Defaults to zero-cost. Pass ``FixedBpsCost(CostConfig(per_side_bps=5))``
        for a 5 bp-per-side world.
    initial_cash : float
        Starting equity. Affects the y-axis of the equity curve only.

    Returns
    -------
    BacktestResult
        Equity curve, position series, trade ledger, returns, costs.
    """
    if cost_model is None:
        cost_model = zero_cost_model()

    signal = generate_signal(close, strategy_config)
    position = signal_to_position(signal)

    bar_returns = close.pct_change().fillna(0.0)
    bar_returns.name = "bar_return"

    gross_returns = (position * bar_returns).astype("float64")
    gross_returns.name = "gross_return"

    cost_drag = cost_model.cost_series(position)
    net_returns = (gross_returns - cost_drag).astype("float64")
    net_returns.name = "net_return"

    equity = ((1.0 + net_returns).cumprod() * initial_cash).astype("float64")
    equity.name = "equity"

    trades = _extract_trades(position, close, net_returns, gross_returns)

    return BacktestResult(
        equity=equity,
        positions=position,
        daily_returns=net_returns,
        trades=trades,
        gross_returns=gross_returns,
        costs=cost_drag,
    )


def run_buy_and_hold(
    *,
    close: pd.Series,
    cost_model: CostModel | None = None,
    initial_cash: float = 100_000.0,
) -> BacktestResult:
    """Buy-and-hold baseline using the same engine - sanity check + benchmark."""
    if cost_model is None:
        cost_model = zero_cost_model()

    position = buy_and_hold_position(close)
    bar_returns = close.pct_change().fillna(0.0)
    gross_returns = (position * bar_returns).astype("float64")
    cost_drag = cost_model.cost_series(position)
    net_returns = (gross_returns - cost_drag).astype("float64")
    equity = ((1.0 + net_returns).cumprod() * initial_cash).astype("float64")
    trades = _extract_trades(position, close, net_returns, gross_returns)

    return BacktestResult(
        equity=equity,
        positions=position,
        daily_returns=net_returns,
        trades=trades,
        gross_returns=gross_returns,
        costs=cost_drag,
    )


def sweep(
    *,
    close: pd.Series,
    grid: list[StrategyConfig],
    cost_model: CostModel | None = None,
    initial_cash: float = 100_000.0,
) -> dict[StrategyConfig, BacktestResult]:
    """Sequential parameter sweep. Returns a dict keyed by config."""
    if cost_model is None:
        cost_model = zero_cost_model()
    return {
        cfg: run_backtest(
            close=close,
            strategy_config=cfg,
            cost_model=cost_model,
            initial_cash=initial_cash,
        )
        for cfg in grid
    }
