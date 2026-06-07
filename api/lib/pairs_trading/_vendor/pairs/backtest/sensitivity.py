"""Cost / borrow sensitivity grid and break-even helpers.

:func:`sensitivity_grid` runs the backtest under a Cartesian product of
``(cost_bps, borrow_bps)`` and returns a long-form DataFrame with one row per
combination. :func:`break_even_cost` linearly interpolates the cost level at
which the *net* Sharpe crosses zero for a fixed borrow rate -- a quick health
check for how much edge a strategy actually has after costs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pairs._exceptions import InputError
from pairs.backtest.accounting import FixedCommission
from pairs.backtest.borrow import ConstantBorrow
from pairs.backtest.costs import CompositeCost
from pairs.backtest.engine import backtest_pair
from pairs.backtest.slippage import ConstantBpsSlippage

__all__ = [
    "break_even_cost",
    "sensitivity_grid",
]


def sensitivity_grid(
    prices_a: pd.Series,
    prices_b: pd.Series,
    signal: pd.Series,
    hedge_ratio: float | pd.Series,
    *,
    cost_grid: dict[str, list[float]] | list[float],
    borrow_grid: list[float],
    capital: float = 1.0,
    **engine_kwargs: Any,
) -> pd.DataFrame:
    """Sweep a 2-D cost/borrow grid and report summary metrics per combination.

    Parameters
    ----------
    prices_a, prices_b : pandas.Series
        Forwarded to :func:`pairs.backtest.engine.backtest_pair`.
    signal : pandas.Series
        Discrete signal from :func:`pairs.strategy.generate_signals`.
    hedge_ratio : float or pandas.Series
        Static or instantaneous hedge ratio.
    cost_grid : dict or list
        Either ``{"bps": [...]}`` or a flat list of round-trip cost values in
        basis points. Each value is converted into a
        :class:`ConstantBpsSlippage` model.
    borrow_grid : list of float
        Annualised borrow rates in basis points.
    capital : float, default ``1.0``
        Notional capital forwarded to the engine.
    **engine_kwargs
        Additional keyword arguments forwarded to ``backtest_pair``.

    Returns
    -------
    pandas.DataFrame
        Long-form table with columns ``cost_bps``, ``borrow_bps``, ``sharpe``,
        ``total_return`` and ``n_trades``. One row per combination.
    """
    if isinstance(cost_grid, dict):
        if "bps" not in cost_grid:
            msg = "cost_grid dict must contain a 'bps' key"
            raise InputError(msg)
        cost_values = [float(v) for v in cost_grid["bps"]]
    else:
        cost_values = [float(v) for v in cost_grid]
    borrow_values = [float(v) for v in borrow_grid]
    if not cost_values or not borrow_values:
        msg = "cost_grid and borrow_grid must be non-empty"
        raise InputError(msg)

    rows: list[dict[str, float]] = []
    for cost_bps in cost_values:
        for borrow_bps in borrow_values:
            model = CompositeCost(
                slippage=ConstantBpsSlippage(bps=cost_bps),
                commission=FixedCommission(per_trade=0.0),
                borrow=ConstantBorrow(rate_bps_annual=borrow_bps),
                name=f"grid_{cost_bps:g}bps_{borrow_bps:g}bp_borrow",
            )
            result = backtest_pair(
                prices_a,
                prices_b,
                signal,
                hedge_ratio,
                cost_model=model,
                capital=capital,
                **engine_kwargs,
            )
            rows.append(
                {
                    "cost_bps": cost_bps,
                    "borrow_bps": borrow_bps,
                    "sharpe": float(result.metrics["sharpe_net"]),
                    "total_return": float(result.metrics["total_pnl_net"] / float(capital)),
                    "n_trades": int(result.metrics["n_trades"]),
                }
            )
    return pd.DataFrame(rows)


def break_even_cost(
    grid_result: pd.DataFrame,
    *,
    borrow_bps: float = 0.0,
) -> float:
    """Linearly interpolate the cost level where net Sharpe crosses zero.

    Parameters
    ----------
    grid_result : pandas.DataFrame
        Output of :func:`sensitivity_grid`.
    borrow_bps : float, default ``0.0``
        Borrow rate to slice on. The function selects the rows whose
        ``borrow_bps`` matches ``borrow_bps`` (within ``1e-9``) and interpolates
        along ``cost_bps``.

    Returns
    -------
    float
        Cost in basis points at which Sharpe crosses zero. Returns
        ``+inf`` when no row in the slice has Sharpe <= 0 (the strategy stays
        profitable across the whole grid) and ``0.0`` when every row already
        has Sharpe <= 0.
    """
    if "borrow_bps" not in grid_result.columns or "cost_bps" not in grid_result.columns:
        msg = "grid_result must contain cost_bps and borrow_bps columns"
        raise InputError(msg)
    slice_df = grid_result.loc[np.isclose(grid_result["borrow_bps"], float(borrow_bps))]
    if slice_df.empty:
        msg = f"no rows in grid_result match borrow_bps={borrow_bps!r}"
        raise InputError(msg)
    slice_df = slice_df.sort_values("cost_bps")
    costs = slice_df["cost_bps"].to_numpy(dtype=float)
    sharpes = slice_df["sharpe"].to_numpy(dtype=float)
    if (sharpes <= 0.0).all():
        return float(costs[0])
    if (sharpes > 0.0).all():
        return float("inf")
    # Find the segment where sharpe transitions from positive to non-positive.
    for i in range(len(costs) - 1):
        s0, s1 = sharpes[i], sharpes[i + 1]
        if s0 > 0.0 >= s1:
            c0, c1 = costs[i], costs[i + 1]
            if s0 == s1:
                return float(c0)
            return float(c0 + (c1 - c0) * (s0 - 0.0) / (s0 - s1))
    return float(costs[-1])
