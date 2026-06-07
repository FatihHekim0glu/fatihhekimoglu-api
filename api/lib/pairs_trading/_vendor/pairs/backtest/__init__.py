"""Backtest engine, cost models and sensitivity sweeps.

The submodules provide three layers:

* **Engine** -- :func:`backtest_pair` runs the trade simulation given a signal
  series and a cost model, returning a :class:`BacktestResult`.
* **Cost models** -- :class:`CostModel` (Protocol), :class:`CompositeCost`,
  and concrete implementations (:class:`ConstantBpsSlippage`,
  :class:`HalfSpreadSlippage`, :class:`AlmgrenChrissSlippage`,
  :class:`FixedCommission`, :class:`PerShareCommission`,
  :class:`ConstantBorrow`, :class:`ProfileBorrow`).
* **Profiles & sweeps** -- :func:`load_profile` materialises a
  :class:`CompositeCost` from a YAML file; :func:`sensitivity_grid` and
  :func:`break_even_cost` quantify how sensitive net Sharpe is to costs.
"""

from __future__ import annotations

from pairs.backtest.accounting import FixedCommission, PerShareCommission
from pairs.backtest.borrow import ConstantBorrow, ProfileBorrow
from pairs.backtest.costs import CompositeCost, CostModel
from pairs.backtest.engine import backtest_pair
from pairs.backtest.profiles import load_profile
from pairs.backtest.results import BacktestConfig, BacktestResult
from pairs.backtest.sensitivity import break_even_cost, sensitivity_grid
from pairs.backtest.slippage import (
    AlmgrenChrissSlippage,
    ConstantBpsSlippage,
    HalfSpreadSlippage,
)

__all__ = [
    "AlmgrenChrissSlippage",
    "BacktestConfig",
    "BacktestResult",
    "CompositeCost",
    "ConstantBorrow",
    "ConstantBpsSlippage",
    "CostModel",
    "FixedCommission",
    "HalfSpreadSlippage",
    "PerShareCommission",
    "ProfileBorrow",
    "backtest_pair",
    "break_even_cost",
    "load_profile",
    "sensitivity_grid",
]
