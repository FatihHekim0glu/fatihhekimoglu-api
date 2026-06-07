"""Transaction cost models.

Costs apply to turnover (notional traded). For a long/flat strategy with
unit-leverage positions in {0, 1}, the turnover per bar is the absolute
change in position, and the cost is::

    cost_t = |position_t - position_{t-1}| * (bps / 10_000)

This is subtracted from gross returns to produce net returns.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from ma_backtester.config import CostConfig


class CostModel(Protocol):
    """Anything that can turn a position series into a per-bar cost drag."""

    def cost_series(self, positions: pd.Series) -> pd.Series: ...  # pragma: no cover


class FixedBpsCost:
    """Linear cost: a fixed number of basis points per side, on turnover."""

    def __init__(self, config: CostConfig) -> None:
        self._config = config

    @property
    def config(self) -> CostConfig:
        return self._config

    def cost_series(self, positions: pd.Series) -> pd.Series:
        """Per-bar cost as a fraction of equity.

        The first bar's cost reflects opening the initial position from
        cash. NaN entries in ``positions`` are treated as flat (zero), so
        a raw un-shifted signal passed in by mistake cannot silently
        poison the downstream equity curve.
        """
        clean = positions.fillna(0.0)
        turnover = clean.diff().abs()
        turnover.iloc[0] = abs(clean.iloc[0])
        result: pd.Series = turnover * self._config.per_side_fraction
        result.name = "cost"
        return result


def zero_cost_model() -> FixedBpsCost:
    return FixedBpsCost(CostConfig(per_side_bps=0.0))
