"""Result containers for the backtest engine.

:class:`BacktestResult` bundles every artefact a backtest produces: the equity
curve, gross and net per-bar returns, the trades log, the position book, a
per-bar cost decomposition, summary metrics and a serialisable manifest.
:class:`BacktestConfig` mirrors the keyword arguments of
:func:`pairs.backtest.engine.backtest_pair` so a run can be replayed from
configuration alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from pairs._exceptions import InputError

__all__ = [
    "BacktestConfig",
    "BacktestResult",
]


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Output of :func:`pairs.backtest.engine.backtest_pair`.

    Attributes
    ----------
    equity : pandas.Series
        Cumulative equity curve, starting at ``capital``.
    returns : pandas.Series
        Per-bar *net* returns (after all costs).
    gross_returns : pandas.Series
        Per-bar returns *before* costs.
    trades : pandas.DataFrame
        One row per trade (open + close pair). Columns include
        ``entry_time``, ``exit_time``, ``side``, ``pnl``, ``exit_reason``.
    positions : pandas.DataFrame
        Per-bar position book with columns ``position``, ``shares_a``,
        ``shares_b``.
    cost_decomposition : pandas.DataFrame
        Per-bar costs split into ``commission``, ``slippage``, ``borrow``,
        ``dividend``. Sums of the four columns equal the total cost drag.
    metrics : dict
        Summary statistics (Sharpe, max drawdown, turnover, hit rate, etc.).
    manifest : dict
        Serialisable reproducibility metadata.
    """

    equity: pd.Series
    returns: pd.Series
    gross_returns: pd.Series
    trades: pd.DataFrame
    positions: pd.DataFrame
    cost_decomposition: pd.DataFrame
    metrics: dict[str, Any]
    manifest: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.equity, pd.Series):
            msg = "equity must be a pandas Series"
            raise InputError(msg)
        for name in ("returns", "gross_returns"):
            series = getattr(self, name)
            if not isinstance(series, pd.Series):
                msg = f"{name} must be a pandas Series"
                raise InputError(msg)
        for name in ("trades", "positions", "cost_decomposition"):
            frame = getattr(self, name)
            if not isinstance(frame, pd.DataFrame):
                msg = f"{name} must be a pandas DataFrame"
                raise InputError(msg)
        expected_cols = {"commission", "slippage", "borrow", "dividend"}
        if not expected_cols.issubset(set(self.cost_decomposition.columns)):
            msg = (
                "cost_decomposition must contain commission/slippage/borrow/dividend "
                f"columns, got {list(self.cost_decomposition.columns)!r}"
            )
            raise InputError(msg)


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Replayable configuration mirror of :func:`backtest_pair` kwargs."""

    capital: float = 1.0
    sizing: Literal["dollar_neutral", "beta_neutral", "unit"] = "dollar_neutral"
    use_open_for_execution: bool = False
    has_adv: bool = False
    has_dividends: bool = False
    has_blacklist: bool = False
    cost_model_name: str = "composite"
    extras: dict[str, Any] = field(default_factory=dict)
