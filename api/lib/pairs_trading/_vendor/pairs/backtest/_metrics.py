"""Internal performance metrics used by the backtest engine.

These helpers are not part of the public API; the engine calls them to build
the ``metrics`` dict on :class:`pairs.backtest.results.BacktestResult`. Keeping
them here means callers can import the public symbols without picking up these
implementation details.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

__all__ = [
    "hit_rate",
    "max_drawdown",
    "sharpe",
    "turnover",
]


def sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Return the annualised Sharpe ratio of a per-period return series.

    Returns are assumed to be excess (or equivalently the risk-free rate is
    zero). The function returns ``0.0`` when there is no variation in the
    inputs, so it never raises on degenerate inputs.
    """
    values = pd.Series(returns).dropna().to_numpy(dtype=float)
    if values.size < 2:
        return 0.0
    std = float(np.std(values, ddof=1))
    if std == 0.0 or not math.isfinite(std):
        return 0.0
    mean = float(np.mean(values))
    return float(mean / std * math.sqrt(float(periods_per_year)))


def max_drawdown(equity: pd.Series) -> float:
    """Return the maximum drawdown of an equity curve as a *positive* fraction.

    A flat or empty equity curve yields ``0.0``.
    """
    values = pd.Series(equity).to_numpy(dtype=float)
    if values.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(values)
    valid = running_max != 0.0
    drawdowns = np.zeros_like(values)
    drawdowns[valid] = (values[valid] - running_max[valid]) / running_max[valid]
    return float(-drawdowns.min()) if drawdowns.size > 0 else 0.0


def turnover(positions: pd.Series | pd.DataFrame) -> float:
    """Return total turnover as the sum of absolute position changes."""
    pos = positions.to_numpy(dtype=float)
    if pos.ndim == 1:
        return float(np.abs(np.diff(pos, prepend=0.0)).sum())
    return float(np.abs(np.diff(pos, axis=0, prepend=np.zeros((1, pos.shape[1])))).sum())  # type: ignore[misc]


def hit_rate(trades: pd.DataFrame) -> float:
    """Return the fraction of trades with strictly positive PnL.

    An empty trades log yields ``0.0`` rather than ``NaN`` so the metric is
    safe to serialise into the manifest.
    """
    if trades is None or len(trades) == 0 or "pnl" not in trades.columns:
        return 0.0
    pnl = trades["pnl"].to_numpy(dtype=float)
    if pnl.size == 0:
        return 0.0
    return float((pnl > 0.0).mean())
