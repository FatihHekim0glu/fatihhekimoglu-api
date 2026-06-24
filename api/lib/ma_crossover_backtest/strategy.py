"""Moving-average crossover signal generation.

Signal convention
-----------------
The signal at bar ``t`` is computed from information available up to and
including the close of bar ``t``. The caller (or engine) is responsible for
applying ``.shift(1)`` to obtain the *realisable* position held over bar
``t+1``. This module returns the raw signal - the shift discipline is
documented and enforced at the boundary, not buried inside the strategy.

Signal rule
-----------
- ``fast_sma(t) > slow_sma(t)``  ->  position 1 (long)
- ``fast_sma(t) <= slow_sma(t)`` ->  position 0 (flat)

The warm-up period (first ``slow_window - 1`` bars) yields NaN signal, which
the engine treats as flat.
"""

from __future__ import annotations

import pandas as pd

from ma_backtester.config import StrategyConfig


def generate_signal(close: pd.Series, config: StrategyConfig) -> pd.Series:
    """Compute the raw (un-shifted) MA-crossover signal.

    Parameters
    ----------
    close : pd.Series
        Adjusted close prices, indexed by date.
    config : StrategyConfig
        Holds ``fast_window`` and ``slow_window``.

    Returns
    -------
    pd.Series
        Series in {0, 1} with NaN during warm-up.

    Notes
    -----
    This function reads only ``close[:t+1]`` to compute ``signal[t]`` - it
    does not peek into the future. The no-lookahead property is verified by
    ``tests/test_no_lookahead.py``.
    """
    if not isinstance(close, pd.Series):
        raise TypeError("close must be a pandas Series")

    fast = close.rolling(window=config.fast_window, min_periods=config.fast_window).mean()
    slow = close.rolling(window=config.slow_window, min_periods=config.slow_window).mean()

    signal = (fast > slow).astype("float64")
    warmup_mask = fast.isna() | slow.isna()
    signal[warmup_mask] = float("nan")
    signal.name = "signal"
    return signal


def signal_to_position(signal: pd.Series) -> pd.Series:
    """Apply the next-bar execution discipline.

    Converts a same-bar signal into the position you *hold* over the next bar.
    NaN warm-up entries become flat (0).
    """
    position: pd.Series = signal.shift(1).fillna(0.0).astype("float64")
    position.name = "position"
    return position


def buy_and_hold_position(close: pd.Series) -> pd.Series:
    """Position series for the benchmark: always long from day 1."""
    position = pd.Series(1.0, index=close.index, dtype="float64", name="position")
    return position
