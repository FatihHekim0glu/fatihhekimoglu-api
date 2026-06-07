"""High-level signal generator for pairs strategies.

:func:`generate_signals` is a thin wrapper around the rule state machine in
:mod:`pairs.strategy.rules` that takes care of:

* parameter validation (delegated through :class:`pairs.strategy.StrategyConfig`
  whenever the caller wants the dataclass form),
* deriving the default ``time_stop_bars`` from a half-life when one is
  available, and
* returning the position series with the same index and dtype contract used
  elsewhere in the package.

The function returns a plain :class:`pandas.Series` so it can be plugged into
``backtest_pair`` without any further conversion.
"""

from __future__ import annotations

import pandas as pd

from pairs._exceptions import InputError
from pairs.strategy.rules import apply_rules

__all__ = ["default_time_stop", "generate_signals"]


def default_time_stop(half_life: float) -> int:
    """Return the default time-stop derived from ``half_life``.

    The convention is ``max(2, 2 * round(half_life))``: two half-lives is the
    point at which an OU process has decayed by roughly 75 %, so a position
    that has not closed by then is almost certainly stuck.
    """
    if float(half_life) <= 0.0:
        msg = f"half_life must be positive, got {half_life!r}"
        raise InputError(msg)
    return max(2, 2 * int(round(float(half_life))))


def generate_signals(
    zscore: pd.Series,
    *,
    z_entry: float = 2.0,
    z_exit: float = 0.5,
    z_stop: float = 3.0,
    time_stop_bars: int | None = None,
    half_life: float | None = None,
) -> pd.Series:
    """Turn a z-score into a discrete ``{-1, 0, +1}`` position series.

    Parameters
    ----------
    zscore : pandas.Series
        Leak-free z-score (see :func:`pairs.spread.zscore`). The output index
        and length match this Series exactly.
    z_entry, z_exit, z_stop : float
        Hysteresis bands. Defaults are 2.0 / 0.5 / 3.0 in z-units.
    time_stop_bars : int or None, optional
        Maximum bars to hold a position. If omitted *and* ``half_life`` is
        provided, defaults to :func:`default_time_stop`. Pass an explicit
        integer to override.
    half_life : float or None, optional
        OU half-life used only to derive ``time_stop_bars`` when the latter is
        not supplied. Ignored otherwise.

    Returns
    -------
    pandas.Series
        Position series with dtype ``int8`` and the same index as ``zscore``.

    Raises
    ------
    pairs.InputError
        On invalid thresholds, non-Series input, or non-positive half-life.

    Notes
    -----
    The function is pure and leak-free: at each bar ``t`` only the z-score
    values in ``zscore.iloc[:t + 1]`` are consulted. The downstream backtester
    must still apply a one-bar execution lag (see
    :func:`pairs.backtest.backtest_pair`).
    """
    if not isinstance(zscore, pd.Series):
        msg = "zscore must be a pandas Series"
        raise InputError(msg)

    if time_stop_bars is None and half_life is not None:
        time_stop_bars = default_time_stop(float(half_life))

    raw = apply_rules(
        zscore,
        z_entry=float(z_entry),
        z_exit=float(z_exit),
        z_stop=float(z_stop),
        time_stop_bars=time_stop_bars,
    )

    out = pd.Series(raw.positions, index=zscore.index, dtype="int8", name="position")
    return out
