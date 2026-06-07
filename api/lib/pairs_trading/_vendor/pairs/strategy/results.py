"""Frozen result dataclasses for the strategy subpackage.

:class:`StrategyConfig` captures the rule parameters that drive
:func:`pairs.strategy.signals.generate_signals` and the sizing convention used
downstream by the backtester. :class:`SignalDiagnostics` records the per-bar
flags produced by the rule state machine so callers can audit *why* a position
flipped.

All numerical attributes are validated in ``__post_init__`` so a malformed
config can never escape its constructor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from pairs._exceptions import InputError

__all__ = [
    "SignalDiagnostics",
    "StrategyConfig",
]


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    """Parameter bundle for the pairs-trading rule.

    Attributes
    ----------
    z_entry : float
        Absolute z-score that triggers a new position. Must be > 0.
    z_exit : float
        Absolute z-score below which an open position is closed. Must satisfy
        ``0 <= z_exit < z_entry``.
    z_stop : float
        Stop-loss z-score. ``|z| > z_stop`` forces an immediate exit. Must
        satisfy ``z_stop > z_entry``.
    time_stop_bars : int or None, optional
        Maximum bars to hold a position. ``None`` disables the time stop.
    half_life : float or None, optional
        Half-life used to auto-derive ``time_stop_bars`` when the caller does
        not pass one explicitly: ``max(2, 2 * round(half_life))``.
    sizing : {"dollar_neutral", "beta_neutral", "unit"}
        Leg-sizing convention consumed by the backtester.
    capital : float
        Notional capital allocated to the pair. Must be > 0.
    """

    z_entry: float = 2.0
    z_exit: float = 0.5
    z_stop: float = 3.0
    time_stop_bars: int | None = None
    half_life: float | None = None
    sizing: Literal["dollar_neutral", "beta_neutral", "unit"] = "dollar_neutral"
    capital: float = 1.0

    def __post_init__(self) -> None:
        entry = float(self.z_entry)
        exit_ = float(self.z_exit)
        stop = float(self.z_stop)
        if entry <= 0.0:
            msg = f"z_entry must be > 0, got {self.z_entry!r}"
            raise InputError(msg)
        if exit_ < 0.0:
            msg = f"z_exit must be >= 0, got {self.z_exit!r}"
            raise InputError(msg)
        if not exit_ < entry:
            msg = f"z_exit must be < z_entry, got exit={exit_!r}, entry={entry!r}"
            raise InputError(msg)
        if not stop > entry:
            msg = f"z_stop must be > z_entry, got stop={stop!r}, entry={entry!r}"
            raise InputError(msg)
        if self.time_stop_bars is not None and int(self.time_stop_bars) <= 0:
            msg = f"time_stop_bars must be positive when set, got {self.time_stop_bars!r}"
            raise InputError(msg)
        if self.half_life is not None and float(self.half_life) <= 0.0:
            msg = f"half_life must be positive when set, got {self.half_life!r}"
            raise InputError(msg)
        if self.sizing not in {"dollar_neutral", "beta_neutral", "unit"}:
            msg = f"sizing must be one of dollar_neutral/beta_neutral/unit, got {self.sizing!r}"
            raise InputError(msg)
        if float(self.capital) <= 0.0:
            msg = f"capital must be positive, got {self.capital!r}"
            raise InputError(msg)


@dataclass(frozen=True, slots=True)
class SignalDiagnostics:
    """Per-bar flags emitted alongside the position series.

    Attributes
    ----------
    positions : pandas.Series
        Discrete position series in ``{-1, 0, +1}`` indexed identically to the
        driving z-score.
    entries : pandas.Series
        Boolean flag, ``True`` at bars where a flat -> non-flat transition
        occurs (entry into a new trade).
    exits : pandas.Series
        Boolean flag, ``True`` at bars where a non-flat -> flat transition
        occurs (any reason).
    stops : pandas.Series
        Boolean flag, ``True`` at exits driven by ``|z| > z_stop``.
    time_stops_triggered : pandas.Series
        Boolean flag, ``True`` at exits driven by the time-stop counter.
    blacklist_exits : pandas.Series
        Boolean flag, ``True`` at exits driven by an external kill-switch
        (set by the engine; the rule machine never writes it).
    """

    positions: pd.Series
    entries: pd.Series
    exits: pd.Series
    stops: pd.Series
    time_stops_triggered: pd.Series
    blacklist_exits: pd.Series

    def __post_init__(self) -> None:
        ref_index = self.positions.index
        for name in ("entries", "exits", "stops", "time_stops_triggered", "blacklist_exits"):
            series: pd.Series = getattr(self, name)
            if not isinstance(series, pd.Series):
                msg = f"{name} must be a pandas Series"
                raise InputError(msg)
            if len(series) != len(ref_index) or not series.index.equals(ref_index):
                msg = f"{name} index must match positions index"
                raise InputError(msg)
