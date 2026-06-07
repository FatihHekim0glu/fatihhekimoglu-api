"""Trading-rule generation for pairs strategies.

Given a leak-free z-score (see :mod:`pairs.spread.zscore`) this subpackage turns
it into a discrete position series in ``{-1, 0, +1}`` using a hysteresis state
machine with optional time-stop and z-stop overrides.

Public API:

* :func:`generate_signals` -- the main entry point. Returns a position series.
* :class:`StrategyConfig` -- frozen container for rule parameters.
* :class:`SignalDiagnostics` -- bookkeeping returned alongside the positions
  (entries, exits, stops, time-stops, blacklist exits).
"""

from __future__ import annotations

from pairs.strategy.results import SignalDiagnostics, StrategyConfig
from pairs.strategy.signals import generate_signals

__all__ = [
    "SignalDiagnostics",
    "StrategyConfig",
    "generate_signals",
]
