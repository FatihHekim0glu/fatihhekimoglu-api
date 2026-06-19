"""Signals subpackage: pure, strictly-causal target-position generators.

Exposes the signal library — ``ma_crossover``, ``momentum``, and ``flat`` — each a
PURE function mapping the closed-bar history up to and INCLUDING bar ``t`` to a
target position for bar ``t+1`` (NEVER reading the forming/partial bar or any
future bar). Importing this subpackage has no side effects.
"""

from __future__ import annotations

from algosystem.signals.library import (
    SignalSpec,
    build_signal,
    flat,
    ma_crossover,
    momentum,
)

__all__ = [
    "SignalSpec",
    "build_signal",
    "flat",
    "ma_crossover",
    "momentum",
]
