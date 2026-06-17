"""Anchored/expanding walk-forward with purge + embargo and per-fold scaler fit.

This subpackage is the heart of the de-leak: every fold recomputes features and
re-fits the standardizer on its TRAIN slice only, a ``look_back``-sized purge
ensures no sequence window straddles a split, and an embargo follows each test
block. Importing this subpackage has no side effects.
"""

from __future__ import annotations

from lstmforecast.walkforward.costs import FixedBpsCost
from lstmforecast.walkforward.engine import (
    FoldResult,
    WalkForwardConfig,
    WalkForwardResult,
    make_folds,
    run_walk_forward,
)

__all__ = [
    "FixedBpsCost",
    "FoldResult",
    "WalkForwardConfig",
    "WalkForwardResult",
    "make_folds",
    "run_walk_forward",
]
