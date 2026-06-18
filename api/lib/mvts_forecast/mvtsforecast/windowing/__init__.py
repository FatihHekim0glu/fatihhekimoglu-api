"""Leakage-safe windowing: purge/embargo walk-forward, RevIN, train-only scaler.

This subpackage is the heart of the de-leak. Sliding windows never read the
target's same-step transform; the standardizer is fitted on the TRAIN fold only
and applied (never re-fitted) to test; the walk-forward purge (>= ``look_back``)
guarantees no window straddles a split; and RevIN instance-normalizes each window
from its OWN statistics only. Importing this subpackage has no side effects.
"""

from __future__ import annotations

from mvtsforecast.windowing.costs import FixedBpsCost
from mvtsforecast.windowing.revin import (
    RevInStats,
    revin_denormalize,
    revin_normalize,
)
from mvtsforecast.windowing.windows import (
    Fold,
    Standardizer,
    WindowSpec,
    assert_no_target_leakage,
    fit_standardizer,
    make_folds,
    make_windows,
)

__all__ = [
    "FixedBpsCost",
    "Fold",
    "RevInStats",
    "Standardizer",
    "WindowSpec",
    "assert_no_target_leakage",
    "fit_standardizer",
    "make_folds",
    "make_windows",
    "revin_denormalize",
    "revin_normalize",
]
