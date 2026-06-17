"""Anchored/expanding walk-forward engine with strict fit-on-train-only.

The leakage-control heart of the project: every estimator is fit inside each
walk-forward train fold and only then forecasts the disjoint test fold.
Importing this subpackage has no side effects.
"""

from __future__ import annotations

from volforecast.walkforward.engine import (
    DEFAULT_MODELS,
    WalkForwardConfig,
    WalkForwardResult,
    run_walk_forward,
)

__all__ = [
    "DEFAULT_MODELS",
    "WalkForwardConfig",
    "WalkForwardResult",
    "run_walk_forward",
]
