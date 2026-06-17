"""Daily sentiment signal construction (train-only scaler + shift(1)).

Standardizes the daily sentiment aggregate with a scaler FIT ON TRAIN ONLY, maps
the standardized score to a per-ticker position (long/short or long/flat), and
applies ``signal.shift(1)`` so a position earned on day ``t`` was decided strictly
before ``t`` (no same-bar lookahead).

Importing this subpackage has no side effects.
"""

from __future__ import annotations

from wsb_sentiment.signal.build import (
    SignalSpec,
    StandardizerState,
    build_positions,
    fit_standardizer,
)

__all__ = [
    "SignalSpec",
    "StandardizerState",
    "build_positions",
    "fit_standardizer",
]
