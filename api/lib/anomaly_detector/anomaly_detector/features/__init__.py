"""Causal per-day feature engineering for the anomaly detectors.

Turns a price/volume series into the ``.shift(1)``-safe feature matrix consumed
by every detector. Importing this subpackage has no side effects.
"""

from __future__ import annotations

from anomaly_detector.features.engineer import (
    DEFAULT_WINDOW,
    FEATURE_NAMES,
    causal_return_zscore,
    engineer_features,
)

__all__ = [
    "DEFAULT_WINDOW",
    "FEATURE_NAMES",
    "causal_return_zscore",
    "engineer_features",
]
