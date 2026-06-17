"""Per-fold feature engineering and supervised-sequence construction.

Both modules enforce the no-lookahead discipline that fixes the original repo's
leakage: technical features are computed on strictly-PAST bars (``.shift(1)``),
returns use ``pct_change(fill_method=None)``, and sequence windows never straddle
a walk-forward split. Importing this subpackage has no side effects.
"""

from __future__ import annotations

from lstmforecast.features.engineer import FeatureSpec, engineer_features
from lstmforecast.features.sequences import create_sequences

__all__ = [
    "FeatureSpec",
    "create_sequences",
    "engineer_features",
]
