"""Data layer: the seeded synthetic event panel and the real-data loaders.

The synthetic panel is the default everywhere; the loaders add an optional,
gracefully-degrading FRED (keyless) + Polygon (PIT) real-data path. Importing
this subpackage has no side effects (lazy network clients).
"""

from __future__ import annotations

from fedcausal.data.loaders import (
    fetch_fred_series,
    fetch_polygon_returns,
    load_event_panel,
    load_surprise_labels,
)
from fedcausal.data.synthetic import (
    SyntheticPanel,
    pure_noise_panel,
    rate_sensitive_panel,
    synthetic_event_panel,
)

__all__ = [
    "SyntheticPanel",
    "fetch_fred_series",
    "fetch_polygon_returns",
    "load_event_panel",
    "load_surprise_labels",
    "pure_noise_panel",
    "rate_sensitive_panel",
    "synthetic_event_panel",
]
