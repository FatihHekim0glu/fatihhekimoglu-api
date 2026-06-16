"""Temporal stability of the cluster assignment.

Stability is measured by re-fitting clusters on rolling windows and comparing
adjacent-window labelings. The headline scalar is the mean adjacent-window
Adjusted Rand Index (ARI); label alignment across windows (and birth/death of
clusters when ``k`` changes) is handled by Hungarian / max-Jaccard matching.

- :mod:`stockclusters.stability.resample` — rolling/resampled cluster re-fits.
- :mod:`stockclusters.stability.ari` — adjacent-window ARI (headline scalar).
- :mod:`stockclusters.stability.align` — cross-window label alignment.

The frozen :class:`StabilityResult` is the common return bundle.

Importing this subpackage has no side effects.
"""

from __future__ import annotations

from stockclusters.stability.align import align_labels, births_and_deaths
from stockclusters.stability.ari import adjacent_window_ari, adjusted_rand_index
from stockclusters.stability.resample import StabilityResult, rolling_stability

__all__ = [
    "StabilityResult",
    "adjacent_window_ari",
    "adjusted_rand_index",
    "align_labels",
    "births_and_deaths",
    "rolling_stability",
]
