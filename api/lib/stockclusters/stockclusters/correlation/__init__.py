"""Correlation estimation, RMT denoising, and the Mantegna distance metric.

This subpackage turns a wide panel of asset log-returns into the inputs that the
clustering stage consumes: a (optionally RMT-denoised) correlation matrix, and a
Mantegna ultrametric-compatible distance matrix.

- :mod:`stockclusters.correlation.estimate` — log-return correlation estimation.
- :mod:`stockclusters.correlation.rmt` — Marchenko-Pastur eigenvalue clipping
  (re-exported from the reused :mod:`stockclusters.estimators.rmt`).
- :mod:`stockclusters.correlation.distance` — Mantegna distance
  :math:`d_{ij} = \\sqrt{2(1 - \\rho_{ij})}` plus MST / subdominant-ultrametric
  helpers.

Importing this subpackage has no side effects.
"""

from __future__ import annotations

from stockclusters.correlation.distance import (
    mantegna_distance,
    minimum_spanning_tree,
    subdominant_ultrametric,
)
from stockclusters.correlation.estimate import correlation_matrix, log_returns
from stockclusters.correlation.rmt import marchenko_pastur_clip

__all__ = [
    "correlation_matrix",
    "log_returns",
    "mantegna_distance",
    "marchenko_pastur_clip",
    "minimum_spanning_tree",
    "subdominant_ultrametric",
]
