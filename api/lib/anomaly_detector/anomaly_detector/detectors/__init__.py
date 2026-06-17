"""Unsupervised anomaly detectors and their shared result container.

Two independent, leakage-safe detectors over a per-day feature matrix:

- :class:`IsolationForestDetector` — an :mod:`sklearn` Isolation Forest
  (Liu, Ting & Zhou 2008); anomaly score ``= -score_samples``.
- :class:`PCAAutoencoderDetector` — a PCA reconstruction-error autoencoder
  (Sakurada & Yairi 2014); anomaly score ``= ||x - reconstruct(x)||^2``. No
  torch / tensorflow.

Both return the shared frozen :class:`AnomalyResult`. Every detector fits on a
TRAIN slice only and scores the disjoint OOS slice. Importing this subpackage
has no side effects (scikit-learn is imported lazily inside the methods).
"""

from __future__ import annotations

from anomaly_detector.detectors.autoencoder import PCAAutoencoderDetector
from anomaly_detector.detectors.iforest import IsolationForestDetector
from anomaly_detector.detectors.result import AnomalyResult

__all__ = [
    "AnomalyResult",
    "IsolationForestDetector",
    "PCAAutoencoderDetector",
]
