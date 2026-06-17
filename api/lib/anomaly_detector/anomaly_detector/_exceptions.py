"""Typed exception hierarchy for the anomaly-detector library.

A single base (:class:`AnomalyDetectorError`) lets callers catch any
library-raised error with one ``except`` clause, while the specific subclasses
let them distinguish data-shape problems from numerical-degeneracy problems.
Importing this module has no side effects.
"""

from __future__ import annotations

# quantcore-candidate: mirrors risk-metrics:src/riskmetrics/_exceptions.py


class AnomalyDetectorError(Exception):
    """Base class for every exception raised by :mod:`anomaly_detector`.

    Catching ``AnomalyDetectorError`` catches all library-specific failures
    while letting unrelated exceptions (e.g. ``KeyboardInterrupt``) propagate.
    """


class ValidationError(AnomalyDetectorError):
    """Raised when an input fails a shape, dtype, alignment, or domain check.

    Examples: a non-2D feature matrix, a price/return series with a malformed
    index, a ``contamination`` outside ``(0, 0.5)``, or a rolling ``window``
    smaller than ``2``.
    """


class InsufficientDataError(ValidationError):
    """Raised when there are too few observations to fit the requested detector.

    For example, a train slice with fewer rows than the number of features
    (so the PCA reconstruction is rank-deficient by construction), or an empty
    walk-forward window. It subclasses :class:`ValidationError` because "not
    enough data" is a special case of a failed input precondition.
    """


class NotFittedError(AnomalyDetectorError):
    """Raised when a detector is scored before it has been fitted on a train slice.

    The full-sample-leakage guard requires that a detector's scaler, model, and
    threshold are all fitted on the TRAIN slice only; scoring a detector whose
    state has not been populated is a programming error, surfaced here rather
    than as an opaque ``AttributeError``.
    """
