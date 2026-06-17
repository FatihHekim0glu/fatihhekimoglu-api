"""Typed exception hierarchy for the wsb_sentiment library.

A single base (:class:`WsbSentimentError`) lets callers catch any
library-raised error with one ``except`` clause, while the specific subclasses
let them distinguish data-shape problems from numerical-degeneracy problems.
Importing this module has no side effects.
"""

from __future__ import annotations

# quantcore-candidate: mirrors hrp-portfolio:src/hrp/_exceptions.py


class WsbSentimentError(Exception):
    """Base class for every exception raised by :mod:`wsb_sentiment`.

    Catching ``WsbSentimentError`` catches all library-specific failures while
    letting unrelated exceptions (e.g. ``KeyboardInterrupt``) propagate.
    """


class ValidationError(WsbSentimentError):
    """Raised when an input fails a shape, dtype, alignment, or domain check.

    Examples: a sentiment panel with mismatched index/columns, a negative
    ``cost_bps``, a ``lag`` smaller than one, or a non-positive aggregation
    ``window``.
    """


class InsufficientDataError(ValidationError):
    """Raised when there are too few observations to estimate the requested quantity.

    For example, fewer in-sample rows than a single walk-forward split requires,
    or an empty rebalance window. It subclasses :class:`ValidationError` because
    "not enough data" is a special case of a failed input precondition.
    """


class SingularCovarianceError(WsbSentimentError):
    """Raised when a covariance/correlation matrix is singular where invertibility is required.

    Reserved for the PCA-of-trial-returns effective-trials estimator and any
    other code path that genuinely requires a full-rank decomposition.
    """
