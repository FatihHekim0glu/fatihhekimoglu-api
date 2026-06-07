"""Exception hierarchy for the estimators subpackage."""

from __future__ import annotations


class EstimatorError(Exception):
    """Base class for all estimator-related errors."""


class NotFittedError(EstimatorError):
    """Raised when accessing fitted attributes before calling ``.fit``."""


class AmbiguousFrequencyError(EstimatorError):
    """Raised when annualization is requested but periods-per-year is undetermined."""


class EstimatorConfigError(EstimatorError):
    """Raised when an estimator is constructed with an invalid configuration."""


class DimensionMismatchError(EstimatorError):
    """Raised when input arrays have incompatible shapes."""
