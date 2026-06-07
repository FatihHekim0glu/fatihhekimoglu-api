"""Estimators for expected returns and asset covariance matrices."""

from __future__ import annotations

from .covariance import EWMACovariance, LedoitWolfShrinkage, SampleCovariance
from .exceptions import (
    AmbiguousFrequencyError,
    DimensionMismatchError,
    EstimatorConfigError,
    EstimatorError,
    NotFittedError,
)
from .means import (
    CAPMReturns,
    EWMAMean,
    ImpliedReturns,
    JorionBayesStein,
    SampleMean,
)
from .protocols import CovarianceEstimator, MeanEstimator, ReturnsLike

# Backward-compatible aliases consumed by the app layer.
LedoitWolfCov = LedoitWolfShrinkage
SampleCov = SampleCovariance

__all__ = [
    "AmbiguousFrequencyError",
    "CAPMReturns",
    "CovarianceEstimator",
    "DimensionMismatchError",
    "EWMACovariance",
    "EWMAMean",
    "EstimatorConfigError",
    "EstimatorError",
    "ImpliedReturns",
    "JorionBayesStein",
    "LedoitWolfCov",
    "LedoitWolfShrinkage",
    "MeanEstimator",
    "NotFittedError",
    "ReturnsLike",
    "SampleCov",
    "SampleCovariance",
    "SampleMean",
]
