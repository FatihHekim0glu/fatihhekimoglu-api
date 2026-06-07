"""Structural typing protocols for estimator interfaces."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

ReturnsLike = pd.DataFrame | np.ndarray


@runtime_checkable
class MeanEstimator(Protocol):
    """Protocol for an estimator that produces an expected-returns vector."""

    mean_: np.ndarray
    n_features_in_: int
    n_samples_: int
    feature_names_in_: np.ndarray

    def fit(self, returns: ReturnsLike, /) -> MeanEstimator: ...


@runtime_checkable
class CovarianceEstimator(Protocol):
    """Protocol for an estimator that produces a covariance matrix."""

    covariance_: np.ndarray
    n_features_in_: int
    n_samples_: int
    feature_names_in_: np.ndarray

    def fit(self, returns: ReturnsLike, /) -> CovarianceEstimator: ...
