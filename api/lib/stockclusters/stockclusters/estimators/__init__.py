"""Covariance and RMT-denoising estimators (reused from HRP).

Each cluster-aware allocation arm is fed the *identical* covariance estimator on
each window, isolating the allocation rule as the only treatment. Importing this
subpackage has no side effects.
"""

from __future__ import annotations

from stockclusters.estimators.covariance import ledoit_wolf_cov, oas_cov, sample_cov
from stockclusters.estimators.rmt import marchenko_pastur_clip

__all__ = [
    "ledoit_wolf_cov",
    "marchenko_pastur_clip",
    "oas_cov",
    "sample_cov",
]
