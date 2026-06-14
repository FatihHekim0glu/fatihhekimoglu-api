"""Covariance, RMT-denoising, and expected-return estimators.

All four allocators are fed the *identical* covariance estimator on each window,
isolating the allocation rule as the only treatment. Importing this subpackage
has no side effects.
"""

from __future__ import annotations

from hrp.estimators.covariance import ledoit_wolf_cov, oas_cov, sample_cov
from hrp.estimators.mu import james_stein_mu, sample_mu
from hrp.estimators.rmt import marchenko_pastur_clip

__all__ = [
    "james_stein_mu",
    "ledoit_wolf_cov",
    "marchenko_pastur_clip",
    "oas_cov",
    "sample_cov",
    "sample_mu",
]
