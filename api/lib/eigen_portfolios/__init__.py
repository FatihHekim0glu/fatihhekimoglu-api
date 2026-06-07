"""Eigen Portfolios: PCA-based factor decomposition with Marchenko-Pastur RMT.

Public surface:

  - :func:`eigen_decompose` — PCA on a returns panel, returns sorted eigvals,
    eigvecs, factor returns, and per-component variance shares.
  - :func:`marchenko_pastur_bulk` — RMT bulk-edge ``λ_+/-`` for a given ``(N, T,
    σ²)``.
  - :func:`fit_sigma` — trace-preserving estimate of ``σ²`` so that the bulk
    matches the empirical eigenvalue spread.

Numpy-only, no I/O. The router layer is responsible for loading prices, calling
this module, and serialising the result.
"""

from __future__ import annotations

from .pca import EigenResult, eigen_decompose
from .rmt import bulk_density, fit_sigma, marchenko_pastur_bulk

__all__ = (
    "EigenResult",
    "bulk_density",
    "eigen_decompose",
    "fit_sigma",
    "marchenko_pastur_bulk",
)
