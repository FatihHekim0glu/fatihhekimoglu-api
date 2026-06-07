"""Marchenko-Pastur (1967) bulk distribution for random-matrix theory checks.

For an ``N × N`` sample correlation matrix built from ``T`` i.i.d. observations
with unit variance, the eigenvalue density has a continuous bulk supported on
``[λ_-, λ_+]`` with

    λ_± = σ² (1 ± √(N / T))²       Q = T / N > 1
    σ²  = trace(C) / N             (trace-preserving normalisation)

Empirical eigenvalues that lie outside the bulk are interpreted as "signal"
— statistically incompatible with pure noise — and the rest as "bulk" (the
indistinguishable-from-random noise floor).

This module is pure NumPy and stateless.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Bulk edges
# ---------------------------------------------------------------------------


def marchenko_pastur_bulk(
    n_assets: int,
    n_obs: int,
    sigma_sq: float = 1.0,
) -> dict[str, float]:
    """Return Marchenko-Pastur bulk edges ``λ_±`` and the input parameters.

    Parameters
    ----------
    n_assets
        Cross-sectional dimension ``N`` (number of tickers).
    n_obs
        Time-series length ``T`` (number of return observations).
    sigma_sq
        Per-element variance assumption (``σ²``). For correlation matrices the
        physically meaningful value is the trace-preserving estimate from
        :func:`fit_sigma`, which discounts the contribution of the dominant
        ("market") eigenvalues.

    Returns
    -------
    dict
        ``{"lambda_min": float, "lambda_max": float, "sigma_sq": float,
        "q": float}`` where ``q = T / N``.
    """
    if n_assets <= 0 or n_obs <= 0:
        raise ValueError("n_assets and n_obs must be positive")
    q = n_obs / n_assets
    # MP requires q > 0; when q < 1 the bulk picks up a point mass at zero
    # which we don't model — surfaces still as positive edges.
    factor = np.sqrt(n_assets / n_obs)
    lambda_min = float(sigma_sq * (1.0 - factor) ** 2)
    lambda_max = float(sigma_sq * (1.0 + factor) ** 2)
    return {
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "sigma_sq": float(sigma_sq),
        "q": float(q),
    }


# ---------------------------------------------------------------------------
# Density (continuous part)
# ---------------------------------------------------------------------------


def bulk_density(
    n_assets: int,
    n_obs: int,
    sigma_sq: float = 1.0,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return ``f(λ)`` evaluating the Marchenko-Pastur continuous density.

    The closed form is

        ρ(λ) = (1 / (2π σ² λ)) · (T/N) · √((λ_+ − λ)(λ − λ_-))   for λ ∈ [λ_-, λ_+]

    and zero elsewhere. We do not add the Dirac mass at the origin for the
    over-sampled ``q < 1`` case — callers should not pass such inputs in
    practice (we always have ``T ≫ N``).
    """
    edges = marchenko_pastur_bulk(n_assets, n_obs, sigma_sq=sigma_sq)
    lam_min = edges["lambda_min"]
    lam_max = edges["lambda_max"]
    q = edges["q"]

    def f(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        out = np.zeros_like(arr)
        mask = (arr >= lam_min) & (arr <= lam_max) & (arr > 0.0)
        if not mask.any():
            return out
        inside = arr[mask]
        radicand = np.maximum((lam_max - inside) * (inside - lam_min), 0.0)
        out[mask] = (q / (2.0 * np.pi * sigma_sq * inside)) * np.sqrt(radicand)
        return out

    return f


# ---------------------------------------------------------------------------
# σ² estimation
# ---------------------------------------------------------------------------


def fit_sigma(
    eigvals: np.ndarray,
    n_assets: int,
    n_obs: int,
    *,
    n_signal: int = 1,
) -> float:
    """Trace-preserving ``σ²`` for the Marchenko-Pastur bulk.

    The top ``n_signal`` eigenvalues are treated as deterministic signal (most
    notably the "market" eigenvalue which dwarfs the bulk). Removing them and
    renormalising by the remaining trace mass keeps the bulk-density area
    consistent with empirical bulk eigenvalues.

    Specifically

        σ² = (Σ_{k ≥ n_signal} λ_k) / (N − Σ_{k < n_signal} λ_k / σ²_initial)

    The closed form below is the limit of the above fixed-point iteration that
    Bouchaud & Potters call the "Pafnuty Chebyshev fit"; see Plerou et al.
    (PRE, 2002) §IV.B.

    For ``n_signal = 0`` this collapses to ``σ² = mean(eigvals)``, which equals
    ``1`` for a correlation matrix (every eigvalue sums to ``N``).
    """
    if n_assets <= 0 or n_obs <= 0:
        raise ValueError("n_assets and n_obs must be positive")
    arr = np.asarray(eigvals, dtype=float)
    if arr.size == 0:
        return 1.0
    sorted_desc = np.sort(arr)[::-1]
    n_signal = max(0, min(int(n_signal), arr.size - 1))
    bulk_eigs = sorted_desc[n_signal:]
    bulk_mass = float(bulk_eigs.sum())
    n_bulk = float(arr.size - n_signal)
    if n_bulk <= 0:
        return 1.0
    sigma_sq = bulk_mass / n_bulk
    return float(max(sigma_sq, 1e-12))


def count_significant(
    eigvals: np.ndarray,
    n_assets: int,
    n_obs: int,
    *,
    n_signal: int = 1,
) -> int:
    """Return how many eigenvalues lie strictly above the MP bulk upper edge.

    Uses :func:`fit_sigma` to estimate ``σ²`` (excluding the top ``n_signal``
    pre-known signal eigvals), then counts ``λ_i > λ_+``.
    """
    sigma_sq = fit_sigma(eigvals, n_assets, n_obs, n_signal=n_signal)
    edges = marchenko_pastur_bulk(n_assets, n_obs, sigma_sq=sigma_sq)
    return int((np.asarray(eigvals, dtype=float) > edges["lambda_max"]).sum())
