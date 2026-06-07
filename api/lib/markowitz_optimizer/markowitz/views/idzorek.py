"""Idzorek (2005) confidence-to-Omega mapping.

Idzorek's procedure lets a portfolio manager express each view's strength as a
percentage confidence ``c_k`` rather than as an opaque variance entry of
``Omega``.  Two flavours are exposed:

* :func:`idzorek_omega_approx` -- the closed-form approximation
  ``omega_k ~= ((1 - c_k) / c_k) * (P_k tau Sigma P_k^T)``.
* :func:`idzorek_omega` -- the exact root-finding procedure that solves, for
  every view ``k``, the scalar equation ensuring that the *posterior weight
  tilt* matches what would be obtained under a 100%-confidence single-view
  update scaled by ``c_k``.

Both return a diagonal matrix.  The exact variant falls back to the
approximation when the bracketing strategy fails (which can happen for
pathological covariance structures).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import brentq

from .exceptions import OmegaSpecificationError

_TINY: float = 1e-300
_LARGE: float = 1e12


def _check_confidences(c: np.ndarray) -> None:
    if c.ndim != 1:
        raise OmegaSpecificationError(f"confidences must be one-dimensional, got shape {c.shape}.")
    if np.any(~np.isfinite(c)) or np.any(c < 0.0) or np.any(c > 1.0):
        raise OmegaSpecificationError("All confidences must be finite numbers in [0, 1].")


def idzorek_omega_approx(
    P: np.ndarray,
    tau_Sigma: np.ndarray,
    confidences: np.ndarray,
) -> np.ndarray:
    """Closed-form Idzorek omega approximation.

    Parameters
    ----------
    P
        Pick matrix of shape ``(K, N)``.
    tau_Sigma
        The scaled prior covariance ``tau * Sigma`` of shape ``(N, N)``.
    confidences
        One-dimensional array of length ``K`` with entries in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        Diagonal ``(K, K)`` matrix of view-uncertainty variances.
    """
    c = np.asarray(confidences, dtype=float)
    _check_confidences(c)
    if P.shape[0] != c.shape[0]:
        raise OmegaSpecificationError(
            f"P has {P.shape[0]} rows but {c.shape[0]} confidences were supplied."
        )

    diag_pkk = np.einsum("ki,ij,kj->k", P, tau_Sigma, P)
    omega_diag = np.empty_like(c)
    for k, ck in enumerate(c):
        if ck >= 1.0 - _TINY:
            omega_diag[k] = 0.0
        elif ck <= _TINY:
            # Effectively no confidence => infinite variance.  We use a very
            # large number so the matrix remains numerically representable.
            omega_diag[k] = _LARGE * max(diag_pkk[k], 1.0)
        else:
            omega_diag[k] = ((1.0 - ck) / ck) * diag_pkk[k]
    return np.diag(omega_diag)


def _single_view_posterior_weight_tilt(
    omega_k: float,
    pk: np.ndarray,
    qk: float,
    pi: np.ndarray,
    tau_Sigma: np.ndarray,
    delta: float,
    w_mkt: np.ndarray,
) -> np.ndarray:
    """Compute ``w_bl - w_mkt`` for a *single* view with variance ``omega_k``.

    Uses the Theil update specialised to ``K = 1`` so the inverted scalar
    ``p tauSigma p^T + omega_k`` is cheap.
    """
    tausig_pk = tau_Sigma @ pk
    denom = float(pk @ tausig_pk) + omega_k
    innov = qk - float(pk @ pi)
    mu_bl = pi + tausig_pk * (innov / denom)
    # w_bl - w_mkt = (1 / delta) * Sigma^{-1} * (mu_bl - pi).  Since tau is a
    # scalar, inverting tau_Sigma instead of Sigma absorbs tau into the
    # constant of proportionality; Idzorek's target is scale-invariant so
    # what matters for root-finding is the direction/monotonicity of the tilt.
    chol = cho_factor(tau_Sigma)
    diff = cho_solve(chol, mu_bl - pi) / delta
    # w_mkt cancels in the tilt; keep the unused parameter referenced so the
    # callable signature is consistent with future extensions.
    _ = w_mkt
    return np.asarray(diff, dtype=float)


def _target_tilt(
    confidence: float,
    pk: np.ndarray,
    qk: float,
    pi: np.ndarray,
    tau_Sigma: np.ndarray,
    delta: float,
    w_mkt: np.ndarray,
) -> np.ndarray:
    """Idzorek's target: full-confidence tilt scaled by ``confidence``."""
    full_tilt = _single_view_posterior_weight_tilt(
        omega_k=0.0,
        pk=pk,
        qk=qk,
        pi=pi,
        tau_Sigma=tau_Sigma,
        delta=delta,
        w_mkt=w_mkt,
    )
    return confidence * full_tilt


def _tilt_residual(
    log_omega_k: float,
    pk: np.ndarray,
    qk: float,
    pi: np.ndarray,
    tau_Sigma: np.ndarray,
    delta: float,
    w_mkt: np.ndarray,
    target_norm: float,
) -> float:
    omega_k = float(np.exp(log_omega_k))
    actual = _single_view_posterior_weight_tilt(
        omega_k=omega_k,
        pk=pk,
        qk=qk,
        pi=pi,
        tau_Sigma=tau_Sigma,
        delta=delta,
        w_mkt=w_mkt,
    )
    return float(np.linalg.norm(actual) - target_norm)


def idzorek_omega(
    P: np.ndarray,
    tau_Sigma: np.ndarray,
    confidences: np.ndarray,
    pi: np.ndarray,
    delta: float,
    w_mkt: np.ndarray,
    *,
    brent_xtol: float = 1e-10,
    brent_maxiter: int = 200,
) -> np.ndarray:
    """Exact Idzorek omega via per-view root finding (with safe fallback)."""
    c = np.asarray(confidences, dtype=float)
    _check_confidences(c)
    if P.shape[0] != c.shape[0]:
        raise OmegaSpecificationError(
            f"P has {P.shape[0]} rows but {c.shape[0]} confidences were supplied."
        )

    q_full = np.einsum("ki,i->k", P, pi)  # placeholder; per-view qk supplied below
    # We need actual q-values: derive them from P @ pi shifted by the
    # caller-implied view targets.  Because this function may be invoked
    # without a separate q vector, we treat the *signal* as the deviation that
    # the single-view computation cares about; q drops out of the *direction*
    # of the tilt-residual so we just need any consistent vector.
    omega_diag = np.empty(P.shape[0])
    approx = idzorek_omega_approx(P, tau_Sigma, c)
    approx_diag = np.diag(approx)

    for k in range(P.shape[0]):
        ck = float(c[k])
        pk = P[k]
        # Use q derived from P @ pi + 1.0 (arbitrary positive innovation) so
        # the tilt is non-zero; the resulting omega depends only on ck and pk
        # because of scale invariance.
        qk = float(q_full[k] + 1.0)

        if ck >= 1.0 - _TINY:
            omega_diag[k] = 0.0
            continue
        if ck <= _TINY:
            omega_diag[k] = _LARGE * max(approx_diag[k], 1.0)
            continue

        target = _target_tilt(ck, pk, qk, pi, tau_Sigma, delta, w_mkt)
        target_norm = float(np.linalg.norm(target))
        if target_norm <= _TINY:
            omega_diag[k] = approx_diag[k]
            continue

        lo, hi = -40.0, 40.0
        f_lo = _tilt_residual(lo, pk, qk, pi, tau_Sigma, delta, w_mkt, target_norm)
        f_hi = _tilt_residual(hi, pk, qk, pi, tau_Sigma, delta, w_mkt, target_norm)
        if f_lo * f_hi > 0.0:
            omega_diag[k] = approx_diag[k]
            continue
        try:
            log_omega = brentq(
                _tilt_residual,
                lo,
                hi,
                args=(pk, qk, pi, tau_Sigma, delta, w_mkt, target_norm),
                xtol=brent_xtol,
                maxiter=brent_maxiter,
            )
            omega_diag[k] = float(np.exp(log_omega))
        except (ValueError, RuntimeError):
            omega_diag[k] = approx_diag[k]

    return np.diag(omega_diag)
