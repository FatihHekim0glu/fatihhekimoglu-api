"""Gaussian emission log-density with a covariance floor.

The per-state emission model is a multivariate Gaussian over the (scaled) feature
vector at each time step. This module computes the LOG emission density
``log p(x_t | state=k)`` for every observation and state, supporting both
``"diag"`` (independent features) and ``"full"`` covariance parameterizations. A
strictly positive covariance FLOOR is applied so a (near-)degenerate state — one
that collapses onto a handful of identical points during EM — cannot drive the
log-density to ``+inf`` or produce a singular covariance.

The kernel is pure numpy/scipy: no global RNG, no hmmlearn. ``hmmlearn`` is a
dev-only parity oracle and is NEVER imported here.

Importing this module has no side effects.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from regimehmm._exceptions import ValidationError
from regimehmm._typing import FloatArray

#: Supported covariance parameterizations for the Gaussian emissions.
CovarianceType = Literal["diag", "full"]

_LOG_2PI = math.log(2.0 * math.pi)


def floor_covariance(
    covariances: FloatArray,
    *,
    covariance_type: CovarianceType = "diag",
    floor: float = 1e-6,
) -> FloatArray:
    r"""Apply a strictly positive variance floor to per-state covariances.

    For ``covariance_type="diag"`` the input is a ``(n_states, n_features)`` array
    of per-feature variances; each entry is raised to at least ``floor``. For
    ``covariance_type="full"`` the input is a ``(n_states, n_features, n_features)``
    stack; the floor is added to the diagonal (equivalently, eigenvalues are
    clipped up to ``floor``) so each matrix is symmetric positive-definite.

    DEGENERACY GUARD: this is the single chokepoint that keeps EM from emitting a
    singular or zero-variance state. A state that collapses onto identical points
    would otherwise send the Gaussian log-density to ``+inf``; the floor makes the
    emission model well-defined for every state on every iteration.

    Parameters
    ----------
    covariances:
        Per-state variances (diag) or covariance matrices (full).
    covariance_type:
        ``"diag"`` or ``"full"``.
    floor:
        The strictly positive minimum variance / minimum eigenvalue.

    Returns
    -------
    FloatArray
        The floored covariances, same shape as the input.

    Raises
    ------
    ValidationError
        If ``floor <= 0`` or ``covariances`` has the wrong rank for
        ``covariance_type``.
    """
    if floor <= 0.0:
        raise ValidationError(f"floor must be strictly positive, got {floor}.")
    cov = np.asarray(covariances, dtype=np.float64)

    if covariance_type == "diag":
        if cov.ndim != 2:
            raise ValidationError(
                f"diag covariances must be 2-D (n_states, n_features), got ndim={cov.ndim}."
            )
        return np.maximum(cov, floor)

    if covariance_type == "full":
        if cov.ndim != 3 or cov.shape[1] != cov.shape[2]:
            raise ValidationError(
                "full covariances must be 3-D (n_states, n_features, n_features), "
                f"got shape={cov.shape}."
            )
        n_states = cov.shape[0]
        floored = np.empty_like(cov)
        for k in range(n_states):
            # Symmetrize, then clip eigenvalues up to ``floor`` so the result is
            # symmetric positive-definite even if the input collapsed to singular.
            sym = 0.5 * (cov[k] + cov[k].T)
            eigvals, eigvecs = np.linalg.eigh(sym)
            clipped = np.maximum(eigvals, floor)
            rebuilt = (eigvecs * clipped) @ eigvecs.T
            # Re-symmetrize to kill round-off asymmetry; guarantees SPD downstream.
            floored[k] = 0.5 * (rebuilt + rebuilt.T)
        return floored

    raise ValidationError(f"unsupported covariance_type: {covariance_type!r}.")


def gaussian_log_density(
    observations: FloatArray,
    means: FloatArray,
    covariances: FloatArray,
    *,
    covariance_type: CovarianceType = "diag",
    floor: float = 1e-6,
) -> FloatArray:
    r"""Per-state Gaussian log emission density for every observation.

    Returns the ``(n_obs, n_states)`` matrix

    .. math::

        \log p(x_t \mid \text{state}=k) =
            -\tfrac{1}{2}\big[d\log(2\pi) + \log\det\Sigma_k
            + (x_t - \mu_k)^\top \Sigma_k^{-1} (x_t - \mu_k)\big],

    where :math:`d` is the feature dimension. The covariance is floored via
    :func:`floor_covariance` before use, so the log-determinant and quadratic form
    are always finite. For ``covariance_type="diag"`` the quadratic form and
    log-determinant reduce to per-feature sums (no matrix inverse needed).

    PARITY REQUIREMENT: validated against ``hmmlearn``'s ``_compute_log_likelihood``
    to ``1e-6`` on seeded 2/3-state data in the parity suite.

    Parameters
    ----------
    observations:
        The ``(n_obs, n_features)`` (already scaled) feature matrix.
    means:
        The ``(n_states, n_features)`` per-state mean vectors.
    covariances:
        Per-state variances (diag) or covariance matrices (full).
    covariance_type:
        ``"diag"`` or ``"full"``.
    floor:
        Variance floor passed through to :func:`floor_covariance`.

    Returns
    -------
    FloatArray
        The ``(n_obs, n_states)`` log-density matrix.

    Raises
    ------
    ValidationError
        If shapes are inconsistent across the three arrays.
    """
    obs = np.asarray(observations, dtype=np.float64)
    mu = np.asarray(means, dtype=np.float64)
    if obs.ndim != 2:
        raise ValidationError(f"observations must be 2-D (n_obs, n_features), got ndim={obs.ndim}.")
    if mu.ndim != 2:
        raise ValidationError(f"means must be 2-D (n_states, n_features), got ndim={mu.ndim}.")
    n_obs, n_features = obs.shape
    n_states, mu_features = mu.shape
    if mu_features != n_features:
        raise ValidationError(
            f"means feature dim {mu_features} != observations feature dim {n_features}."
        )

    cov = floor_covariance(covariances, covariance_type=covariance_type, floor=floor)

    if covariance_type == "diag":
        if cov.shape != (n_states, n_features):
            raise ValidationError(
                f"diag covariances shape {cov.shape} != (n_states, n_features) "
                f"({n_states}, {n_features})."
            )
        # log b_k(x_t) = -0.5 [ d log(2pi) + sum_j log var_kj
        #                       + sum_j (x_tj - mu_kj)^2 / var_kj ].
        log_det = np.sum(np.log(cov), axis=1)  # (n_states,)
        inv_var = 1.0 / cov  # (n_states, n_features)
        # (n_obs, 1, n_features) - (1, n_states, n_features) -> (n_obs, n_states, n_features)
        centered = obs[:, np.newaxis, :] - mu[np.newaxis, :, :]
        quad = np.sum((centered**2) * inv_var[np.newaxis, :, :], axis=2)  # (n_obs, n_states)
        log_density = -0.5 * (n_features * _LOG_2PI + log_det[np.newaxis, :] + quad)
        return np.asarray(log_density, dtype=np.float64)

    if covariance_type == "full":
        if cov.shape != (n_states, n_features, n_features):
            raise ValidationError(
                f"full covariances shape {cov.shape} != "
                f"(n_states, n_features, n_features) ({n_states}, {n_features}, {n_features})."
            )
        out = np.empty((n_obs, n_states), dtype=np.float64)
        for k in range(n_states):
            # Cholesky-based log-det and solve: stable + matches hmmlearn's full path.
            chol = np.linalg.cholesky(cov[k])
            log_det = 2.0 * np.sum(np.log(np.diagonal(chol)))
            centered = obs - mu[k]  # (n_obs, n_features)
            solved = np.linalg.solve(chol, centered.T)  # (n_features, n_obs)
            quad = np.sum(solved**2, axis=0)  # (n_obs,)
            out[:, k] = -0.5 * (n_features * _LOG_2PI + log_det + quad)
        return out

    raise ValidationError(f"unsupported covariance_type: {covariance_type!r}.")
