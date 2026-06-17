"""Baum-Welch EM with multiple seeded restarts.

Fits a Gaussian HMM by the Expectation-Maximization (Baum-Welch) algorithm: the
E-step is the log-space forward-backward pass (:mod:`regimehmm.hmm.forward_backward`),
the M-step re-estimates ``pi``, ``A``, the per-state means, and the (floored)
covariances from the smoothed sufficient statistics. To dodge EM's local optima we
run several restarts from different seeded initializations and keep the highest
log-likelihood fit. Restart seeds come from reproducible PCG64 substreams
(:func:`regimehmm._rng.spawn_substreams`), so a fixed master seed reproduces the
fit exactly.

CORRECTNESS INVARIANT: within a single restart the per-iteration log-likelihood is
non-decreasing (Baum-Welch is monotone). This is asserted at fit time and pinned by
a unit test.

Importing this module has no side effects.
"""

from __future__ import annotations

import numpy as np

from regimehmm._constants import EPS
from regimehmm._exceptions import InsufficientDataError, ValidationError
from regimehmm._rng import spawn_substreams
from regimehmm._typing import FloatArray
from regimehmm.hmm.filter import HMMModel
from regimehmm.hmm.forward_backward import forward_backward
from regimehmm.hmm.kernel import CovarianceType, floor_covariance, gaussian_log_density

#: Tolerance (in nats) for the per-iteration monotonicity assertion. Baum-Welch is
#: monotone in exact arithmetic; we allow a tiny negative drift from floating-point
#: round-off before treating a decrease as a genuine invariant violation.
_MONOTONIC_TOL = 1e-6


def fit_hmm(
    observations: FloatArray,
    n_states: int,
    *,
    covariance_type: CovarianceType = "diag",
    n_restarts: int = 8,
    max_iter: int = 200,
    tol: float = 1e-4,
    covariance_floor: float = 1e-6,
    seed: int = 7,
) -> HMMModel:
    r"""Fit a Gaussian HMM by Baum-Welch EM with seeded restarts.

    Runs ``n_restarts`` independent EM optimizations from distinct seeded
    initializations and returns the :class:`~regimehmm.hmm.filter.HMMModel` with
    the highest training log-likelihood. Each restart alternates:

    * **E-step** — :func:`regimehmm.hmm.forward_backward.forward_backward` to get
      the smoothed posteriors ``gamma`` and pair-marginals ``xi``;
    * **M-step** — re-estimate ``pi = gamma_1``, ``A`` from summed ``xi``, the
      per-state means as ``gamma``-weighted observation averages, and the
      covariances as ``gamma``-weighted second moments, FLOORED via
      :func:`regimehmm.hmm.kernel.floor_covariance`.

    A restart stops when the log-likelihood gain falls below ``tol`` or after
    ``max_iter`` iterations.

    LEAKAGE NOTE: ``fit_hmm`` must be called on the TRAIN fold only. The fitted
    model is then consumed by the ONLINE filter (:mod:`regimehmm.hmm.filter`) to
    label the out-of-sample window — never by the smoothed/Viterbi decoders.

    REPRODUCIBILITY: restart initializations draw from
    :func:`regimehmm._rng.spawn_substreams(seed, n_restarts)`, so a fixed ``seed``
    reproduces the fit byte-for-byte.

    MONOTONICITY INVARIANT: within each restart the per-iteration log-likelihood is
    non-decreasing; a violation beyond floating tolerance raises (unit-tested).

    Parameters
    ----------
    observations:
        The ``(n_obs, n_features)`` scaled feature matrix (scaler fit train-only).
    n_states:
        The number of hidden regimes ``K`` (``>= 1``).
    covariance_type:
        ``"diag"`` or ``"full"`` emission covariance.
    n_restarts:
        Number of seeded EM restarts (best LL wins).
    max_iter:
        Maximum EM iterations per restart.
    tol:
        Convergence tolerance on the log-likelihood increment.
    covariance_floor:
        Strictly positive variance floor passed to the M-step.
    seed:
        Master seed for the restart substreams.

    Returns
    -------
    HMMModel
        The best (highest training log-likelihood) fitted model.

    Raises
    ------
    ValidationError
        If ``n_states < 1``, ``n_restarts < 1``, ``max_iter < 1``, ``tol <= 0``,
        ``covariance_floor <= 0``, or ``observations`` is not 2-D.
    InsufficientDataError
        If there are fewer observations than states.
    """
    obs = np.asarray(observations, dtype=np.float64)
    if obs.ndim != 2:
        raise ValidationError(f"observations must be 2-D (n_obs, n_features), got ndim={obs.ndim}.")
    if n_states < 1:
        raise ValidationError(f"n_states must be >= 1, got {n_states}.")
    if n_restarts < 1:
        raise ValidationError(f"n_restarts must be >= 1, got {n_restarts}.")
    if max_iter < 1:
        raise ValidationError(f"max_iter must be >= 1, got {max_iter}.")
    if tol <= 0.0:
        raise ValidationError(f"tol must be strictly positive, got {tol}.")
    if covariance_floor <= 0.0:
        raise ValidationError(
            f"covariance_floor must be strictly positive, got {covariance_floor}."
        )
    n_obs = obs.shape[0]
    if n_obs < n_states:
        raise InsufficientDataError(f"need at least n_states={n_states} observations, got {n_obs}.")

    streams = spawn_substreams(seed, n_restarts)
    best_model: HMMModel | None = None
    best_ll = -np.inf
    for restart, gen in enumerate(streams):
        startprob, transmat, means, covariances = _init_params(
            obs, n_states, covariance_type=covariance_type, rng=gen
        )
        model, trace = em_single_run(
            obs,
            startprob,
            transmat,
            means,
            covariances,
            covariance_type=covariance_type,
            max_iter=max_iter,
            tol=tol,
            covariance_floor=covariance_floor,
        )
        if trace[-1] > best_ll:
            best_ll = trace[-1]
            best_model = HMMModel(
                startprob=model.startprob,
                transmat=model.transmat,
                means=model.means,
                covariances=model.covariances,
                covariance_type=model.covariance_type,
                log_likelihood=model.log_likelihood,
                n_iter=model.n_iter,
                converged=model.converged,
                meta={**model.meta, "best_restart": restart, "n_restarts": n_restarts},
            )

    assert best_model is not None  # n_restarts >= 1 guarantees at least one fit
    return best_model


def em_single_run(
    observations: FloatArray,
    startprob: FloatArray,
    transmat: FloatArray,
    means: FloatArray,
    covariances: FloatArray,
    *,
    covariance_type: CovarianceType = "diag",
    max_iter: int = 200,
    tol: float = 1e-4,
    covariance_floor: float = 1e-6,
) -> tuple[HMMModel, list[float]]:
    r"""Run a SINGLE EM optimization from a given initialization.

    The deterministic inner loop used by :func:`fit_hmm` (one restart). Returns the
    converged model together with the per-iteration log-likelihood trace, so the
    monotonicity invariant can be asserted by the caller and inspected in tests.

    Parameters
    ----------
    observations:
        The ``(n_obs, n_features)`` scaled feature matrix.
    startprob, transmat, means, covariances:
        The initial parameter bundle for this restart.
    covariance_type:
        ``"diag"`` or ``"full"``.
    max_iter:
        Maximum EM iterations.
    tol:
        Convergence tolerance on the log-likelihood increment.
    covariance_floor:
        Strictly positive variance floor for the M-step.

    Returns
    -------
    tuple[HMMModel, list[float]]
        The fitted model and the (non-decreasing) per-iteration log-likelihood
        trace.

    Raises
    ------
    ValidationError
        If shapes are inconsistent or scalar parameters are out of range.
    """
    obs = np.asarray(observations, dtype=np.float64)
    if obs.ndim != 2:
        raise ValidationError(f"observations must be 2-D (n_obs, n_features), got ndim={obs.ndim}.")
    if max_iter < 1:
        raise ValidationError(f"max_iter must be >= 1, got {max_iter}.")
    if tol <= 0.0:
        raise ValidationError(f"tol must be strictly positive, got {tol}.")
    if covariance_floor <= 0.0:
        raise ValidationError(
            f"covariance_floor must be strictly positive, got {covariance_floor}."
        )

    pi = np.asarray(startprob, dtype=np.float64).copy()
    a = np.asarray(transmat, dtype=np.float64).copy()
    mu = np.asarray(means, dtype=np.float64).copy()
    n_states = mu.shape[0]
    if pi.shape != (n_states,):
        raise ValidationError(f"startprob shape {pi.shape} != (n_states,) ({n_states},).")
    if a.shape != (n_states, n_states):
        raise ValidationError(
            f"transmat shape {a.shape} != (n_states, n_states) ({n_states}, {n_states})."
        )
    cov = floor_covariance(
        np.asarray(covariances, dtype=np.float64),
        covariance_type=covariance_type,
        floor=covariance_floor,
    )

    trace: list[float] = []
    n_iter = 0
    converged = False
    for iteration in range(1, max_iter + 1):
        n_iter = iteration
        log_pi = np.log(np.clip(pi, EPS, None))
        log_a = np.log(np.clip(a, EPS, None))
        log_b = gaussian_log_density(obs, mu, cov, covariance_type=covariance_type)

        fb = forward_backward(log_pi, log_a, log_b)
        ll = fb.log_likelihood
        if trace:
            # MONOTONICITY INVARIANT: Baum-Welch never decreases the LL (up to
            # round-off). A genuine decrease signals a bug, not a local optimum.
            if ll < trace[-1] - _MONOTONIC_TOL:
                raise ValidationError(
                    "EM log-likelihood decreased "
                    f"({trace[-1]:.10f} -> {ll:.10f}); monotonicity invariant violated."
                )
            if ll - trace[-1] < tol:
                trace.append(ll)
                converged = True
                break
        trace.append(ll)

        pi, a, mu, cov = _m_step(
            obs,
            fb.gamma,
            fb.xi,
            covariance_type=covariance_type,
            covariance_floor=covariance_floor,
        )

    model = HMMModel(
        startprob=pi,
        transmat=a,
        means=mu,
        covariances=cov,
        covariance_type=covariance_type,
        log_likelihood=trace[-1],
        n_iter=n_iter,
        converged=converged,
        meta={"covariance_floor": covariance_floor},
    )
    return model, trace


def _init_params(
    observations: FloatArray,
    n_states: int,
    *,
    covariance_type: CovarianceType,
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Seeded initialization: uniform pi/A, random-data-point means, global covariance.

    Means are seeded by sampling ``n_states`` distinct observation rows (a cheap,
    robust, hmmlearn-like init). The shared starting covariance is the global
    feature covariance, which keeps every restart numerically well-posed.
    """
    obs = np.asarray(observations, dtype=np.float64)
    n_obs, n_features = obs.shape

    startprob = np.full(n_states, 1.0 / n_states, dtype=np.float64)
    transmat = np.full((n_states, n_states), 1.0 / n_states, dtype=np.float64)

    idx = rng.choice(n_obs, size=n_states, replace=n_obs < n_states)
    means = obs[idx].copy()
    # Jitter to break exact ties when two sampled rows coincide.
    means = means + rng.normal(scale=1e-4, size=means.shape)

    global_var = np.var(obs, axis=0)
    global_var = np.maximum(global_var, EPS)
    if covariance_type == "diag":
        covariances = np.tile(global_var, (n_states, 1))
    else:
        base = np.cov(obs, rowvar=False)
        base = np.atleast_2d(base)
        if base.shape != (n_features, n_features):
            base = np.diag(global_var)
        covariances = np.tile(base, (n_states, 1, 1))
    return startprob, transmat, means, covariances


def _m_step(
    observations: FloatArray,
    gamma: FloatArray,
    xi: FloatArray,
    *,
    covariance_type: CovarianceType,
    covariance_floor: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Baum-Welch M-step: re-estimate ``(pi, A, means, covariances)`` from gamma/xi."""
    obs = np.asarray(observations, dtype=np.float64)
    n_features = obs.shape[1]
    n_states = gamma.shape[1]

    # pi = gamma_1.
    startprob = gamma[0].copy()
    startprob = startprob / np.clip(startprob.sum(), EPS, None)

    # A_ij = sum_t xi_t(i, j) / sum_t gamma_t(i)  (gamma summed over t = 0..T-2).
    if xi.shape[0] > 0:
        xi_sum = xi.sum(axis=0)  # (n_states, n_states)
        denom = xi_sum.sum(axis=1, keepdims=True)
        transmat = xi_sum / np.clip(denom, EPS, None)
    else:
        transmat = np.full((n_states, n_states), 1.0 / n_states, dtype=np.float64)

    # Weighted means: mu_k = sum_t gamma_t(k) x_t / sum_t gamma_t(k).
    state_weight = gamma.sum(axis=0)  # (n_states,)
    safe_weight = np.clip(state_weight, EPS, None)
    means = (gamma.T @ obs) / safe_weight[:, np.newaxis]

    if covariance_type == "diag":
        covariances = np.empty((n_states, n_features), dtype=np.float64)
        for k in range(n_states):
            centered = obs - means[k]  # (n_obs, n_features)
            covariances[k] = (gamma[:, k] @ (centered**2)) / safe_weight[k]
    else:
        covariances = np.empty((n_states, n_features, n_features), dtype=np.float64)
        for k in range(n_states):
            centered = obs - means[k]
            weighted = centered * gamma[:, k][:, np.newaxis]
            covariances[k] = (weighted.T @ centered) / safe_weight[k]
    covariances = floor_covariance(
        covariances, covariance_type=covariance_type, floor=covariance_floor
    )
    return startprob, transmat, means, covariances
