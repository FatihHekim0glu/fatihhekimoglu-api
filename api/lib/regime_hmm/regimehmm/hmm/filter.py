r"""Online forward filter — the ONLY tradable, no-lookahead regime posterior.

This module is the heart of the project's leakage discipline. The online filter
posterior at time ``t``,

.. math::

    p(\text{state}_t = k \mid x_1, \dots, x_t),

conditions on data UP TO AND INCLUDING ``t`` only — never the future. It is the
normalized forward message ``alpha_t`` and is the **only** regime signal that may
drive an out-of-sample trade. By contrast the smoothed (forward-backward)
posterior and the Viterbi path both use the WHOLE sample and therefore PEEK AHEAD;
they are in-sample EDA only and must NEVER be turned into an OOS label or signal.

The frozen :class:`HMMModel` produced by EM lives here (it is what the filter
consumes). Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import logsumexp

from regimehmm._constants import EPS
from regimehmm._exceptions import ValidationError
from regimehmm._typing import FloatArray
from regimehmm.hmm.kernel import CovarianceType, gaussian_log_density


@dataclass(frozen=True, slots=True)
class HMMModel:
    """An immutable, fitted Gaussian HMM.

    The parameter bundle the online filter (and Viterbi/forward-backward) consume.
    Frozen and slotted so a fitted model is hashable-by-identity, cheap, and safe
    to cache; :meth:`to_dict` makes it JSON-serializable across the API boundary.

    Attributes
    ----------
    startprob:
        ``(n_states,)`` initial-state distribution ``pi`` (sums to 1).
    transmat:
        ``(n_states, n_states)`` row-stochastic transition matrix ``A``.
    means:
        ``(n_states, n_features)`` per-state emission means.
    covariances:
        Per-state variances (diag) or covariance matrices (full).
    covariance_type:
        ``"diag"`` or ``"full"``.
    log_likelihood:
        The training-sequence log-likelihood at convergence.
    n_iter:
        The number of EM iterations the winning restart ran.
    converged:
        Whether the winning restart met the EM tolerance before the iteration cap.
    """

    startprob: FloatArray
    transmat: FloatArray
    means: FloatArray
    covariances: FloatArray
    covariance_type: CovarianceType
    log_likelihood: float
    n_iter: int
    converged: bool
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_states(self) -> int:
        """The number of hidden states ``K``."""
        return int(np.asarray(self.means).shape[0])

    @property
    def n_features(self) -> int:
        """The emission feature dimension ``d``."""
        return int(np.asarray(self.means).shape[1])

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the fitted model."""
        return {
            "startprob": [float(v) for v in self.startprob],
            "transmat": [[float(v) for v in row] for row in self.transmat],
            "means": [[float(v) for v in row] for row in self.means],
            "covariances": _covariances_to_list(self.covariances),
            "covariance_type": str(self.covariance_type),
            "log_likelihood": float(self.log_likelihood),
            "n_iter": int(self.n_iter),
            "converged": bool(self.converged),
            "meta": dict(self.meta),
        }


def _covariances_to_list(covariances: FloatArray) -> list[Any]:
    """Recursively coerce a (diag or full) covariance array to nested lists."""
    return [_covariances_to_list(c) if hasattr(c, "__len__") else float(c) for c in covariances]


def online_filter(model: HMMModel, observations: FloatArray) -> FloatArray:
    r"""Causal online forward-filter posteriors (the ONLY tradable signal).

    Returns the ``(n_obs, n_states)`` matrix of filtered posteriors

    .. math::

        f_t(k) = p(\text{state}_t = k \mid x_1, \dots, x_t),

    each row computed by the log-space forward recursion normalized AT EACH STEP.
    Row ``t`` depends on observations ``x_1..x_t`` ONLY — it is invariant to any
    ``x_s`` with ``s > t``.

    NO-LOOKAHEAD GUARANTEE (property-tested): perturbing the returns AFTER time
    ``t`` leaves ``f_t`` byte-identical (future-perturbation invariance /
    prefix-determinism). This is what makes the filtered posterior — and only the
    filtered posterior — a legitimate out-of-sample regime signal. Smoothed and
    Viterbi posteriors fail this test and are non-tradable.

    Parameters
    ----------
    model:
        A fitted :class:`HMMModel` (its parameters were estimated on the TRAIN
        fold only).
    observations:
        The ``(n_obs, n_features)`` scaled feature matrix for the window being
        filtered (the scaler was also fit train-only).

    Returns
    -------
    FloatArray
        The ``(n_obs, n_states)`` causal filtered posterior; each row sums to 1.

    Raises
    ------
    ValidationError
        If ``observations`` is not 2-D or its feature dimension does not match the
        model.
    """
    obs = np.asarray(observations, dtype=np.float64)
    if obs.ndim != 2:
        raise ValidationError(f"observations must be 2-D (n_obs, n_features), got ndim={obs.ndim}.")
    if obs.shape[1] != model.n_features:
        raise ValidationError(
            f"observations feature dim {obs.shape[1]} != model feature dim {model.n_features}."
        )

    n_obs = obs.shape[0]
    n_states = model.n_states

    means = np.asarray(model.means, dtype=np.float64)
    covariances = np.asarray(model.covariances, dtype=np.float64)
    # Per-observation, per-state log emission densities. Row t uses obs[t] ONLY,
    # so the emission contribution at t is independent of any obs[s], s != t.
    log_emission = gaussian_log_density(
        obs, means, covariances, covariance_type=model.covariance_type
    )

    startprob = np.asarray(model.startprob, dtype=np.float64)
    transmat = np.asarray(model.transmat, dtype=np.float64)
    log_startprob = np.log(np.clip(startprob, EPS, None))
    log_transmat = np.log(np.clip(transmat, EPS, None))

    filtered = np.empty((n_obs, n_states), dtype=np.float64)
    # NO-LOOKAHEAD: the forward recursion only ever consumes log_emission[0..t] to
    # form row t. Normalizing per step keeps each row a proper posterior over
    # data <= t; future emissions are never read when computing row t.
    log_alpha_prev = log_startprob + log_emission[0]
    norm = logsumexp(log_alpha_prev)
    filtered[0] = np.exp(log_alpha_prev - norm)
    log_filtered_prev = log_alpha_prev - norm
    for t in range(1, n_obs):
        # log_pred[j] = logsumexp_i(log_filtered_prev[i] + log_transmat[i, j]).
        log_pred = logsumexp(log_filtered_prev[:, np.newaxis] + log_transmat, axis=0)
        log_unnorm = log_pred + log_emission[t]
        norm = logsumexp(log_unnorm)
        log_filtered_prev = log_unnorm - norm
        filtered[t] = np.exp(log_filtered_prev)
    return filtered


def filtered_states(model: HMMModel, observations: FloatArray) -> FloatArray:
    """Causal hard regime labels: the argmax of the online filter posterior.

    A convenience wrapper around :func:`online_filter` returning the
    ``(n_obs,)`` integer MAP state under the FILTERED (no-lookahead) posterior.
    Unlike the Viterbi path this never peeks ahead, so it is tradable.

    Parameters
    ----------
    model:
        A fitted :class:`HMMModel`.
    observations:
        The ``(n_obs, n_features)`` scaled feature matrix.

    Returns
    -------
    FloatArray
        The ``(n_obs,)`` filtered hard-state labels (integer-valued floats).

    Raises
    ------
    ValidationError
        If ``observations`` is malformed.
    """
    posterior = online_filter(model, observations)
    labels = np.argmax(posterior, axis=1).astype(np.float64)
    return np.asarray(labels, dtype=np.float64)
