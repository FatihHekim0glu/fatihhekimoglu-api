"""Viterbi MAP decoding — EDA ONLY, never tradable.

The Viterbi algorithm returns the single most likely STATE SEQUENCE given the
ENTIRE observation sequence, ``argmax_{s_1..s_T} p(s_1..s_T | x_1..x_T)``. Because
every label conditions on the whole sample (including the future), the Viterbi
path PEEKS AHEAD: a return realized at ``t+5`` can change the decoded label at
``t``. It is therefore an in-sample / exploratory tool ONLY and must NEVER be used
to generate an out-of-sample regime label or trading signal. The only tradable
decoder is the online forward filter in :mod:`regimehmm.hmm.filter`.

Importing this module has no side effects.
"""

from __future__ import annotations

import numpy as np

from regimehmm._constants import EPS
from regimehmm._exceptions import ValidationError
from regimehmm._typing import FloatArray
from regimehmm.hmm.filter import HMMModel
from regimehmm.hmm.kernel import gaussian_log_density


def viterbi_path(model: HMMModel, observations: FloatArray) -> FloatArray:
    r"""Most likely hidden-state path (Viterbi MAP decoding) — EDA only.

    Runs the log-space Viterbi recursion to return the ``(n_obs,)`` integer state
    sequence maximizing the joint posterior over the whole sequence:

    .. math::

        s^\* = \operatorname*{arg\,max}_{s_1, \dots, s_T}
               \; p(s_1, \dots, s_T \mid x_1, \dots, x_T).

    LOOK-AHEAD WARNING: the Viterbi label at ``t`` depends on observations both
    before AND after ``t``. This is acceptable for in-sample regime visualization
    and parity checking against ``hmmlearn.GaussianHMM.decode`` (validated to an
    exact label match in the parity suite), but it is **never** a valid OOS signal.
    Any tradable label must come from :func:`regimehmm.hmm.filter.filtered_states`.

    Parameters
    ----------
    model:
        A fitted :class:`~regimehmm.hmm.filter.HMMModel`.
    observations:
        The ``(n_obs, n_features)`` scaled feature matrix.

    Returns
    -------
    FloatArray
        The ``(n_obs,)`` Viterbi MAP state path (integer-valued floats).

    Raises
    ------
    ValidationError
        If ``observations`` is malformed or its feature dimension mismatches the
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
    log_emission = gaussian_log_density(
        obs, means, covariances, covariance_type=model.covariance_type
    )
    log_startprob = np.log(np.clip(np.asarray(model.startprob, dtype=np.float64), EPS, None))
    log_transmat = np.log(np.clip(np.asarray(model.transmat, dtype=np.float64), EPS, None))

    # LOOK-AHEAD: this trellis maximizes over the whole sequence (the backtrace at
    # the end can rewrite earlier labels), which is exactly why Viterbi is EDA-only.
    log_delta = np.empty((n_obs, n_states), dtype=np.float64)
    backptr = np.empty((n_obs, n_states), dtype=np.intp)
    log_delta[0] = log_startprob + log_emission[0]
    backptr[0] = 0
    for t in range(1, n_obs):
        # scores[i, j] = log_delta[t-1, i] + log_transmat[i, j].
        scores = log_delta[t - 1][:, np.newaxis] + log_transmat
        backptr[t] = np.argmax(scores, axis=0)
        log_delta[t] = log_emission[t] + np.max(scores, axis=0)

    path = np.empty(n_obs, dtype=np.intp)
    path[-1] = int(np.argmax(log_delta[-1]))
    for t in range(n_obs - 2, -1, -1):
        path[t] = backptr[t + 1, path[t + 1]]
    return path.astype(np.float64)
