"""State canonicalization for stable, cross-fold regime labels.

A raw HMM fit assigns arbitrary integer indices to its hidden states: the "high
vol" regime might be state 2 in one fold and state 0 in the next, purely because
of restart initialization. That label arbitrariness would scramble any cross-fold
characterization or golden regression. This module imposes a CANONICAL ordering —
states sorted by ascending mean conditional return, tie-broken by ascending
volatility — and returns a permutation that relabels the model and any decoded
state series consistently.

After canonicalization, "regime 0" always means the same kind of regime (lowest
mean return) across folds, so the characterization is relabeling-invariant
(property-tested).

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from regimehmm._exceptions import ValidationError
from regimehmm._typing import FloatArray
from regimehmm.hmm.filter import HMMModel

#: Feature column that carries the asset RETURN by convention (the ordering key).
_RETURN_FEATURE: int = 0


def _state_return_variance(model: HMMModel) -> FloatArray:
    """Per-state variance of the RETURN feature, for either covariance type.

    For ``"diag"`` covariances (shape ``(K, d)``) this is ``cov[:, 0]``; for
    ``"full"`` covariances (shape ``(K, d, d)``) it is the leading diagonal entry
    ``cov[:, 0, 0]``. The result is the variance used for the volatility tie-break.
    """
    cov = np.asarray(model.covariances, dtype="float64")
    if cov.ndim == 2:
        return np.ascontiguousarray(cov[:, _RETURN_FEATURE])
    if cov.ndim == 3:
        return np.ascontiguousarray(cov[:, _RETURN_FEATURE, _RETURN_FEATURE])
    raise ValidationError(f"covariances must be 2-D (diag) or 3-D (full), got ndim={cov.ndim}.")


def canonical_order(model: HMMModel) -> FloatArray:
    r"""Return the permutation that sorts states into canonical order.

    States are ordered by ASCENDING mean conditional (per-state) return; ties on
    the mean are broken by ASCENDING per-state volatility. The result is the index
    array ``perm`` such that ``perm[0]`` is the original index of the new "regime
    0" (lowest mean / lowest vol), and so on.

    Parameters
    ----------
    model:
        A fitted :class:`~regimehmm.hmm.filter.HMMModel`. The ordering key is the
        per-state mean of the RETURN feature (feature column 0 by convention) and,
        as a tie-break, the per-state return volatility.

    Returns
    -------
    FloatArray
        The ``(n_states,)`` integer permutation mapping new label -> original
        index.
    """
    means = np.asarray(model.means, dtype="float64")
    if means.ndim != 2:
        raise ValidationError(f"means must be 2-dimensional, got ndim={means.ndim}.")
    mean_return = means[:, _RETURN_FEATURE]
    vol = np.sqrt(_state_return_variance(model))
    # lexsort uses the LAST key as primary; mean is primary, vol is the tie-break.
    # The sort is stable, so any exact ties surviving both keys keep input order,
    # making the permutation fully deterministic across folds.
    perm = np.lexsort((vol, mean_return))
    return np.ascontiguousarray(perm, dtype="float64")


def canonicalize_model(model: HMMModel) -> HMMModel:
    """Return a relabeled copy of ``model`` in canonical state order.

    Applies :func:`canonical_order` to permute ``startprob``, the rows and columns
    of ``transmat``, ``means``, and ``covariances`` so the returned model's state
    ``k`` is the canonical regime ``k``. The fit's likelihood is unchanged (a pure
    relabeling); the model becomes comparable across folds and serializable to a
    stable schema.

    Parameters
    ----------
    model:
        The fitted model to canonicalize.

    Returns
    -------
    HMMModel
        A canonically-ordered copy of ``model``.
    """
    perm = canonical_order(model).astype(np.intp)

    startprob = np.ascontiguousarray(np.asarray(model.startprob, dtype="float64")[perm])
    transmat_raw = np.asarray(model.transmat, dtype="float64")
    # Permute BOTH axes: row i col j of the new matrix is A[perm[i], perm[j]], so a
    # canonical-label transition keeps its original probability.
    transmat = np.ascontiguousarray(transmat_raw[np.ix_(perm, perm)])
    means = np.ascontiguousarray(np.asarray(model.means, dtype="float64")[perm])
    covariances = np.ascontiguousarray(np.asarray(model.covariances, dtype="float64")[perm])

    return replace(
        model,
        startprob=startprob,
        transmat=transmat,
        means=means,
        covariances=covariances,
    )


def relabel_states(states: FloatArray, order: FloatArray) -> FloatArray:
    """Relabel a decoded state series under a canonical-ordering permutation.

    Given a raw decoded series (from the filter or Viterbi) and the ``order``
    permutation from :func:`canonical_order`, return the series with each raw state
    index replaced by its canonical label, so downstream characterization and
    golden regressions are invariant to the fit's arbitrary internal labelling.

    Parameters
    ----------
    states:
        The ``(n_obs,)`` raw decoded state labels.
    order:
        The ``(n_states,)`` permutation (new label -> original index) from
        :func:`canonical_order`.

    Returns
    -------
    FloatArray
        The ``(n_obs,)`` canonically-relabeled state series.

    Raises
    ------
    ValidationError
        If ``order`` is not a valid permutation or ``states`` contains an
        out-of-range label.
    """
    order_arr = np.asarray(order)
    if order_arr.ndim != 1:
        raise ValidationError(f"order must be 1-dimensional, got ndim={order_arr.ndim}.")
    order_int = np.round(order_arr).astype(np.intp)
    if not np.allclose(order_int, order_arr):
        raise ValidationError("order must contain integer-valued permutation indices.")
    n_states = order_int.shape[0]
    if n_states == 0:
        raise ValidationError("order must be a non-empty permutation.")
    if sorted(order_int.tolist()) != list(range(n_states)):
        raise ValidationError(
            f"order must be a permutation of 0..{n_states - 1}, got {order_int.tolist()}."
        )

    states_arr = np.asarray(states)
    if states_arr.ndim != 1:
        raise ValidationError(f"states must be 1-dimensional, got ndim={states_arr.ndim}.")
    states_int = np.round(states_arr).astype(np.intp)
    if not np.allclose(states_int, states_arr):
        raise ValidationError("states must contain integer-valued labels.")
    if states_int.size and (int(states_int.min()) < 0 or int(states_int.max()) >= n_states):
        raise ValidationError(
            f"states contains a label outside 0..{n_states - 1} for the given order."
        )

    # ``order`` maps new label -> original index; its inverse maps original index
    # (the raw decoded label) -> canonical new label, which is what we apply.
    inverse = np.empty(n_states, dtype=np.intp)
    inverse[order_int] = np.arange(n_states, dtype=np.intp)
    relabeled = inverse[states_int]
    return np.ascontiguousarray(relabeled, dtype="float64")
