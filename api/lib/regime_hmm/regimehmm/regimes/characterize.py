"""Per-regime characterization — the project's real deliverable.

Given a (canonicalized) decoded regime series and the realized returns, this module
computes the per-regime descriptive statistics that ARE the honest headline of the
project: each regime's mean return, volatility, persistence (self-transition
probability), expected duration, and worst drawdown. Regime CHARACTERIZATION is the
win; the regime-timing overlay (see :mod:`regimehmm.backtest.overlay`) is the
honest null.

Importing this module has no side effects.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from regimehmm._exceptions import ValidationError
from regimehmm._typing import FloatArray
from regimehmm.backtest.stats import max_drawdown
from regimehmm.hmm.filter import HMMModel


@dataclass(frozen=True, slots=True)
class RegimeStats:
    """Immutable descriptive statistics for a single regime.

    Attributes
    ----------
    state:
        The canonical regime label (``0`` = lowest mean return).
    frequency:
        The fraction of observations assigned to this regime.
    mean_return:
        The annualized mean return while in this regime.
    volatility:
        The annualized return volatility while in this regime.
    persistence:
        The self-transition probability ``A[k, k]`` (how "sticky" the regime is).
    expected_duration:
        The expected dwell time ``1 / (1 - persistence)`` in periods.
    max_drawdown:
        The worst peak-to-trough drawdown of the within-regime return stream
        (``<= 0``).
    """

    state: int
    frequency: float
    mean_return: float
    volatility: float
    persistence: float
    expected_duration: float
    max_drawdown: float

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this regime's stats."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RegimeCharacterization:
    """Immutable bundle of per-regime statistics for a fitted model.

    Attributes
    ----------
    n_states:
        The number of regimes.
    stats:
        The per-regime :class:`RegimeStats`, in canonical order.
    """

    n_states: int
    stats: tuple[RegimeStats, ...]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the characterization."""
        return {
            "n_states": int(self.n_states),
            "stats": [s.to_dict() for s in self.stats],
            "meta": dict(self.meta),
        }


def characterize_regimes(
    model: HMMModel,
    states: FloatArray,
    returns: pd.Series,
    *,
    periods_per_year: int = 252,
) -> RegimeCharacterization:
    r"""Compute per-regime descriptive statistics (the headline deliverable).

    For each canonical regime ``k`` this measures, over the observations the
    decoder assigned to ``k``: the occupancy ``frequency``, the annualized
    ``mean_return`` and ``volatility`` of the realized returns, the ``persistence``
    ``A[k, k]`` and the implied ``expected_duration = 1 / (1 - A[k, k])``, and the
    within-regime ``max_drawdown``.

    LABEL DISCIPLINE: ``states`` must already be in CANONICAL order (see
    :mod:`regimehmm.regimes.canonicalize`) so the output is relabeling-invariant
    across folds (property-tested). The decoded ``states`` should come from the
    smoothed/Viterbi decoder for in-sample EDA, or the online filter for an honest
    out-of-sample picture — but characterization itself is descriptive, not a
    tradable signal.

    Parameters
    ----------
    model:
        The fitted (canonicalized) :class:`~regimehmm.hmm.filter.HMMModel`,
        supplying the transition matrix for persistence/duration.
    states:
        The ``(n_obs,)`` canonical decoded regime labels.
    returns:
        The realized per-period return series aligned to ``states``.
    periods_per_year:
        Annualization factor for the mean/volatility (``252`` for daily).

    Returns
    -------
    RegimeCharacterization
        The frozen per-regime statistics bundle in canonical order.

    Raises
    ------
    ValidationError
        If ``states`` and ``returns`` are misaligned or contain out-of-range
        labels.
    """
    transmat = np.asarray(model.transmat, dtype="float64")
    if transmat.ndim != 2 or transmat.shape[0] != transmat.shape[1]:
        raise ValidationError("model.transmat must be a square 2-D matrix.")
    n_states = int(transmat.shape[0])

    if not isinstance(returns, pd.Series):
        raise ValidationError("returns must be a pandas Series.")
    ret_values = returns.to_numpy(dtype="float64")

    states_arr = np.asarray(states)
    if states_arr.ndim != 1:
        raise ValidationError(f"states must be 1-dimensional, got ndim={states_arr.ndim}.")
    states_int = np.round(states_arr).astype(np.intp)
    if not np.allclose(states_int, states_arr):
        raise ValidationError("states must contain integer-valued labels.")
    if states_int.shape[0] != ret_values.shape[0]:
        raise ValidationError(
            f"states ({states_int.shape[0]}) and returns ({ret_values.shape[0]}) "
            "must be the same length."
        )
    if states_int.size and (int(states_int.min()) < 0 or int(states_int.max()) >= n_states):
        raise ValidationError(f"states contains a label outside 0..{n_states - 1}.")

    n_obs = int(states_int.shape[0])
    sqrt_ppy = math.sqrt(periods_per_year)

    stats: list[RegimeStats] = []
    for k in range(n_states):
        mask = states_int == k
        count = int(mask.sum())
        frequency = count / n_obs if n_obs else float("nan")

        if count > 0:
            in_regime = ret_values[mask]
            mean_return = float(in_regime.mean()) * periods_per_year
            # Sample std (ddof=1) needs >= 2 points; a single observation has
            # undefined within-regime volatility/drawdown.
            volatility = float(in_regime.std(ddof=1)) * sqrt_ppy if count > 1 else float("nan")
            mdd = max_drawdown(in_regime) if count > 1 else float("nan")
        else:
            mean_return = float("nan")
            volatility = float("nan")
            mdd = float("nan")

        persistence = float(transmat[k, k])
        gap = 1.0 - persistence
        expected_duration = 1.0 / gap if gap > 0.0 else float("inf")

        stats.append(
            RegimeStats(
                state=k,
                frequency=frequency,
                mean_return=mean_return,
                volatility=volatility,
                persistence=persistence,
                expected_duration=expected_duration,
                max_drawdown=mdd,
            )
        )

    return RegimeCharacterization(
        n_states=n_states,
        stats=tuple(stats),
        meta={"n_obs": n_obs, "periods_per_year": int(periods_per_year)},
    )
