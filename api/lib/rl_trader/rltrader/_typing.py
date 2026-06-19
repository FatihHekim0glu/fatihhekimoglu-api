"""Shared type aliases for the rl-trader library.

These aliases document *intent* at function boundaries (a single-asset price
path vs. a per-bar return series vs. a look-back observation matrix vs. an
action / target-position sequence) without committing to a single concrete
container. Functions coerce inputs to the canonical pandas/numpy type via
:mod:`rltrader._validation` at the boundary, so the aliases are deliberately
broad. Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# quantcore-candidate: mirrors gnn-stocks:src/gnnstocks/_typing.py, reframed for the
# single-asset RL trading-env domain (price path, returns, observations, actions).

#: A single-asset PRICE path (levels). Accepted at the boundary as a 1-D Series,
#: a 1-D ndarray, or any sequence coercible to a 1-D Series; differenced via
#: ``pct_change(fill_method=None)``.
PriceSeries: TypeAlias = "pd.Series | NDArray[np.float64]"

#: A single-asset per-bar RETURN series (``r_t = p_t / p_{t-1} - 1``). Same shape
#: conventions as :data:`PriceSeries`.
ReturnSeries: TypeAlias = "pd.Series | NDArray[np.float64]"

#: A look-back OBSERVATION matrix shaped ``(n_steps, obs_dim)`` — the per-bar
#: agent observation (a window of past returns/features PLUS the current
#: position). The NEXT bar's return NEVER appears here (strict causality).
ObservationMatrix: TypeAlias = NDArray[np.float64]

#: A single-bar OBSERVATION vector shaped ``(obs_dim,)`` — what the policy sees at
#: one decision time ``t`` (data <= ``t`` only).
ObservationVector: TypeAlias = NDArray[np.float64]

#: A 1-D ACTION / target-position sequence indexed by bar — the position the
#: agent holds over each bar, in ``{-1, 0, +1}`` (short / flat / long) or a
#: continuous target weight in ``[-1, 1]``. The position set at ``t`` earns the
#: ``t -> t+1`` return.
ActionSequence: TypeAlias = "pd.Series | NDArray[np.float64]"

#: A float64 numpy array of unspecified shape (compute-kernel intermediate).
FloatArray: TypeAlias = NDArray[np.float64]
