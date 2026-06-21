"""Shared type aliases for the rl-allocator library.

These aliases document *intent* at function boundaries (a multi-asset return
PANEL vs. a per-bar weight VECTOR vs. a weight PATH over time vs. a look-back
observation tensor) without committing to a single concrete container. Functions
coerce inputs to the canonical pandas/numpy type via
:mod:`rlallocator._validation` at the boundary, so the aliases are deliberately
broad. Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# quantcore-candidate: mirrors rl-trader:src/rltrader/_typing.py, GENERALIZED from
# the single-asset position domain to the MULTI-ASSET weight-simplex domain (return
# panel, weight vector, weight path, multi-asset observations).

#: A MULTI-ASSET return PANEL shaped ``(n_bars, n_assets)`` — per-bar simple
#: returns for each asset. Accepted at the boundary as a 2-D DataFrame, a 2-D
#: ndarray, or a mapping coercible to a wide return frame.
ReturnPanel: TypeAlias = "pd.DataFrame | NDArray[np.float64]"

#: A MULTI-ASSET price PANEL shaped ``(n_bars, n_assets)`` — per-bar price levels;
#: differenced via ``pct_change(fill_method=None)`` (no forward-fill across gaps).
PricePanel: TypeAlias = "pd.DataFrame | NDArray[np.float64]"

#: A single-bar WEIGHT VECTOR shaped ``(n_assets,)`` — the target portfolio weights
#: held over one bar. A valid (long-only) simplex: non-negative and summing to one.
WeightVector: TypeAlias = NDArray[np.float64]

#: A WEIGHT PATH shaped ``(n_bars, n_assets)`` — the target portfolio weight vector
#: held over each bar. The weights set at ``t`` earn the ``t -> t+1`` asset returns
#: (strictly causal); turnover at ``t`` is ``||w_t - w_{t-1}||_1``.
WeightPath: TypeAlias = "pd.DataFrame | NDArray[np.float64]"

#: A look-back OBSERVATION matrix shaped ``(n_steps, obs_dim)`` — the per-bar agent
#: observation (a flattened window of past per-asset returns PLUS the current
#: weights). The NEXT bar's returns NEVER appear here (strict causality).
ObservationMatrix: TypeAlias = NDArray[np.float64]

#: A single-bar OBSERVATION vector shaped ``(obs_dim,)`` — what the policy sees at
#: one decision time ``t`` (data <= ``t`` only).
ObservationVector: TypeAlias = NDArray[np.float64]

#: An ``N x N`` covariance matrix (DataFrame or ndarray) of the train-window asset
#: returns, consumed by the Markowitz / risk-parity baselines.
CovMatrix: TypeAlias = "pd.DataFrame | NDArray[np.float64]"

#: A float64 numpy array of unspecified shape (compute-kernel intermediate).
FloatArray: TypeAlias = NDArray[np.float64]
