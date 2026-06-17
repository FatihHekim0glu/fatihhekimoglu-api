"""Shared type aliases for the nn-vs-bs library.

These aliases document *intent* at function boundaries (an option-parameter
panel vs. a price vector vs. a feature matrix) without committing to a single
concrete container. Functions coerce inputs to the canonical pandas/numpy type
via :mod:`nnvsbs._validation` at the boundary, so the aliases are deliberately
broad. Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# quantcore-candidate: mirrors lstm-forecast:src/lstmforecast/_typing.py

#: A float64 numpy array of unspecified shape (compute-kernel intermediate).
FloatArray: TypeAlias = NDArray[np.float64]

#: A 1-D array of option parameters or scalar quantities (S, K, T, prices, ...).
#: Accepted at the boundary as a ``pd.Series`` or a 1-D ``np.ndarray``.
VectorLike: TypeAlias = "pd.Series | NDArray[np.float64]"

#: A panel of option contracts: one row per quote, columns are the contract
#: parameters ``(S, K, T, r, q, sigma, ...)`` plus (on real chains) a market
#: price. Accepted as a ``pd.DataFrame`` or a 2-D ndarray; canonicalized to a
#: ``pd.DataFrame`` internally.
ChainLike: TypeAlias = "pd.DataFrame | NDArray[np.float64]"

#: A 2-D feature matrix: rows indexed by quote, columns by engineered feature
#: (moneyness, T, r, q, realized vol — NEVER implied vol). Accepted as a
#: DataFrame or a 2-D ndarray; canonicalized to ``pd.DataFrame``.
FeatureFrameLike: TypeAlias = "pd.DataFrame | NDArray[np.float64]"
