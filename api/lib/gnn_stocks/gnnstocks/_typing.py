"""Shared type aliases for the gnn-stocks library.

These aliases document *intent* at function boundaries (a wide returns panel vs.
a per-node feature matrix vs. a node-by-node adjacency vs. a cross-sectional
label vector) without committing to a single concrete container. Functions
coerce inputs to the canonical pandas/numpy type via
:mod:`gnnstocks._validation` at the boundary, so the aliases are deliberately
broad. Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# quantcore-candidate: mirrors mvts-forecast:src/mvtsforecast/_typing.py

#: A wide panel of asset RETURNS: rows indexed by time, columns by node (asset).
#: Accepted at the boundary as a DataFrame, an ndarray, or a mapping coercible to
#: a DataFrame; canonicalized to ``pd.DataFrame`` internally.
ReturnsLike: TypeAlias = "pd.DataFrame | NDArray[np.float64]"

#: A wide panel of asset PRICES (levels). Same shape conventions as
#: :data:`ReturnsLike`; differenced via ``pct_change(fill_method=None)``.
PricesLike: TypeAlias = "pd.DataFrame | NDArray[np.float64]"

#: A per-node FEATURE matrix shaped ``(n_nodes, n_features)`` — the cross-section
#: of node-level predictors (return, vol, momentum, ...) at one rebalance date.
#: The forward-return label NEVER appears here (anti-leakage).
FeatureMatrix: TypeAlias = NDArray[np.float64]

#: A square node-by-node ADJACENCY (or normalized-adjacency) matrix shaped
#: ``(n_nodes, n_nodes)``. Symmetric for the correlation-kNN and sector builders.
AdjacencyMatrix: TypeAlias = NDArray[np.float64]

#: A 1-D cross-sectional vector indexed by node — next-period forward returns
#: (the prediction target) or per-node model scores (the ranking signal).
CrossSection: TypeAlias = "pd.Series | NDArray[np.float64]"

#: A float64 numpy array of unspecified shape (compute-kernel intermediate).
FloatArray: TypeAlias = NDArray[np.float64]
