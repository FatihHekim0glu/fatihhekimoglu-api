"""Shared type aliases for the anomaly-detector library.

These aliases document *intent* at function boundaries (a price series vs. a
return series vs. a per-day feature matrix) without committing to a single
concrete container. Functions coerce inputs to the canonical pandas type via
:mod:`anomaly_detector._validation` at the boundary, so the aliases are
deliberately broad. Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# quantcore-candidate: mirrors factorlab:src/factorlab/_typing.py

#: A one-column price level series (or a single-asset price panel): rows indexed
#: by time. Accepted at the boundary as a Series/DataFrame or a 1-D ndarray;
#: differenced via ``pct_change(fill_method=None)``.
PricesLike: TypeAlias = "pd.Series | pd.DataFrame | NDArray[np.float64]"

#: A series of simple returns indexed by time. Same shape conventions as
#: :data:`PricesLike`; canonicalized to ``pd.Series`` internally.
ReturnsLike: TypeAlias = "pd.Series | NDArray[np.float64]"

#: A per-day feature matrix: rows indexed by time, columns by feature name
#: (log-return, realized vol, z-score, range/ATR, volume z, autocorrelation).
#: Canonicalized to ``pd.DataFrame`` internally.
FeatureMatrixLike: TypeAlias = "pd.DataFrame | NDArray[np.float64]"

#: A 1-D series of per-day anomaly scores (higher = more anomalous), indexed by
#: time. Returned by every detector's ``score`` method.
ScoresLike: TypeAlias = "pd.Series | NDArray[np.float64]"

#: A float64 numpy array of unspecified shape (compute-kernel intermediate).
FloatArray: TypeAlias = NDArray[np.float64]
