"""Shared type aliases for the volforecast library.

These aliases document *intent* at function boundaries (a return series vs. an
OHLC price frame vs. a realized-volatility series) without committing to a single
concrete container. Functions coerce inputs to the canonical pandas type via
:mod:`volforecast._validation` at the boundary, so the aliases are deliberately
broad. Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# quantcore-candidate: mirrors factorlab:src/factorlab/_typing.py

#: A 1-D series of returns indexed by time. Accepted at the boundary as a Series,
#: a 1-D ndarray, or any sequence coercible to a Series; canonicalized to
#: ``pd.Series`` internally.
ReturnsLike: TypeAlias = "pd.Series | NDArray[np.float64]"

#: A 1-D series of price levels (e.g. close prices). Same conventions as
#: :data:`ReturnsLike`; differenced via ``pct_change(fill_method=None)``.
PricesLike: TypeAlias = "pd.Series | NDArray[np.float64]"

#: A wide OHLC frame: rows indexed by time, columns ``{open, high, low, close}``.
#: The estimators in :mod:`volforecast.realized` consume this for range-based RV.
OHLCLike: TypeAlias = "pd.DataFrame"

#: A 1-D series of realized-volatility values indexed by time (the forecast
#: target and the feature inputs to HAR / XGBoost).
RVLike: TypeAlias = "pd.Series | NDArray[np.float64]"

#: A float64 numpy array of unspecified shape (compute-kernel intermediate).
FloatArray: TypeAlias = NDArray[np.float64]
