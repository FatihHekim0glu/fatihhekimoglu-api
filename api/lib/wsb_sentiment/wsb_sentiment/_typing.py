"""Shared type aliases for the wsb_sentiment library.

These aliases document *intent* at function boundaries (a returns matrix vs. a
price matrix vs. a sentiment/position vector) without committing to a single
concrete container. Functions coerce inputs to the canonical pandas type via
:mod:`wsb_sentiment._validation` at the boundary, so the aliases are
deliberately broad. Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# quantcore-candidate: mirrors hrp-portfolio:src/hrp/_typing.py

#: A wide panel of asset returns: rows indexed by time, columns by ticker.
#: Accepted at the boundary as a DataFrame, an ndarray, or a mapping coercible
#: to a DataFrame; canonicalized to ``pd.DataFrame`` internally.
ReturnsLike: TypeAlias = "pd.DataFrame | NDArray[np.float64]"

#: A wide panel of asset prices (levels). Same shape conventions as
#: :data:`ReturnsLike`; differenced via ``pct_change(fill_method=None)``.
PricesLike: TypeAlias = "pd.DataFrame | NDArray[np.float64]"

#: A wide panel of per-(ticker, day) daily sentiment scores (rows = day,
#: columns = ticker), e.g. mean compound score or positive-share.
SentimentLike: TypeAlias = "pd.DataFrame | NDArray[np.float64]"

#: A vector or panel of per-ticker positions (long/short/flat) on each date.
PositionsLike: TypeAlias = "pd.Series | pd.DataFrame | NDArray[np.float64]"

#: A square covariance (or correlation) matrix, dimension x dimension.
MatrixLike: TypeAlias = "pd.DataFrame | NDArray[np.float64]"

#: A float64 numpy array of unspecified shape (compute-kernel intermediate).
FloatArray: TypeAlias = NDArray[np.float64]
