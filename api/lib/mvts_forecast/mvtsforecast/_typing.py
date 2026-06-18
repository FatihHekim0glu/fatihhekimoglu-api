"""Shared type aliases for the mvts-forecast library.

These aliases document *intent* at function boundaries (a wide returns panel vs.
a single target series vs. a 3-D windowed sequence tensor) without committing to
a single concrete container. Functions coerce inputs to the canonical
pandas/numpy type via :mod:`mvtsforecast._validation` at the boundary, so the
aliases are deliberately broad. Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# quantcore-candidate: mirrors factorlab:src/factorlab/_typing.py

#: A wide panel of asset RETURNS: rows indexed by time, columns by asset/feature.
#: Accepted at the boundary as a DataFrame, an ndarray, or a mapping coercible to
#: a DataFrame; canonicalized to ``pd.DataFrame`` internally.
ReturnsLike: TypeAlias = "pd.DataFrame | NDArray[np.float64]"

#: A wide panel of asset PRICES (levels). Same shape conventions as
#: :data:`ReturnsLike`; differenced via ``pct_change(fill_method=None)``.
PricesLike: TypeAlias = "pd.DataFrame | NDArray[np.float64]"

#: A 1-D series of (returns of) the forecast TARGET column, indexed by time.
TargetLike: TypeAlias = "pd.Series | NDArray[np.float64]"

#: A 3-D supervised sequence tensor shaped ``(n_samples, look_back, n_features)``
#: — the canonical multivariate encoder input produced by
#: ``windowing.windows.make_windows``.
SequenceTensor: TypeAlias = NDArray[np.float64]

#: A float64 numpy array of unspecified shape (compute-kernel intermediate).
FloatArray: TypeAlias = NDArray[np.float64]
