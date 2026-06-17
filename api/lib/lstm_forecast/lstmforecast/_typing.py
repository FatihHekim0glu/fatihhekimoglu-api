"""Shared type aliases for the lstm-forecast library.

These aliases document *intent* at function boundaries (a price series vs. a
return series vs. a feature matrix vs. a 3-D sequence tensor) without committing
to a single concrete container. Functions coerce inputs to the canonical
pandas/numpy type via :mod:`lstmforecast._validation` at the boundary, so the
aliases are deliberately broad. Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# quantcore-candidate: mirrors factorlab:src/factorlab/_typing.py

#: A 1-D series of price *levels* indexed by time. Accepted at the boundary as a
#: ``pd.Series`` or a 1-D ``np.ndarray``; canonicalized to ``pd.Series``.
PriceLike: TypeAlias = "pd.Series | NDArray[np.float64]"

#: A 1-D series of (log-)returns indexed by time. Same shape conventions as
#: :data:`PriceLike`; derived via ``np.log(price).diff()`` / ``pct_change``.
ReturnLike: TypeAlias = "pd.Series | NDArray[np.float64]"

#: A 2-D feature matrix: rows indexed by time, columns by engineered feature.
#: Accepted as a DataFrame or a 2-D ndarray; canonicalized to ``pd.DataFrame``.
FeatureFrameLike: TypeAlias = "pd.DataFrame | NDArray[np.float64]"

#: A 3-D supervised sequence tensor shaped ``(n_samples, look_back, n_features)``
#: — the canonical LSTM input produced by ``features.sequences.create_sequences``.
SequenceTensor: TypeAlias = NDArray[np.float64]

#: A float64 numpy array of unspecified shape (compute-kernel intermediate).
FloatArray: TypeAlias = NDArray[np.float64]
