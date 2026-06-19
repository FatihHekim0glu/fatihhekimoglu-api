"""Shared type aliases for the fed-causal library.

These aliases document *intent* at function boundaries (a wide panel of single-
name returns vs. a market-return series vs. an abnormal-return matrix) without
committing to a single concrete container. Functions coerce inputs to the
canonical pandas type via :mod:`fedcausal._validation` at the boundary, so the
aliases are deliberately broad. Importing this module has no side effects.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# quantcore-candidate: mirrors hrp-portfolio:src/hrp/_typing.py

#: A wide panel of single-name simple returns: rows indexed by (trading) date,
#: columns by ticker. Accepted at the boundary as a DataFrame, an ndarray, or a
#: mapping coercible to a DataFrame; canonicalized to ``pd.DataFrame`` internally.
ReturnsLike: TypeAlias = "pd.DataFrame | NDArray[np.float64]"

#: A 1-D market (or benchmark) return series indexed by date, used as the
#: regressor in the market model. Canonicalized to ``pd.Series`` internally.
MarketReturnsLike: TypeAlias = "pd.Series | NDArray[np.float64]"

#: A matrix of abnormal returns (actual minus model-expected), name x time or
#: time x name depending on context; coerced at the boundary.
AbnormalReturnsLike: TypeAlias = "pd.DataFrame | NDArray[np.float64]"

#: A float64 numpy array of unspecified shape (compute-kernel intermediate).
FloatArray: TypeAlias = NDArray[np.float64]

#: A FOMC monetary-policy surprise label, derived ONLY from information available
#: at the announcement (the sign of the target-rate change). ``"neutral"`` means
#: no change; ``"hawkish"`` a hike (tightening); ``"dovish"`` a cut (easing).
SurpriseLabel: TypeAlias = Literal["hawkish", "dovish", "neutral"]

#: The expected-return model used to compute abnormal returns. ``"market"`` fits
#: a one-factor market model (alpha + beta * market) on the estimation window;
#: ``"mean_adjusted"`` uses the estimation-window mean return.
ModelKind: TypeAlias = Literal["market", "mean_adjusted"]

#: Where the event panel came from. Returned alongside results so callers (and
#: the API ``data_source`` field) can report provenance.
DataSource: TypeAlias = Literal["synthetic", "fred+polygon"]
