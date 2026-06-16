"""Shared type aliases for the lendingclub-default library.

These aliases document *intent* at function boundaries (a loan-application panel
vs. a feature matrix vs. a label/probability vector) without committing to a
single concrete container. Functions coerce inputs to the canonical pandas type
via :mod:`lendingclub_default._validation` at the boundary, so the aliases are
deliberately broad. Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# quantcore-candidate: mirrors factorlab:src/factorlab/_typing.py

#: A loan-application panel: one row per loan, columns are LC application-time
#: (and, pre-leakage-drop, post-funding) fields. Accepted at the boundary as a
#: DataFrame or a mapping coercible to one; canonicalized to ``pd.DataFrame``.
PanelLike: TypeAlias = "pd.DataFrame"

#: A 2-D numeric design matrix (rows = loans, columns = engineered features) as
#: produced by the fitted feature pipeline.
FeatureMatrixLike: TypeAlias = "pd.DataFrame | NDArray[np.float64]"

#: A binary label vector (1 = default / charged-off, 0 = fully paid), one entry
#: per loan, indexed to match the panel.
LabelLike: TypeAlias = "pd.Series | NDArray[np.int_]"

#: A vector of predicted default probabilities (PDs) in ``[0, 1]``, one per loan.
ProbaLike: TypeAlias = "pd.Series | NDArray[np.float64]"

#: A float64 numpy array of unspecified shape (compute-kernel intermediate).
FloatArray: TypeAlias = NDArray[np.float64]
