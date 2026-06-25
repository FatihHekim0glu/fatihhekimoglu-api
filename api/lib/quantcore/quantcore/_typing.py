"""Shared type aliases for the quantcore library.

These aliases document *intent* at function boundaries (a 1-D return / loss /
performance series vs. a 2-D performance matrix) without committing to a single
concrete container. Functions coerce inputs to ``float64`` numpy arrays at the
boundary via :mod:`quantcore._validation`, so the aliases are deliberately broad
and torch-free. Importing this module has no side effects.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

#: A 1-D numeric series accepted at the boundary: a 1-D ndarray or any sequence
#: of floats coercible to one. Coerced to ``float64`` internally. (pandas Series
#: are accepted too — they satisfy ``Sequence[float]`` for coercion purposes —
#: but quantcore never imports pandas, keeping the dependency set numpy/scipy/
#: statsmodels only.)
Series1D: TypeAlias = "NDArray[np.float64] | Sequence[float]"

#: A float64 numpy array of unspecified shape (compute-kernel intermediate).
FloatArray: TypeAlias = NDArray[np.float64]
