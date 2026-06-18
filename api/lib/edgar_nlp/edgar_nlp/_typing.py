"""Shared type aliases for the edgar-nlp library.

These aliases document *intent* at function boundaries (a tokenized document vs.
a tone-panel frame vs. a forward-return vector) without committing to a single
concrete container. Functions coerce inputs to the canonical type at the boundary
via :mod:`edgar_nlp._validation`, so the aliases are deliberately broad. Importing
this module has no side effects.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# quantcore-candidate: mirrors hrp-portfolio:src/hrp/_typing.py

#: A sequence of word tokens extracted from filing text (already lowercased and
#: stripped of punctuation by the tokenizer at the boundary).
TokenList: TypeAlias = "list[str]"

#: A long-form tone panel: one row per (filing, acceptance-date) with at least a
#: ``tone`` column and a forward-return column. Accepted at the boundary as a
#: DataFrame; canonicalized internally.
TonePanelLike: TypeAlias = "pd.DataFrame"

#: A wide panel of forward returns: rows indexed by date, columns by issuer.
#: Differenced via ``pct_change(fill_method=None)`` — never forward-filled.
ReturnsLike: TypeAlias = "pd.DataFrame | NDArray[np.float64]"

#: A wide panel of issuer prices (levels). Same shape conventions as
#: :data:`ReturnsLike`; differenced via ``pct_change(fill_method=None)``.
PricesLike: TypeAlias = "pd.DataFrame | NDArray[np.float64]"

#: A vector of per-period returns (e.g. the tone-tertile long-short spread).
ReturnSeriesLike: TypeAlias = "pd.Series | NDArray[np.float64]"

#: A float64 numpy array of unspecified shape (compute-kernel intermediate).
FloatArray: TypeAlias = NDArray[np.float64]
