"""Shared type aliases for the finbert-sentiment library.

These aliases document *intent* at function boundaries (a batch of raw input
texts vs. a vector of integer labels vs. a class-score matrix) without
committing to a single concrete container. Functions coerce inputs to the
canonical type via :mod:`finbert_sentiment._validation` at the boundary, so the
aliases are deliberately broad. Importing this module has no side effects.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

# quantcore-candidate: mirrors factorlab:src/factorlab/_typing.py

#: A batch of raw input sentences to classify. Accepted at the boundary as any
#: sequence of strings; canonicalized to ``list[str]``.
TextBatch: TypeAlias = "Sequence[str]"

#: A 1-D vector of integer class labels in ``{0, 1, 2}`` (see
#: :data:`finbert_sentiment._constants.LABELS` for the index meaning).
LabelArray: TypeAlias = "NDArray[np.int64]"

#: A 1-D vector of string class labels drawn from
#: :data:`finbert_sentiment._constants.LABELS`.
StringLabels: TypeAlias = "Sequence[str]"

#: A ``(n_texts, n_classes)`` matrix of per-class scores (softmax probabilities
#: or lexicon pseudo-probabilities), rows summing to one on the simplex.
ScoreMatrix: TypeAlias = "NDArray[np.float64]"

#: A float64 numpy array of unspecified shape (compute-kernel intermediate).
FloatArray: TypeAlias = NDArray[np.float64]
