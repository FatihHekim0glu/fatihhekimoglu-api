"""Shared type aliases for the rag10k library.

These aliases document *intent* at function boundaries (a single dense embedding
vector vs. a stacked matrix of chunk vectors vs. a generic float array) without
committing to a single concrete container. Functions coerce inputs to the
canonical numpy type via :mod:`rag10k._validation` at the boundary, so the
aliases are deliberately broad. Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

# quantcore-candidate: mirrors mvts-forecast:src/mvtsforecast/_typing.py

#: A single dense embedding vector of shape ``(embedding_dim,)`` — the output of
#: the ONNX sentence-transformer for one query or one chunk.
EmbeddingVector: TypeAlias = NDArray[np.float32]

#: A stacked matrix of dense embeddings of shape ``(n_chunks, embedding_dim)`` —
#: the committed corpus index, one L2-normalized row per chunk.
EmbeddingMatrix: TypeAlias = NDArray[np.float32]

#: A 1-D array of cosine similarity scores, one per candidate chunk.
ScoreArray: TypeAlias = NDArray[np.float64]

#: A 1-D integer array of chunk row indices (e.g. top-k selections into the index).
IndexArray: TypeAlias = NDArray[np.int64]

#: A float64 numpy array of unspecified shape (compute-kernel intermediate).
FloatArray: TypeAlias = NDArray[np.float64]
