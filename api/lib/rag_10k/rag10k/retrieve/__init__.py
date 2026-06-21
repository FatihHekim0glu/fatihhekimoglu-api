"""Retrieval: the committed read-only vector index + cosine top-k search.

:mod:`rag10k.retrieve.index` loads the prebuilt, committed corpus index (chunk
vectors + chunk text + provenance) read-only. :mod:`rag10k.retrieve.search` embeds
a query with the ONNX embedder, runs a cosine top-k over the index, and applies
the similarity-threshold ABSTENTION gate. Both are deterministic and torch-free.

IMPORT PURITY: numpy only at import; onnxruntime / tokenizers stay lazy behind the
embedder. Importing this package has no side effects.
"""

from __future__ import annotations

from rag10k.retrieve.index import CorpusIndex, default_index_path, index_present
from rag10k.retrieve.search import RetrievalResult, RetrievedChunk, search

__all__ = [
    "CorpusIndex",
    "RetrievalResult",
    "RetrievedChunk",
    "default_index_path",
    "index_present",
    "search",
]
