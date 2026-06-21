"""Embedding: the ONNX serve embedder + the offline index/export builder.

:mod:`rag10k.embed.onnx_embedder` embeds queries and chunks at SERVE time with the
committed ONNX sentence-transformer via onnxruntime + tokenizers (NO torch; lazy
session). :mod:`rag10k.embed.build_index` is the OFFLINE ``[embed]`` path (torch +
sentence-transformers + onnx): it exports the embedder to int8 ONNX with 1e-4
parity and embeds the committed corpus into the read-only index.

IMPORT PURITY: no onnxruntime / tokenizers / torch import happens at package
import — the serve session is built lazily on first embed, and the offline
builder imports torch only when invoked.
"""

from __future__ import annotations

from rag10k.embed.onnx_embedder import (
    OnnxEmbedder,
    default_artifact_dir,
    embedder_artifacts_present,
)

__all__ = ["OnnxEmbedder", "default_artifact_dir", "embedder_artifacts_present"]
