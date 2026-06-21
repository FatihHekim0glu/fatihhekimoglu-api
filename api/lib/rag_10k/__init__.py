"""Vendored copy of rag-10k/src/rag10k — DO NOT EDIT IN PLACE.

Re-vendor from `~/projects/rag-10k/src/rag10k` if upstream changes (INCLUDING the
committed serve artifacts that ship with the package: the int8-quantized
``artifacts/embedder.int8.onnx`` + ``artifacts/tokenizer.json`` + the read-only
``artifacts/corpus_index.npz`` index + the frozen-harness ``artifacts/eval.json``
+ ``artifacts/eval_set.json``). The FP32 export intermediate (``embedder.onnx``,
~90MB) is an OFFLINE build artifact and is intentionally NOT vendored — the serve
path loads ONLY the int8 graph.

The vendored package uses absolute imports (``from rag10k.serve import ...``), so
we add this directory to ``sys.path`` once on first import. This lets the vendored
modules resolve ``rag10k`` as a top-level package without any edits to the
vendored files themselves (mirrors api/lib/mvts_forecast, api/lib/lstm_forecast,
and api/lib/hrp).

The package is import-pure: importing it pulls in NO torch, NO onnxruntime, NO
tokenizers, and NO anthropic/openai. Retrieval is served torch-free through
onnxruntime + tokenizers (the ``[serve]`` extra) and the committed int8 embedder +
read-only index; only the lazily-built LLM client (key-gated) ever imports a
provider SDK. Absent an ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY``, the rag-10k
router serves the KEY-FREE extractive cited answer — it never hard-fails.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_VENDOR_DIR = _Path(__file__).resolve().parent
_VENDOR_PATH = str(_VENDOR_DIR)
if _VENDOR_PATH not in _sys.path:
    _sys.path.insert(0, _VENDOR_PATH)
