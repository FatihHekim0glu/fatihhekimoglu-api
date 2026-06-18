"""Vendored copy of finbert-sentiment/src/finbert_sentiment — DO NOT EDIT IN PLACE.

Re-vendor from ``~/projects/finbert-sentiment/src/finbert_sentiment`` if upstream
changes (INCLUDING the committed ``artifacts/`` directory: the DistilBERT
``model.int8.onnx`` graph, its ``tokenizer.json``, and the offline-measured
``metrics.json`` that ships with the package).

The vendored package uses absolute imports (``from finbert_sentiment.service
import ...``), so we add this directory to ``sys.path`` once on first import. This
lets the vendored modules resolve ``finbert_sentiment`` as a top-level package
without any edits to the vendored files themselves (mirrors api/lib/hrp and
api/lib/lstm_forecast).

The package is import-pure: importing it pulls in NO torch, NO transformers, and
NO onnxruntime. The finbert-sentiment router serves inference through
onnxruntime + tokenizers ONLY (the ``[serve]`` extra) when the committed ONNX
artifact is present, else the torch-free lexicon baseline; torch/transformers are
never imported on this serve path.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_VENDOR_DIR = _Path(__file__).resolve().parent
_VENDOR_PATH = str(_VENDOR_DIR)
if _VENDOR_PATH not in _sys.path:
    _sys.path.insert(0, _VENDOR_PATH)
