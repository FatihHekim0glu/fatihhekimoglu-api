"""Vendored copy of nn-vs-bs/src/nnvsbs — DO NOT EDIT IN PLACE.

Re-vendor from `~/projects/nn-vs-bs/src/nnvsbs` if upstream changes
(INCLUDING the committed ``artifacts/*.onnx`` model that ships with the package).

The vendored package uses absolute imports (``from nnvsbs.serve import ...``),
so we add this directory to ``sys.path`` once on first import. This lets the
vendored modules resolve ``nnvsbs`` as a top-level package without any edits
to the vendored files themselves (mirrors api/lib/hrp and api/lib/lstm_forecast).

The package is import-pure: importing it pulls in NO torch and NO onnxruntime.
The nn-vs-bs router serves inference through onnxruntime ONLY (the ``[serve]``
extra); torch is never imported on this path.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_VENDOR_DIR = _Path(__file__).resolve().parent
_VENDOR_PATH = str(_VENDOR_DIR)
if _VENDOR_PATH not in _sys.path:
    _sys.path.insert(0, _VENDOR_PATH)
