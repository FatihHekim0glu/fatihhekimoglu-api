"""Vendored copy of lstm-forecast/src/lstmforecast — DO NOT EDIT IN PLACE.

Re-vendor from `~/projects/lstm-forecast/src/lstmforecast` if upstream changes
(INCLUDING the committed ``artifacts/*.onnx`` model that ships with the package).

The vendored package uses absolute imports (``from lstmforecast.serve import ...``),
so we add this directory to ``sys.path`` once on first import. This lets the
vendored modules resolve ``lstmforecast`` as a top-level package without any edits
to the vendored files themselves (mirrors api/lib/hrp and api/lib/regime_hmm).

The package is import-pure: importing it pulls in NO TensorFlow and NO onnxruntime.
The lstm-forecast router serves inference through onnxruntime ONLY (the ``[serve]``
extra); TensorFlow is never imported on this path.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Register the vendored quantcore path before lstmforecast so its
# ``from quantcore import ...`` re-exports resolve on first import
# (mirrors api/lib/regime_hmm and api/lib/crypto_arb_scanner).
from .. import quantcore as _quantcore_vendor  # noqa: F401

_VENDOR_DIR = _Path(__file__).resolve().parent
_VENDOR_PATH = str(_VENDOR_DIR)
if _VENDOR_PATH not in _sys.path:
    _sys.path.insert(0, _VENDOR_PATH)
