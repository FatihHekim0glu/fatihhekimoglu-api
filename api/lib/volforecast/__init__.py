"""Vendored copy of volforecast/src/volforecast — DO NOT EDIT IN PLACE.

Re-vendor from ``~/projects/volforecast/src/volforecast`` if upstream changes.

The vendored package uses absolute imports (``from volforecast.garch import
...``), so we add this directory to ``sys.path`` once on first import. This lets
the vendored modules resolve ``volforecast`` as a top-level package without any
edits to the vendored files themselves (mirrors api/lib/hrp).

CONTAINER GUARANTEE: importing ``volforecast`` (or the serve-path ``pipeline``)
pulls in NO TensorFlow — the research-only LSTM (``volforecast.ml.lstm``) is
never re-exported by ``volforecast/__init__.py``, ``volforecast.ml.__init__``,
or ``volforecast.pipeline``, so the serve path can never import it.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_VENDOR_DIR = _Path(__file__).resolve().parent
_VENDOR_PATH = str(_VENDOR_DIR)
if _VENDOR_PATH not in _sys.path:
    _sys.path.insert(0, _VENDOR_PATH)
