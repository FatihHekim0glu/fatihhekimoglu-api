"""Vendored copy of markowitz-optimizer/src - DO NOT EDIT IN PLACE.

Re-vendor from `~/markowitz-optimizer/src/markowitz` if upstream changes.

The vendored package uses absolute imports (``from markowitz.core import ...``),
so we add this directory to ``sys.path`` once on first import. This lets the
vendored modules resolve ``markowitz`` as a top-level package without any
edits to the vendored files themselves.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_VENDOR_DIR = _Path(__file__).resolve().parent
_VENDOR_PATH = str(_VENDOR_DIR)
if _VENDOR_PATH not in _sys.path:
    _sys.path.insert(0, _VENDOR_PATH)
