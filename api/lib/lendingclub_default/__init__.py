"""Vendored copy of lendingclub-default/src/lendingclub_default — DO NOT EDIT IN PLACE.

Re-vendor from `~/projects/lendingclub-default/src/lendingclub_default` if upstream
changes (copy the inner package byte-for-byte, including ``artifacts/``).

The vendored package uses absolute imports (``from lendingclub_default.data import
...``), so we add this directory to ``sys.path`` once on first import. This lets the
vendored modules resolve ``lendingclub_default`` as a top-level package without any
edits to the vendored files themselves (mirrors api/lib/hrp).
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_VENDOR_DIR = _Path(__file__).resolve().parent
_VENDOR_PATH = str(_VENDOR_DIR)
if _VENDOR_PATH not in _sys.path:
    _sys.path.insert(0, _VENDOR_PATH)
