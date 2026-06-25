"""Vendored copy of quantcore/src/quantcore — DO NOT EDIT IN PLACE.

Re-vendor from `~/projects/quantcore/src/quantcore` if upstream changes.

quantcore is the shared, torch-free honest-statistics kernel
(DSR/PSR/HAC/Diebold-Mariano/Memmel/PBO/walk-forward/verdict). Several other
vendored libraries (e.g. ``regimehmm``) now import it as a top-level package
(``from quantcore import deflated_sharpe_ratio``), so we add this directory to
``sys.path`` once on first import. This lets the vendored modules resolve
``quantcore`` without any edits to the vendored files themselves (mirrors
api/lib/regime_hmm and api/lib/hrp).
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_VENDOR_DIR = _Path(__file__).resolve().parent
_VENDOR_PATH = str(_VENDOR_DIR)
if _VENDOR_PATH not in _sys.path:
    _sys.path.insert(0, _VENDOR_PATH)
