"""Vendored copy of regime-hmm/src/regimehmm — DO NOT EDIT IN PLACE.

Re-vendor from `~/projects/regime-hmm/src/regimehmm` if upstream changes.

The vendored package uses absolute imports (``from regimehmm.hmm import ...``),
so we add this directory to ``sys.path`` once on first import. This lets the
vendored modules resolve ``regimehmm`` as a top-level package without any edits
to the vendored files themselves (mirrors api/lib/hrp).

DEPENDENCY: the vendored ``regimehmm`` now imports the shared ``quantcore``
kernel (``regimehmm.evaluation.dsr`` re-exports DSR/PSR from quantcore), and
``regimehmm.__init__`` imports ``evaluation.dsr`` on load. We therefore register
the vendored ``quantcore`` source tree onto ``sys.path`` FIRST so ``import
quantcore`` resolves before any ``regimehmm`` import triggers it.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Register the vendored quantcore path before regimehmm so its
# ``from quantcore import ...`` re-exports resolve on first import.
from .. import quantcore as _quantcore_vendor  # noqa: F401

_VENDOR_DIR = _Path(__file__).resolve().parent
_VENDOR_PATH = str(_VENDOR_DIR)
if _VENDOR_PATH not in _sys.path:
    _sys.path.insert(0, _VENDOR_PATH)
