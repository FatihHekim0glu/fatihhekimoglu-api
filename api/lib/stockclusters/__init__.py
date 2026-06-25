"""Vendored copy of stock-clusters/src/stockclusters — DO NOT EDIT IN PLACE.

Re-vendor from `~/projects/stock-clusters/src/stockclusters` if upstream changes.

The vendored package uses absolute imports (``from stockclusters.correlation
import ...``), so we add this directory to ``sys.path`` once on first import. This
lets the vendored modules resolve ``stockclusters`` as a top-level package without
any edits to the vendored files themselves (mirrors api/lib/hrp).

DEPENDENCY: the vendored ``stockclusters`` now imports the shared ``quantcore``
kernel (``stockclusters.evaluation.dsr`` re-exports DSR/PSR from quantcore, and
``stockclusters.cli`` uses ``quantcore.variance_of_trial_sharpes``). We therefore
register the vendored ``quantcore`` source tree onto ``sys.path`` FIRST so
``import quantcore`` resolves before any ``stockclusters`` import triggers it
(mirrors api/lib/regime_hmm).
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Register the vendored quantcore path before stockclusters so its
# ``import quantcore`` re-exports resolve on first import.
from .. import quantcore as _quantcore_vendor  # noqa: F401

_VENDOR_DIR = _Path(__file__).resolve().parent
_VENDOR_PATH = str(_VENDOR_DIR)
if _VENDOR_PATH not in _sys.path:
    _sys.path.insert(0, _VENDOR_PATH)
