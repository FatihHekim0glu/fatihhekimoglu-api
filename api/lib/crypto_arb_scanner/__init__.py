"""Vendored copy of crypto-arb-scanner/src/cryptoarb — DO NOT EDIT IN PLACE.

Re-vendor from `~/projects/crypto-arb-scanner/src/cryptoarb` if upstream changes.

The vendored package uses absolute imports (``from cryptoarb.books import ...``),
so we add this directory to ``sys.path`` once on first import. This lets the
vendored modules resolve ``cryptoarb`` as a top-level package without any edits
to the vendored files themselves (mirrors api/lib/hrp).

DEPENDENCY: the vendored ``cryptoarb`` now imports the shared ``quantcore`` kernel
(``cryptoarb.evaluation.dsr`` re-exports the PSR/DSR kernel from quantcore). We
therefore register the vendored ``quantcore`` source tree onto ``sys.path`` FIRST
so ``import quantcore`` resolves before any ``cryptoarb`` import triggers it
(mirrors api/lib/edgar_nlp and api/lib/regime_hmm).
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Register the vendored quantcore path before cryptoarb so its
# ``from quantcore.dsr import ...`` re-exports resolve on first import.
from .. import quantcore as _quantcore_vendor  # noqa: F401

_VENDOR_DIR = _Path(__file__).resolve().parent
_VENDOR_PATH = str(_VENDOR_DIR)
if _VENDOR_PATH not in _sys.path:
    _sys.path.insert(0, _VENDOR_PATH)
