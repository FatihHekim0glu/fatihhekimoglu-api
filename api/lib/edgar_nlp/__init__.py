"""Vendored copy of edgar-nlp/src/edgar_nlp — DO NOT EDIT IN PLACE.

Re-vendor from `~/projects/edgar-nlp/src/edgar_nlp` if upstream changes.

The vendored package uses absolute imports (``from edgar_nlp.parse import ...``),
so we add this directory to ``sys.path`` once on first import. This lets the
vendored modules resolve ``edgar_nlp`` as a top-level package without any edits
to the vendored files themselves (mirrors api/lib/hrp and api/lib/finbert_sentiment).

DEPENDENCY: the vendored ``edgar_nlp`` now imports the shared ``quantcore`` kernel
(``edgar_nlp.evaluation.dsr`` re-exports PSR/DSR from quantcore, and
``edgar_nlp.evaluation.hac`` delegates its Newey-West kernel to it). We therefore
register the vendored ``quantcore`` source tree onto ``sys.path`` FIRST so
``import quantcore`` resolves before any ``edgar_nlp`` import triggers it (mirrors
api/lib/regime_hmm).
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Register the vendored quantcore path before edgar_nlp so its
# ``from quantcore import ...`` re-exports resolve on first import.
from .. import quantcore as _quantcore_vendor  # noqa: F401

_VENDOR_DIR = _Path(__file__).resolve().parent
_VENDOR_PATH = str(_VENDOR_DIR)
if _VENDOR_PATH not in _sys.path:
    _sys.path.insert(0, _VENDOR_PATH)
