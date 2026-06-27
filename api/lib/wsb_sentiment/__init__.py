"""Vendored copy of wsb-sentiment/src/wsb_sentiment — DO NOT EDIT IN PLACE.

Re-vendor from `~/projects/wsb-sentiment/src/wsb_sentiment` if upstream changes.

The vendored package uses absolute imports (``from wsb_sentiment.backtest import
...``), so we add this directory to ``sys.path`` once on first import. This lets
the vendored modules resolve ``wsb_sentiment`` as a top-level package without any
edits to the vendored files themselves (mirrors api/lib/hrp).

IMPORT PURITY: importing the vendored ``wsb_sentiment`` pulls NO praw / torch /
transformers / vaderSentiment / textblob / plotly — those are imported lazily
inside the functions that need them, and ingestion/scoring are OFFLINE batch
paths never run at request time.

DEPENDENCY: the vendored ``wsb_sentiment`` now imports the shared ``quantcore``
kernel (``wsb_sentiment.evaluation.dsr`` re-exports DSR/PSR and the ``_norm_*``
helpers from quantcore, and ``wsb_sentiment.evaluation.hac`` sources the
Newey-West / Andrews kernel from it). ``wsb_sentiment.__init__`` imports the
evaluation layer on load, so we register the vendored ``quantcore`` source tree
onto ``sys.path`` FIRST so ``import quantcore`` resolves before any
``wsb_sentiment`` import triggers it (mirrors api/lib/regime_hmm).
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Register the vendored quantcore path before wsb_sentiment so its
# ``from quantcore import ...`` re-exports resolve on first import.
from .. import quantcore as _quantcore_vendor  # noqa: F401

_VENDOR_DIR = _Path(__file__).resolve().parent
_VENDOR_PATH = str(_VENDOR_DIR)
if _VENDOR_PATH not in _sys.path:
    _sys.path.insert(0, _VENDOR_PATH)
