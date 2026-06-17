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
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_VENDOR_DIR = _Path(__file__).resolve().parent
_VENDOR_PATH = str(_VENDOR_DIR)
if _VENDOR_PATH not in _sys.path:
    _sys.path.insert(0, _VENDOR_PATH)
