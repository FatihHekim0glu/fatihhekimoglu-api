"""Vendored compute modules for the pairs-trading tool.

DO NOT EDIT FILES UNDER `_vendor/` — they are a verbatim copy of
`pairs-trading/src/pairs/` (upstream package name `pairs`). Re-vendor with
`cp -r` if upstream changes.

The vendored package uses absolute imports of the form ``from pairs.X import Y``,
so we prepend the ``_vendor/`` directory to ``sys.path`` at import time. This
lets the router do ``from pairs.selection import screen_cointegration`` without
rewriting the source.
"""

from __future__ import annotations

import sys
from pathlib import Path

_VENDOR_ROOT = Path(__file__).parent / "_vendor"
_vendor_str = str(_VENDOR_ROOT)
if _vendor_str not in sys.path:
    sys.path.insert(0, _vendor_str)
