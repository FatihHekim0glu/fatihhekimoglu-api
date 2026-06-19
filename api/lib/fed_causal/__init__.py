"""Vendored copy of fed-causal/src/fedcausal — DO NOT EDIT IN PLACE.

Re-vendor from `~/projects/fed-causal/src/fedcausal` if upstream changes
(INCLUDING the committed FOMC calendar ``events/calendar_data.py`` and the
precomputed-reference ``artifacts/reference.json`` that ship with the package).

The vendored package uses absolute imports (``from fedcausal.serve import ...``),
so we add this directory to ``sys.path`` once on first import. This lets the
vendored modules resolve ``fedcausal`` as a top-level package without any edits to
the vendored files themselves (mirrors api/lib/hrp, api/lib/mvts_forecast and
api/lib/gnn_stocks).

The package is import-pure and TORCH-FREE: importing it pulls in NO torch, NO
onnx/onnxruntime and NO sklearn — the whole event-study + HAC + placebo + DiD
stack is pure numpy/scipy/statsmodels. There are ZERO import-time side effects
(no network, no FRED/Polygon, no statsmodels/plotly at import). The fed-causal
router scores the seeded synthetic event panel (or the committed reference) per
request and NEVER refits a heavy model live; the estimation-window-only market
model is a cheap OLS. Real data (keyless FRED + Polygon PIT universe) is fetched
lazily via httpx and degrades to the synthetic/committed panel on any failure.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_VENDOR_DIR = _Path(__file__).resolve().parent
_VENDOR_PATH = str(_VENDOR_DIR)
if _VENDOR_PATH not in _sys.path:
    _sys.path.insert(0, _VENDOR_PATH)
