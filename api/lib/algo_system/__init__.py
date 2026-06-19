"""Vendored copy of algo-system/src/algosystem — DO NOT EDIT IN PLACE.

Re-vendor from `~/projects/algo-system/src/algosystem` if upstream changes
(INCLUDING the committed precomputed-reference ``artifacts/reference.json`` that
ships with the package).

The vendored package uses absolute imports (``from algosystem.serve import ...``),
so we add this directory to ``sys.path`` once on first import. This lets the
vendored modules resolve ``algosystem`` as a top-level package without any edits to
the vendored files themselves (mirrors api/lib/fed_causal, api/lib/hrp and
api/lib/rl_trader).

The package is import-pure and TORCH-FREE: importing it pulls in NO torch, NO
onnx/onnxruntime, NO sklearn, NO stable-baselines3 and NO gymnasium — the whole
signal -> purged walk-forward backtest -> simulated bar-by-bar paper-broker
execution -> backtest<->live PARITY ORACLE -> DM/DSR/PBO/HAC stack is pure
numpy/scipy/statsmodels. There are ZERO import-time side effects (no network, no
broker, no Polygon, no plotly/typer at import; clients are lazy). The algo-system
router runs the FULL pipeline on the seeded synthetic default per request (a cheap
vectorized backtest + a fast paper-broker replay) and NEVER trains a heavy model.
Execution is SIMULATED (next-bar-open fills + costs + slippage) — there is no live
broker and no broker key. Real data (the Polygon PIT single-asset bars) is fetched
lazily and degrades to the synthetic bars on any failure.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_VENDOR_DIR = _Path(__file__).resolve().parent
_VENDOR_PATH = str(_VENDOR_DIR)
if _VENDOR_PATH not in _sys.path:
    _sys.path.insert(0, _VENDOR_PATH)
