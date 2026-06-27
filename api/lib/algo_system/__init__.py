"""Vendored copy of algo-system/src/algosystem — DO NOT EDIT IN PLACE.

Re-vendor from `~/projects/algo-system/src/algosystem` if upstream changes
(INCLUDING the committed precomputed-reference ``artifacts/reference.json`` that
ships with the package).

The vendored package uses absolute imports (``from algosystem.serve import ...``),
so we add this directory to ``sys.path`` once on first import. This lets the
vendored modules resolve ``algosystem`` as a top-level package without any edits to
the vendored files themselves (mirrors api/lib/fed_causal, api/lib/hrp and
api/lib/rl_trader).

DEPENDENCY: the vendored ``algosystem`` now imports the shared ``quantcore``
kernel — ``algosystem.evaluation.dsr`` / ``.pbo`` / ``.hac`` re-export PSR/DSR,
PBO/CSCV and the Newey-West HAC standard error from quantcore (drift-elimination;
the math is byte-identical because quantcore was seeded from this repo), and
``algosystem.__init__`` imports the evaluation layer on load. We therefore register
the vendored ``quantcore`` source tree onto ``sys.path`` FIRST so ``import
quantcore`` resolves before any ``algosystem`` import triggers it.

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

# Register the vendored quantcore path before algosystem so its
# ``from quantcore import ...`` re-exports resolve on first import.
from .. import quantcore as _quantcore_vendor  # noqa: F401

_VENDOR_DIR = _Path(__file__).resolve().parent
_VENDOR_PATH = str(_VENDOR_DIR)
if _VENDOR_PATH not in _sys.path:
    _sys.path.insert(0, _VENDOR_PATH)
