"""Vendored copy of mvts-forecast/src/mvtsforecast — DO NOT EDIT IN PLACE.

Re-vendor from `~/projects/mvts-forecast/src/mvtsforecast` if upstream changes
(INCLUDING the committed ``artifacts/*.onnx`` deep models + ``metrics.json`` that
ship with the package).

The vendored package uses absolute imports (``from mvtsforecast.serve import ...``),
so we add this directory to ``sys.path`` once on first import. This lets the
vendored modules resolve ``mvtsforecast`` as a top-level package without any edits
to the vendored files themselves (mirrors api/lib/lstm_forecast and api/lib/hrp).

The package is import-pure: importing it pulls in NO torch and NO onnxruntime. The
mvts-forecast router serves the deep models through onnxruntime ONLY (the
``[serve]`` extra); torch is never imported on this path. The naive + ARIMA
baselines are pure numpy/statsmodels and run live.

DEPENDENCY: the vendored ``mvtsforecast`` now imports the shared ``quantcore``
kernel (``mvtsforecast.evaluation.dsr`` re-exports DSR/PSR + the honest
cross-trial ``variance_of_trial_sharpes`` helper from quantcore, and
``serve``/``train`` import it on the verdict path). We therefore register the
vendored ``quantcore`` source tree onto ``sys.path`` FIRST so ``import
quantcore`` resolves before any ``mvtsforecast`` import triggers it (mirrors
api/lib/regime_hmm and api/lib/stockclusters).
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Register the vendored quantcore path before mvtsforecast so its
# ``from quantcore import ...`` re-exports resolve on first import.
from .. import quantcore as _quantcore_vendor  # noqa: F401

_VENDOR_DIR = _Path(__file__).resolve().parent
_VENDOR_PATH = str(_VENDOR_DIR)
if _VENDOR_PATH not in _sys.path:
    _sys.path.insert(0, _VENDOR_PATH)
