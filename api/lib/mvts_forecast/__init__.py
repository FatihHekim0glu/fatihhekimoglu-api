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
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_VENDOR_DIR = _Path(__file__).resolve().parent
_VENDOR_PATH = str(_VENDOR_DIR)
if _VENDOR_PATH not in _sys.path:
    _sys.path.insert(0, _VENDOR_PATH)
