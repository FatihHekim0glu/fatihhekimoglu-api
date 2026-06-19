"""Vendored copy of rl-trader/src/rltrader — DO NOT EDIT IN PLACE.

Re-vendor from `~/projects/rl-trader/src/rltrader` if upstream changes
(INCLUDING the committed ``artifacts/policy.onnx`` ONNX policy + ``metrics.json``
+ precomputed OOS equity that ship with the package).

The vendored package uses absolute imports (``from rltrader.serve import ...``),
so we add this directory to ``sys.path`` once on first import. This lets the
vendored modules resolve ``rltrader`` as a top-level package without any edits to
the vendored files themselves (mirrors api/lib/gnn_stocks and api/lib/mvts_forecast).

The package is import-pure: importing it pulls in NO torch, NO stable-baselines3,
NO gymnasium and NO onnxruntime. The rl-trader router serves the committed PPO
policy through onnxruntime ONLY (the ``[serve]`` extra); torch / sb3 / gymnasium
are never imported on this serve path. The buy-hold / flat-cash / random baselines
are pure numpy and run live.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_VENDOR_DIR = _Path(__file__).resolve().parent
_VENDOR_PATH = str(_VENDOR_DIR)
if _VENDOR_PATH not in _sys.path:
    _sys.path.insert(0, _VENDOR_PATH)
