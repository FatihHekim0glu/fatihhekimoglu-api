"""Models: naive + ARIMA baselines (torch-free, live), deep encoders, ONNX serve.

IMPORT PURITY: importing this subpackage pulls in ONLY the baseline + config
dataclasses (pure numpy). torch (``models.lstm`` / ``models.patchtst`` /
``models.transformer_vs`` builders+trainers) and onnxruntime
(``models.onnx_runtime``) are imported LAZILY inside their functions, never at
module load — so ``import mvtsforecast`` never imports torch or onnxruntime.

Importing this subpackage has no side effects.
"""

from __future__ import annotations

from mvtsforecast.models.arima import ArimaResult
from mvtsforecast.models.lstm import LstmConfig
from mvtsforecast.models.naive import NaiveResult, naive_forecast, persistence_returns
from mvtsforecast.models.patchtst import PatchTSTConfig
from mvtsforecast.models.transformer_vs import TransformerVSConfig

__all__ = [
    "ArimaResult",
    "LstmConfig",
    "NaiveResult",
    "PatchTSTConfig",
    "TransformerVSConfig",
    "naive_forecast",
    "persistence_returns",
]
