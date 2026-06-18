"""Offline training and ONNX export (the ``[train]`` path — never in the container).

Everything in this subpackage is gated behind the heavy ``[train]`` extra
(torch + transformers + onnx) and is imported LAZILY inside functions, so
importing this subpackage stays free of any deep-learning framework. The serve
container installs only ``[serve]`` (onnxruntime + tokenizers) and never touches
these modules. Importing this subpackage has no side effects.
"""

from __future__ import annotations

from finbert_sentiment.model.export import (
    ExportResult,
    export_available,
    export_to_onnx,
    write_metrics_json,
)
from finbert_sentiment.model.train import (
    TrainConfig,
    TrainResult,
    build_label_arrays,
    train_available,
    train_distilbert,
)

__all__ = [
    "ExportResult",
    "TrainConfig",
    "TrainResult",
    "build_label_arrays",
    "export_available",
    "export_to_onnx",
    "train_available",
    "train_distilbert",
    "write_metrics_json",
]
