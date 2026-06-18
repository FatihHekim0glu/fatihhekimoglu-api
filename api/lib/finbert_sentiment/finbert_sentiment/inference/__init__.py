"""Inference layer: lazy ONNX session + a unified predictor over both backends.

The public surface is the :class:`~finbert_sentiment.inference.predictor.Predictor`
and the :class:`~finbert_sentiment.inference.predictor.Prediction` it returns. The
predictor serves the transformer-ONNX model when its artifacts are present and
falls back to the torch-free lexicon otherwise — behind one ``predict(texts)``
interface. Nothing here imports onnxruntime/tokenizers at module load; both are
imported lazily inside the session. Importing this subpackage has no side effects.
"""

from __future__ import annotations

from finbert_sentiment.inference.onnx_session import OnnxSentimentSession, default_artifact_dir
from finbert_sentiment.inference.predictor import Prediction, Predictor, load_predictor

__all__ = [
    "OnnxSentimentSession",
    "Prediction",
    "Predictor",
    "default_artifact_dir",
    "load_predictor",
]
