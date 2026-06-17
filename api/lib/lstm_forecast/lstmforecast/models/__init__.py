"""Models: the persistence baseline, the Keras LSTM (train-only), and the ONNX serve path.

IMPORT PURITY: importing this subpackage pulls in ONLY the baseline (pure
numpy). TensorFlow (``models.lstm``) and onnxruntime (``models.onnx_runtime``)
are imported LAZILY inside their functions, never at module load — so
``import lstmforecast`` never imports TF or onnxruntime.

Importing this subpackage has no side effects.
"""

from __future__ import annotations

from lstmforecast.models.baselines import PersistenceForecaster

__all__ = ["PersistenceForecaster"]
