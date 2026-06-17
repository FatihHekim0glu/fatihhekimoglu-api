"""Machine-learning volatility forecasters.

CONTAINER GUARANTEE: this subpackage re-exports ONLY the XGBoost arm. The
research-only LSTM (:mod:`volforecast.ml.lstm`) is deliberately NOT imported
here, so neither ``import volforecast`` nor ``import volforecast.ml`` pulls in
TensorFlow. Callers who genuinely want the research LSTM must import
``volforecast.ml.lstm`` explicitly (and install the ``[research]`` extra).

Importing this subpackage has no side effects (``xgboost`` is imported lazily
inside the fitting function).
"""

from __future__ import annotations

from volforecast.ml.xgb import DEFAULT_XGB_PARAMS, XGBForecaster, fit_xgb

__all__ = [
    "DEFAULT_XGB_PARAMS",
    "XGBForecaster",
    "fit_xgb",
]
