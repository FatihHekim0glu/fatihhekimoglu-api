"""Honest-statistics layer: QLIKE/MSE losses, DM/SPA inference, DSR, and verdicts.

The headline ``best_model`` / ``ml_beats_garch`` verdict is a PURE function of
the OOS QLIKE and the DM/SPA significance. Importing this subpackage has no side
effects (``arch`` is imported lazily inside the SPA function).
"""

from __future__ import annotations

from volforecast.evaluation.dsr import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from volforecast.evaluation.qlike import mse, qlike, qlike_loss_series
from volforecast.evaluation.tests import (
    DMResult,
    SPAResult,
    diebold_mariano,
    hansen_spa,
    newey_west_lrv,
)
from volforecast.evaluation.verdict import (
    ML_MODELS,
    REFERENCE_MODELS,
    BestModelClass,
    Verdict,
    derive_verdict,
)

__all__ = [
    "ML_MODELS",
    "REFERENCE_MODELS",
    "BestModelClass",
    "DMResult",
    "SPAResult",
    "Verdict",
    "deflated_sharpe_ratio",
    "derive_verdict",
    "diebold_mariano",
    "hansen_spa",
    "mse",
    "newey_west_lrv",
    "probabilistic_sharpe_ratio",
    "qlike",
    "qlike_loss_series",
]
