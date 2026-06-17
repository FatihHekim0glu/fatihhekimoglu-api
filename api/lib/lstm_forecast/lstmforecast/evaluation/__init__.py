"""Honest inference layer: return-space metrics, DM/HAC, DSR, and the verdict.

The headline ``beats_naive`` verdict is a PURE FUNCTION of the inference outputs
(MASE, the Diebold-Mariano p-value, directional accuracy). It cannot read ``True``
unless the model genuinely beats persistence AND the difference is significant.
There is NO price-level R² anywhere. Importing this subpackage has no side
effects.
"""

from __future__ import annotations

from lstmforecast.evaluation.dsr import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from lstmforecast.evaluation.metrics import (
    ForecastMetrics,
    diebold_mariano,
    directional_accuracy,
    forecast_metrics,
    hac_standard_error,
    mae,
    mase_vs_persistence,
    rmse,
)
from lstmforecast.evaluation.verdict import Verdict, derive_verdict

__all__ = [
    "ForecastMetrics",
    "Verdict",
    "deflated_sharpe_ratio",
    "derive_verdict",
    "diebold_mariano",
    "directional_accuracy",
    "forecast_metrics",
    "hac_standard_error",
    "mae",
    "mase_vs_persistence",
    "probabilistic_sharpe_ratio",
    "rmse",
]
