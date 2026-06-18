"""Honest inference layer: return-space metrics, DM, DSR, and the verdict.

The headline ``deep_beats_naive`` verdict is a PURE FUNCTION of the inference
outputs (the Diebold-Mariano p-value/sign and the Deflated Sharpe). It cannot
read ``True`` unless a deep model genuinely beats the naive baseline AND the
difference is statistically significant AND the DSR clears zero. There is NO
price-level R² anywhere. Importing this subpackage has no side effects.
"""

from __future__ import annotations

from mvtsforecast.evaluation.diebold_mariano import diebold_mariano, dm_favours_model
from mvtsforecast.evaluation.dsr import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from mvtsforecast.evaluation.metrics import (
    ForecastMetrics,
    andrews_lag,
    directional_accuracy,
    forecast_metrics,
    hac_standard_error,
    mae,
    mase_vs_naive,
    net_pnl_sharpe,
    rmse,
)
from mvtsforecast.evaluation.verdict import Verdict, VerdictResult, derive_verdict

__all__ = [
    "ForecastMetrics",
    "Verdict",
    "VerdictResult",
    "andrews_lag",
    "deflated_sharpe_ratio",
    "derive_verdict",
    "diebold_mariano",
    "directional_accuracy",
    "dm_favours_model",
    "forecast_metrics",
    "hac_standard_error",
    "mae",
    "mase_vs_naive",
    "net_pnl_sharpe",
    "probabilistic_sharpe_ratio",
    "rmse",
]
