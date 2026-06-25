"""quantcore — shared, torch-free honest-statistics primitives.

The single source of truth for the honest-stats primitives previously
reimplemented (with drift) across the ML+Finance portfolio. Dependencies are
``numpy``, ``scipy``, ``statsmodels`` ONLY — no torch / sklearn / onnx.

Public API (grouped by area):

Sharpe
    ``sharpe_ratio`` — per-observation (``periods_per_year=1``) AND annualized.

Probabilistic / Deflated Sharpe (Bailey & López de Prado)
    ``probabilistic_sharpe_ratio`` — PSR (FULL-kurtosis bracket).
    ``deflated_sharpe_ratio`` — DSR (extreme-value ``SR*_0`` benchmark).
    ``variance_of_trial_sharpes`` — REAL cross-trial variance ``V`` helper.
    ``expected_sharpe_variance`` — analytic single-series ``V`` proxy.
    ``effective_n_trials`` — honest multiplicity count (product of swept axes).

HAC / Newey-West
    ``newey_west_lrv`` — long-run variance ``omega_hat``.
    ``newey_west_se`` — standard error of the mean ``sqrt(omega_hat / T)``.
    ``andrews_lag`` — Andrews (1991) automatic Bartlett lag.

Diebold-Mariano (HAC variance)
    ``diebold_mariano`` — ``(stat, pvalue)``; Normal default, opt-in t / HLN.
    ``dm_favours_a`` — significance + favourable-sign gate.

Jobson-Korkie-Memmel
    ``jobson_korkie_memmel`` — Sharpe-difference test p-value (Memmel 2003).

Probability of Backtest Overfitting (CSCV)
    ``probability_of_backtest_overfitting`` / ``PBOResult``.

Walk-forward
    ``purged_walk_forward_folds`` / ``WalkForwardFold`` — leakage-free folds.

Verdict
    ``edge_verdict`` / ``Verdict`` / ``VerdictResult`` — the canonical edge gate.

Errors
    ``QuantCoreError`` / ``ValidationError``.
"""

from __future__ import annotations

from quantcore._constants import EULER_MASCHERONI, PERIODS_PER_YEAR
from quantcore._exceptions import QuantCoreError, ValidationError
from quantcore.diebold_mariano import diebold_mariano, dm_favours_a
from quantcore.dsr import (
    deflated_sharpe_ratio,
    effective_n_trials,
    expected_sharpe_variance,
    probabilistic_sharpe_ratio,
    variance_of_trial_sharpes,
)
from quantcore.hac import andrews_lag, newey_west_lrv, newey_west_se
from quantcore.memmel import jobson_korkie_memmel
from quantcore.pbo import PBOResult, probability_of_backtest_overfitting
from quantcore.sharpe import sharpe_ratio
from quantcore.verdict import Verdict, VerdictResult, edge_verdict
from quantcore.walk_forward import WalkForwardFold, purged_walk_forward_folds

__version__ = "0.1.0"

__all__ = [
    # Sharpe
    "sharpe_ratio",
    # PSR / DSR + honest-input helpers
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
    "variance_of_trial_sharpes",
    "expected_sharpe_variance",
    "effective_n_trials",
    # HAC / Newey-West
    "newey_west_lrv",
    "newey_west_se",
    "andrews_lag",
    # Diebold-Mariano
    "diebold_mariano",
    "dm_favours_a",
    # Jobson-Korkie-Memmel
    "jobson_korkie_memmel",
    # PBO / CSCV
    "probability_of_backtest_overfitting",
    "PBOResult",
    # Walk-forward
    "purged_walk_forward_folds",
    "WalkForwardFold",
    # Verdict
    "edge_verdict",
    "Verdict",
    "VerdictResult",
    # Constants
    "PERIODS_PER_YEAR",
    "EULER_MASCHERONI",
    # Errors
    "QuantCoreError",
    "ValidationError",
]
