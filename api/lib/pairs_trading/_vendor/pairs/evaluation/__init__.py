"""Evaluation harness for the pairs-trading library.

The submodules in this package implement the rigorous out-of-sample
evaluation toolkit used to certify a strategy before publication:
walk-forward + combinatorial purged cross-validation for path
construction, deflated Sharpe / PSR for multiple-testing correction,
Memmel's z-test for pairwise Sharpe equality, Hansen's SPA for the
"any model better than the benchmark" question, HAC standard errors,
stationary-bootstrap confidence intervals, the CSCV Probability of
Backtest Overfitting and a JSON-backed trial log that enforces OOS-
reuse discipline.

All public classes and functions are re-exported from this module so
that callers can rely on a flat import surface, e.g.::

    from pairs.evaluation import EvaluationProtocol, TrialLog
"""

from __future__ import annotations

from .bootstrap_ci import stationary_bootstrap_ci
from .cpcv import cpcv_paths
from .dsr import deflated_sharpe_ratio, effective_n_trials, probabilistic_sharpe_ratio
from .hac import andrews_lag, newey_west_se
from .memmel import memmel_test
from .pbo import pbo_cscv
from .protocol import EvaluationProtocol
from .results import (
    BootstrapCI,
    CPCVResult,
    DSRResult,
    MemmelResult,
    PBOResult,
    ProtocolReport,
    SPAResult,
    WalkForwardResult,
)
from .spa import hansen_spa
from .splits import IsOosSplit
from .trial_log import TrialLog
from .walk_forward import walk_forward_anchored

__all__ = [
    "BootstrapCI",
    "CPCVResult",
    "DSRResult",
    "EvaluationProtocol",
    "IsOosSplit",
    "MemmelResult",
    "PBOResult",
    "ProtocolReport",
    "SPAResult",
    "TrialLog",
    "WalkForwardResult",
    "andrews_lag",
    "cpcv_paths",
    "deflated_sharpe_ratio",
    "effective_n_trials",
    "hansen_spa",
    "memmel_test",
    "newey_west_se",
    "pbo_cscv",
    "probabilistic_sharpe_ratio",
    "stationary_bootstrap_ci",
    "walk_forward_anchored",
]
