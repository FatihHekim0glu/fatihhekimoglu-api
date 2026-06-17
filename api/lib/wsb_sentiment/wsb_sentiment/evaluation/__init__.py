"""Honest-statistics layer: DSR/PSR, PBO/CSCV, HAC, Memmel-JK, bootstrap CI, verdict.

The headline ``signal_has_edge`` verdict is a pure function of the inference
outputs assembled here. Importing this subpackage has no side effects.
"""

from __future__ import annotations

from wsb_sentiment.evaluation.bootstrap_ci import stationary_bootstrap_ci
from wsb_sentiment.evaluation.dsr import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from wsb_sentiment.evaluation.hac import andrews_lag, newey_west_se
from wsb_sentiment.evaluation.memmel import memmel_test
from wsb_sentiment.evaluation.pbo import pbo_cscv
from wsb_sentiment.evaluation.results import (
    BootstrapCI,
    CPCVResult,
    DSRResult,
    MemmelResult,
    PBOResult,
)
from wsb_sentiment.evaluation.stats import (
    HonestStats,
    compute_honest_stats,
    effective_n_trials,
    hac_tstat,
)
from wsb_sentiment.evaluation.verdict import Verdict, derive_verdict

__all__ = [
    "BootstrapCI",
    "CPCVResult",
    "DSRResult",
    "HonestStats",
    "MemmelResult",
    "PBOResult",
    "Verdict",
    "andrews_lag",
    "compute_honest_stats",
    "deflated_sharpe_ratio",
    "derive_verdict",
    "effective_n_trials",
    "hac_tstat",
    "memmel_test",
    "newey_west_se",
    "pbo_cscv",
    "probabilistic_sharpe_ratio",
    "stationary_bootstrap_ci",
]
