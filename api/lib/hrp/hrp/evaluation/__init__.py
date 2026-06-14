"""Honest-statistics layer: DSR/PSR, Sharpe-difference inference, and verdicts.

The headline verdict is a pure function of the inference outputs. Importing this
subpackage has no side effects.
"""

from __future__ import annotations

from hrp.evaluation.comparison import (
    ComparisonResult,
    block_bootstrap_sharpe_gap,
    jobson_korkie_memmel,
)
from hrp.evaluation.dsr import deflated_sharpe_ratio, probabilistic_sharpe_ratio
from hrp.evaluation.verdict import Verdict, derive_verdict

__all__ = [
    "ComparisonResult",
    "Verdict",
    "block_bootstrap_sharpe_gap",
    "deflated_sharpe_ratio",
    "derive_verdict",
    "jobson_korkie_memmel",
    "probabilistic_sharpe_ratio",
]
