"""Honest-statistics layer: DSR/PSR, Sharpe-difference inference, and verdicts.

The headline verdict is a pure function of the inference outputs. Importing this
subpackage has no side effects.
"""

from __future__ import annotations

from stockclusters.evaluation.comparison import (
    ComparisonResult,
    block_bootstrap_sharpe_gap,
    jobson_korkie_memmel,
)
from stockclusters.evaluation.dsr import deflated_sharpe_ratio, probabilistic_sharpe_ratio
from stockclusters.evaluation.verdict import (
    ClusteringVerdict,
    derive_clustering_verdict,
)

__all__ = [
    "ClusteringVerdict",
    "ComparisonResult",
    "block_bootstrap_sharpe_gap",
    "deflated_sharpe_ratio",
    "derive_clustering_verdict",
    "jobson_korkie_memmel",
    "probabilistic_sharpe_ratio",
]
