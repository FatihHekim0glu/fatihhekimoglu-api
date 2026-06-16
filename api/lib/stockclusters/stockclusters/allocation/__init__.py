"""Cluster-aware allocation schemes and the diversification horse race.

Three strategies are compared honestly out-of-sample:

- ``1/N`` — the naive equal-weight DeMiguel benchmark.
- ``cluster-EW`` — equal weight across clusters, equal weight within.
- ``stripped-HRP`` — inverse-variance within cluster, equal across clusters.

All weights are ``shift(1)``-applied at the rebalance boundary (no look-ahead). The
frozen :class:`DiversificationResult` bundles the OOS Sharpes and the inference
outputs (Memmel-JK p-value, DSR).

Importing this subpackage has no side effects.
"""

from __future__ import annotations

from stockclusters.allocation.schemes import (
    DiversificationResult,
    cluster_equal_weight,
    one_over_n_weights,
    run_diversification,
    stripped_hrp_weights,
)

__all__ = [
    "DiversificationResult",
    "cluster_equal_weight",
    "one_over_n_weights",
    "run_diversification",
    "stripped_hrp_weights",
]
