"""HRP tree-clustering stages: distance, linkage, quasi-diagonalization.

The genuinely novel HRP code lives here (parity-validated against PyPortfolioOpt
and a second reference). Importing this subpackage has no side effects.
"""

from __future__ import annotations

from hrp.cluster.distance import correl_dist, euclidean_codistance
from hrp.cluster.linkage import linkage_matrix
from hrp.cluster.quasidiag import get_quasi_diag

__all__ = [
    "correl_dist",
    "euclidean_codistance",
    "get_quasi_diag",
    "linkage_matrix",
]
