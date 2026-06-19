"""Evaluation layer: HAC SEs, multiple-testing corrections, the PURE verdict.

Importing this subpackage has no side effects.
"""

from __future__ import annotations

from fedcausal.evaluation.hac import andrews_lag, newey_west_se
from fedcausal.evaluation.multiple_testing import (
    MultipleTestingResult,
    benjamini_hochberg,
    romano_wolf,
)
from fedcausal.evaluation.verdict import (
    FedVerdict,
    VerdictInputs,
    derive_verdict,
    fed_effect_is_tradable,
)

__all__ = [
    "FedVerdict",
    "MultipleTestingResult",
    "VerdictInputs",
    "andrews_lag",
    "benjamini_hochberg",
    "derive_verdict",
    "fed_effect_is_tradable",
    "newey_west_se",
    "romano_wolf",
]
