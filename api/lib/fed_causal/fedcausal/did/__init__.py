"""Difference-in-differences layer: clustered DiD + honest heterogeneity framing.

Importing this subpackage has no side effects.
"""

from __future__ import annotations

from fedcausal.did.heterogeneity import (
    HeterogeneitySpread,
    describe_heterogeneity,
    heterogeneity_spread,
)
from fedcausal.did.model import (
    DiDResult,
    build_did_panel,
    estimate_did,
)

__all__ = [
    "DiDResult",
    "HeterogeneitySpread",
    "build_did_panel",
    "describe_heterogeneity",
    "estimate_did",
    "heterogeneity_spread",
]
