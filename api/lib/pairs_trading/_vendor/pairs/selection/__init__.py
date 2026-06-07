"""Pair-selection sub-package.

End-to-end candidate generation, pre-screen, cointegration screening with
multiple-testing correction, and the Romano-Wolf step-down procedure for
out-of-sample Sharpe comparisons.

Public entry points are re-exported here so consumers can do::

    from pairs.selection import generate_candidates, screen_cointegration

without reaching into sub-modules.
"""

from __future__ import annotations

from pairs.selection._hurst import hurst_exponent
from pairs.selection.candidates import generate_candidates
from pairs.selection.effective_n import estimate_effective_n
from pairs.selection.mtc import apply_mtc
from pairs.selection.pre_screen import apply_pre_screen
from pairs.selection.results import Candidate, RWResult, ScreenResult
from pairs.selection.romano_wolf import romano_wolf_stepdown
from pairs.selection.screen import screen_cointegration

__all__ = [
    "Candidate",
    "RWResult",
    "ScreenResult",
    "apply_mtc",
    "apply_pre_screen",
    "estimate_effective_n",
    "generate_candidates",
    "hurst_exponent",
    "romano_wolf_stepdown",
    "screen_cointegration",
]
