"""Numerical convex-optimization layer for mean-variance portfolios.

This sub-package exposes the :class:`MeanVariance` optimizer (a thin wrapper
around CVXPY), a small DSL of pluggable constraints, and the
Cornuejols--Tutuncu reformulation used to recover the tangency portfolio as
a convex sub-problem.

The public surface is intentionally small; see the individual module
docstrings for the full contract.
"""

from __future__ import annotations

from markowitz.optimizer.constraints import (
    Constraint,
    ConstraintContext,
    CustomConstraint,
    LeverageCap,
    LongOnly,
    SectorCap,
    TurnoverCap,
    WeightBounds,
)
from markowitz.optimizer.cornuejols_tutuncu import (
    back_transform,
    detect_degeneracy,
    reformulate_max_sharpe,
)
from markowitz.optimizer.exceptions import (
    InfeasibleError,
    NumericalError,
    OptimizationError,
    SolverError,
    UnboundedError,
)
from markowitz.optimizer.mean_variance import MeanVariance
from markowitz.optimizer.solvers import SolverResult, solve_problem

__all__ = [
    "Constraint",
    "ConstraintContext",
    "CustomConstraint",
    "InfeasibleError",
    "LeverageCap",
    "LongOnly",
    "MeanVariance",
    "NumericalError",
    "OptimizationError",
    "SectorCap",
    "SolverError",
    "SolverResult",
    "TurnoverCap",
    "UnboundedError",
    "WeightBounds",
    "back_transform",
    "detect_degeneracy",
    "reformulate_max_sharpe",
    "solve_problem",
]
