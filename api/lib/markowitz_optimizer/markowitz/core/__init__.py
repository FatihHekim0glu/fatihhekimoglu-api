"""Mathematical core for Markowitz mean-variance optimization.

This subpackage provides:

* Numerically stable linear algebra primitives (``linalg``).
* Closed-form Merton scalars and analytic frontier (``merton_scalars``,
  ``frontier``).
* An immutable ``Portfolio`` value object (``portfolio``).
* A taxonomy of domain-specific exceptions (``exceptions``).
"""

from __future__ import annotations

from markowitz.core.exceptions import (
    CoreError,
    InfeasibleFrontierError,
    NumericalError,
    SingularCovarianceError,
)
from markowitz.core.frontier import AnalyticFrontier
from markowitz.core.linalg import (
    cholesky_solve,
    condition_number,
    psd_check,
    regularize,
)
from markowitz.core.merton_scalars import MertonABCD, compute_abcd
from markowitz.core.portfolio import Portfolio, portfolio_performance

__all__ = [
    "AnalyticFrontier",
    "CoreError",
    "InfeasibleFrontierError",
    "MertonABCD",
    "NumericalError",
    "Portfolio",
    "SingularCovarianceError",
    "cholesky_solve",
    "compute_abcd",
    "condition_number",
    "portfolio_performance",
    "psd_check",
    "regularize",
]
