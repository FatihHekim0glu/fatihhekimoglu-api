"""markowitz-optimizer -- research-grade mean-variance portfolio toolkit.

Public entry point. Exposes the installed package version via
:pydata:`markowitz.__version__` and re-exports the headline public API so
that ``from markowitz import MeanVariance`` works without users having to
remember submodule paths.
"""

from __future__ import annotations

import contextlib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

__version__: str = "0.0.0"

with contextlib.suppress(PackageNotFoundError):
    # Running from a source checkout that has not been installed will raise;
    # the sentinel "0.0.0" above is the fallback.
    __version__ = _pkg_version("markowitz-optimizer")

# Headline re-exports (one-import-deep public API).
from markowitz.backtest import OneOverN, WalkForward
from markowitz.core import AnalyticFrontier, Portfolio
from markowitz.estimators import (
    CAPMReturns,
    EWMACovariance,
    EWMAMean,
    ImpliedReturns,
    JorionBayesStein,
    LedoitWolfShrinkage,
    SampleCovariance,
    SampleMean,
)
from markowitz.optimizer import (
    CustomConstraint,
    InfeasibleError,
    LeverageCap,
    LongOnly,
    MeanVariance,
    OptimizationError,
    SectorCap,
    SolverError,
    TurnoverCap,
    UnboundedError,
    WeightBounds,
)
from markowitz.views import (
    AbsoluteView,
    BlackLitterman,
    RelativeView,
    Views,
    implied_returns,
    infer_delta,
)

__all__ = [
    "AbsoluteView",
    "AnalyticFrontier",
    "BlackLitterman",
    "CAPMReturns",
    "CustomConstraint",
    "EWMACovariance",
    "EWMAMean",
    "ImpliedReturns",
    "InfeasibleError",
    "JorionBayesStein",
    "LedoitWolfShrinkage",
    "LeverageCap",
    "LongOnly",
    "MeanVariance",
    "OneOverN",
    "OptimizationError",
    "Portfolio",
    "RelativeView",
    "SampleCovariance",
    "SampleMean",
    "SectorCap",
    "SolverError",
    "TurnoverCap",
    "UnboundedError",
    "Views",
    "WalkForward",
    "WeightBounds",
    "__version__",
    "implied_returns",
    "infer_delta",
]
