"""Spread estimation, OU dynamics, z-scores and Kalman hedging.

This subpackage provides the core toolkit for turning two cointegrated price
series into a tradable spread:

* :func:`ols_hedge` / :func:`tls_hedge` -- static hedge ratio estimation.
* :func:`build_spread` -- assemble the spread series from a hedge ratio.
* :func:`fit_ou` / :func:`half_life` -- continuous-time mean reversion dynamics.
* :func:`zscore` -- rolling or stationary z-score with leak-free semantics.
* :class:`KalmanHedge` -- time-varying hedge via Kalman filtering.
* :func:`ou_diagnostics` -- battery of statistical sanity checks.

All public estimators return frozen dataclasses defined in
:mod:`pairs.spread.results` so downstream consumers can rely on stable attribute
access and immutability.
"""

from __future__ import annotations

from pairs.spread.diagnostics import ou_diagnostics
from pairs.spread.half_life import half_life
from pairs.spread.hedge import ols_hedge, tls_hedge
from pairs.spread.kalman import KalmanHedge
from pairs.spread.ou import fit_ou
from pairs.spread.results import (
    HalfLifeResult,
    HedgeResult,
    KalmanHedgeResult,
    OUDiagnostics,
    OUResult,
)
from pairs.spread.spread import build_spread
from pairs.spread.zscore import zscore

__all__ = [
    "HalfLifeResult",
    "HedgeResult",
    "KalmanHedge",
    "KalmanHedgeResult",
    "OUDiagnostics",
    "OUResult",
    "build_spread",
    "fit_ou",
    "half_life",
    "ols_hedge",
    "ou_diagnostics",
    "tls_hedge",
    "zscore",
]
