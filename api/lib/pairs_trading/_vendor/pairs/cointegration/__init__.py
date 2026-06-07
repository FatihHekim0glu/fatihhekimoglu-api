"""Cointegration test suite for pairs-trading.

This sub-package exposes the four standard cointegration diagnostics
(Engle-Granger, Johansen, KPSS and per-leg ADF/DF-GLS) plus a
stationary-bootstrap p-value and a thin pipeline that aggregates them.
All public functions return frozen, slotted dataclasses defined in
:mod:`pairs.cointegration.results`.
"""

from __future__ import annotations

from .adf import unit_root_check
from .bootstrap import bootstrap_coint_pvalue
from .engle_granger import engle_granger
from .johansen import johansen
from .kpss import kpss_spread
from .pipeline import full_pipeline
from .results import (
    BootstrapResult,
    CointegrationResult,
    JohansenResult,
    KPSSResult,
    PipelineResult,
    TestDirection,
    UnitRootResult,
)

__all__ = [
    "BootstrapResult",
    "CointegrationResult",
    "JohansenResult",
    "KPSSResult",
    "PipelineResult",
    "TestDirection",
    "UnitRootResult",
    "bootstrap_coint_pvalue",
    "engle_granger",
    "full_pipeline",
    "johansen",
    "kpss_spread",
    "unit_root_check",
]
