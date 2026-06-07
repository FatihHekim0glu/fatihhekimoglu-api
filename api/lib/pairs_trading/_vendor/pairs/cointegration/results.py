"""Immutable result containers for the cointegration sub-package.

All result classes are implemented as frozen, slotted dataclasses with
keyword-only constructors.  Each class performs lightweight validation
in :meth:`__post_init__` so that consumers can rely on the invariants
without re-checking them downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class TestDirection(StrEnum):
    """Which way the Engle-Granger regression was estimated.

    ``Y0_ON_Y1`` indicates that :math:`y_0` was regressed on :math:`y_1`,
    yielding a spread :math:`y_0 - \\beta y_1`.  ``Y1_ON_Y0`` is the
    mirror image.  Storing both directions lets callers reason about the
    sensitivity of the test to the choice of dependent variable.
    """

    Y0_ON_Y1 = "y0_on_y1"
    Y1_ON_Y0 = "y1_on_y0"


@dataclass(frozen=True, slots=True, kw_only=True)
class CointegrationResult:
    """Outcome of an Engle-Granger two-step cointegration test."""

    stat: float
    pvalue: float
    crit_values: tuple[float, float, float]
    direction_used: TestDirection
    beta: float
    alpha: float
    residuals: pd.Series
    autolag_used: str | None
    n_obs: int
    pvalue_other_direction: float
    method: str = "engle_granger"

    def __post_init__(self) -> None:
        if not np.isfinite(self.stat):
            raise ValueError("stat must be finite")
        if not (0.0 <= self.pvalue <= 1.0):
            raise ValueError(f"pvalue must be in [0, 1]; got {self.pvalue}")
        if not (0.0 <= self.pvalue_other_direction <= 1.0):
            raise ValueError("pvalue_other_direction must be in [0, 1]")
        if len(self.crit_values) != 3:
            raise ValueError("crit_values must contain exactly three thresholds")
        if self.n_obs <= 0:
            raise ValueError("n_obs must be positive")
        if not isinstance(self.residuals, pd.Series):
            raise TypeError("residuals must be a pandas Series")


@dataclass(frozen=True, slots=True, kw_only=True)
class JohansenResult:
    """Outcome of the Johansen rank test (trace and max-eigen statistics)."""

    trace_stats: NDArray[np.float64]
    trace_crit_95: NDArray[np.float64]
    max_eig_stats: NDArray[np.float64]
    max_eig_crit_95: NDArray[np.float64]
    rank: int
    eigenvectors: NDArray[np.float64]
    n_obs: int

    def __post_init__(self) -> None:
        if self.rank < 0:
            raise ValueError(f"rank must be non-negative; got {self.rank}")
        if self.n_obs <= 0:
            raise ValueError("n_obs must be positive")
        if self.trace_stats.shape != self.trace_crit_95.shape:
            raise ValueError("trace statistic / critical-value shape mismatch")
        if self.max_eig_stats.shape != self.max_eig_crit_95.shape:
            raise ValueError("max-eigenvalue statistic / critical-value shape mismatch")


@dataclass(frozen=True, slots=True, kw_only=True)
class KPSSResult:
    """Outcome of a KPSS stationarity test on a candidate spread."""

    stat: float
    pvalue: float
    pvalue_interpolated: bool
    crit_values: dict[str, float]
    nlags_used: int
    is_stationary: bool

    def __post_init__(self) -> None:
        if not np.isfinite(self.stat):
            raise ValueError("stat must be finite")
        if not (0.0 <= self.pvalue <= 1.0):
            raise ValueError("pvalue must be in [0, 1]")
        if self.nlags_used < 0:
            raise ValueError("nlags_used must be non-negative")


@dataclass(frozen=True, slots=True, kw_only=True)
class UnitRootResult:
    """Outcome of an I(1) integration check for a single leg."""

    leg_name: str
    levels_pvalue: float
    diff_pvalue: float
    is_i1: bool
    method: str
    n_obs: int

    def __post_init__(self) -> None:
        if not (0.0 <= self.levels_pvalue <= 1.0):
            raise ValueError("levels_pvalue must be in [0, 1]")
        if not (0.0 <= self.diff_pvalue <= 1.0):
            raise ValueError("diff_pvalue must be in [0, 1]")
        if self.n_obs <= 0:
            raise ValueError("n_obs must be positive")


@dataclass(frozen=True, slots=True, kw_only=True)
class BootstrapResult:
    """Bootstrap-based p-value for the Engle-Granger statistic."""

    pvalue: float
    n_boot: int
    block_length: int
    observed_stat: float
    null_stat_quantiles: dict[float, float]

    def __post_init__(self) -> None:
        if not (0.0 <= self.pvalue <= 1.0):
            raise ValueError("pvalue must be in [0, 1]")
        if self.n_boot <= 0:
            raise ValueError("n_boot must be positive")
        if self.block_length <= 0:
            raise ValueError("block_length must be positive")


@dataclass(frozen=True, slots=True, kw_only=True)
class PipelineResult:
    """Aggregate outcome of the full cointegration pipeline."""

    eg: CointegrationResult
    johansen: JohansenResult | None
    kpss: KPSSResult
    leg0_unit_root: UnitRootResult
    leg1_unit_root: UnitRootResult
    bootstrap: BootstrapResult | None = field(default=None)
    cointegrated: bool
    alpha: float

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must lie in (0, 1); got {self.alpha}")
