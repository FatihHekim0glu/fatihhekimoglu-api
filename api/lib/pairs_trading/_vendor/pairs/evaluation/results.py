"""Immutable result containers for the evaluation sub-package.

All classes are frozen, slotted dataclasses with keyword-only constructors.
Light validation in ``__post_init__`` enforces invariants that downstream
consumers are entitled to assume (finite scalars, probabilities in ``[0, 1]``,
non-negative counts).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True, kw_only=True)
class WalkForwardResult:
    """Concatenated out-of-sample series from an anchored walk-forward run."""

    oos_returns: pd.Series
    fold_count: int
    fold_starts: tuple[pd.Timestamp, ...]
    fold_ends: tuple[pd.Timestamp, ...]
    oos_sharpe: float
    sharpe_ci_low: float
    sharpe_ci_high: float
    purge_days: int
    embargo_days: int

    def __post_init__(self) -> None:
        if self.fold_count < 0:
            raise ValueError("fold_count must be non-negative")
        if not isinstance(self.oos_returns, pd.Series):
            raise TypeError("oos_returns must be a pandas Series")
        if self.sharpe_ci_low > self.sharpe_ci_high:
            raise ValueError("sharpe_ci_low must be <= sharpe_ci_high")
        if self.purge_days < 0:
            raise ValueError("purge_days must be non-negative")
        if self.embargo_days < 0:
            raise ValueError("embargo_days must be non-negative")


@dataclass(frozen=True, slots=True, kw_only=True)
class CPCVResult:
    """Combinatorial purged cross-validation paths and aggregate metrics."""

    paths: tuple[pd.Series, ...]
    n_groups: int
    k_test: int
    n_combinations: int
    path_sharpes: tuple[float, ...]
    median_path_sharpe: float

    def __post_init__(self) -> None:
        if self.n_groups <= 0:
            raise ValueError("n_groups must be positive")
        if not (0 < self.k_test < self.n_groups):
            raise ValueError("k_test must satisfy 0 < k_test < n_groups")
        if self.n_combinations <= 0:
            raise ValueError("n_combinations must be positive")
        if len(self.path_sharpes) != len(self.paths):
            raise ValueError("path_sharpes length must match number of paths")


@dataclass(frozen=True, slots=True, kw_only=True)
class DSRResult:
    """Deflated Sharpe Ratio outputs (Bailey-Lopez de Prado 2014)."""

    realized_sr: float
    deflated_threshold: float
    psr_of_threshold: float
    dsr: float
    p_value: float
    n_trials_effective: float
    sample_size: int

    def __post_init__(self) -> None:
        if not (0.0 <= self.psr_of_threshold <= 1.0):
            raise ValueError("psr_of_threshold must lie in [0, 1]")
        if not (0.0 <= self.dsr <= 1.0):
            raise ValueError("dsr must lie in [0, 1]")
        if not (0.0 <= self.p_value <= 1.0):
            raise ValueError("p_value must lie in [0, 1]")
        if self.n_trials_effective < 1.0:
            raise ValueError("n_trials_effective must be >= 1")
        if self.sample_size <= 0:
            raise ValueError("sample_size must be positive")


@dataclass(frozen=True, slots=True, kw_only=True)
class PBOResult:
    """Combinatorially symmetric cross-validation probability of backtest overfitting."""

    pbo: float
    logit_lambdas: tuple[float, ...]
    n_splits: int
    s_partitions: int

    def __post_init__(self) -> None:
        if not (0.0 <= self.pbo <= 1.0):
            raise ValueError("pbo must lie in [0, 1]")
        if self.n_splits <= 0:
            raise ValueError("n_splits must be positive")
        if self.s_partitions <= 0 or self.s_partitions % 2 != 0:
            raise ValueError("s_partitions must be a positive even integer")


@dataclass(frozen=True, slots=True, kw_only=True)
class MemmelResult:
    """Memmel (2003) closed-form test of Sharpe-ratio equality."""

    sr_a: float
    sr_b: float
    z_stat: float
    p_value: float
    n_obs: int
    correlation: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.p_value <= 1.0):
            raise ValueError("p_value must lie in [0, 1]")
        if self.n_obs <= 0:
            raise ValueError("n_obs must be positive")
        if not (-1.0 - 1e-9 <= self.correlation <= 1.0 + 1e-9):
            raise ValueError("correlation must lie in [-1, 1]")


@dataclass(frozen=True, slots=True, kw_only=True)
class BootstrapCI:
    """Bootstrap confidence interval for a scalar statistic."""

    point_estimate: float
    ci_low: float
    ci_high: float
    alpha: float
    n_boot: int
    expected_block: int

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must lie in (0, 1)")
        if self.ci_low > self.ci_high:
            raise ValueError("ci_low must be <= ci_high")
        if self.n_boot <= 0:
            raise ValueError("n_boot must be positive")
        if self.expected_block <= 0:
            raise ValueError("expected_block must be positive")


@dataclass(frozen=True, slots=True, kw_only=True)
class SPAResult:
    """Hansen (2005) Superior Predictive Ability test results."""

    p_value_consistent: float
    p_value_lower: float
    p_value_upper: float
    best_model: str
    n_models: int
    n_boot: int

    def __post_init__(self) -> None:
        for label, value in (
            ("p_value_consistent", self.p_value_consistent),
            ("p_value_lower", self.p_value_lower),
            ("p_value_upper", self.p_value_upper),
        ):
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{label} must lie in [0, 1]")
        if self.n_models <= 0:
            raise ValueError("n_models must be positive")
        if self.n_boot <= 0:
            raise ValueError("n_boot must be positive")


@dataclass(frozen=True, slots=True, kw_only=True)
class ProtocolReport:
    """Aggregate evaluation report assembled by :class:`EvaluationProtocol`."""

    walk_forward: WalkForwardResult
    cpcv: CPCVResult
    dsr: DSRResult
    pbo: PBOResult | None
    memmel: MemmelResult | None
    spa: SPAResult | None
    hac_se: float
    is_oos_decay_pct: float
    broken_pair_count: int
    spec_hash: str
    trial_id: int
    seed: int
    metadata: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not np.isfinite(self.hac_se):
            raise ValueError("hac_se must be finite")
        if self.broken_pair_count < 0:
            raise ValueError("broken_pair_count must be non-negative")
        if self.trial_id < 0:
            raise ValueError("trial_id must be non-negative")
        if not self.spec_hash:
            raise ValueError("spec_hash must be a non-empty string")
