"""Immutable result containers for the evaluation sub-package.

All classes are frozen, slotted dataclasses with keyword-only constructors.
Light validation in ``__post_init__`` enforces invariants that downstream
consumers are entitled to assume (finite scalars, probabilities in ``[0, 1]``,
non-negative counts). Each carries a JSON-serializable :meth:`to_dict` so the
result crosses the API boundary cleanly.

Importing this module has no side effects.

quantcore-candidate: trimmed from
pairs-trading:evaluation/results.py (CPCV / PBO / Memmel / Bootstrap / DSR).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


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

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` (path series dropped)."""
        return {
            "n_groups": int(self.n_groups),
            "k_test": int(self.k_test),
            "n_combinations": int(self.n_combinations),
            "path_sharpes": [float(s) for s in self.path_sharpes],
            "median_path_sharpe": float(self.median_path_sharpe),
        }


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

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        return {
            "realized_sr": float(self.realized_sr),
            "deflated_threshold": float(self.deflated_threshold),
            "psr_of_threshold": float(self.psr_of_threshold),
            "dsr": float(self.dsr),
            "p_value": float(self.p_value),
            "n_trials_effective": float(self.n_trials_effective),
            "sample_size": int(self.sample_size),
        }


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

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` (per-split logits dropped)."""
        return {
            "pbo": float(self.pbo),
            "n_splits": int(self.n_splits),
            "s_partitions": int(self.s_partitions),
        }


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

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        return {
            "sr_a": float(self.sr_a),
            "sr_b": float(self.sr_b),
            "z_stat": float(self.z_stat),
            "p_value": float(self.p_value),
            "n_obs": int(self.n_obs),
            "correlation": float(self.correlation),
        }


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

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        return {
            "point_estimate": float(self.point_estimate),
            "ci_low": float(self.ci_low),
            "ci_high": float(self.ci_high),
            "alpha": float(self.alpha),
            "n_boot": int(self.n_boot),
            "expected_block": int(self.expected_block),
        }
