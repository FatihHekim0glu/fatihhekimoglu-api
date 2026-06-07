"""Frozen result containers and configuration objects for the portfolio layer.

Every public artefact returned by :mod:`pairs.portfolio` is one of the
dataclasses defined here. They are intentionally immutable (``frozen=True``)
and validate their own invariants in ``__post_init__`` so a malformed result
can never escape a constructor. Configuration objects use the same pattern so
that user-supplied risk-management knobs are sanity-checked at construction
time rather than deep inside a daily loop.

The classes here are deliberately free of behaviour: they only carry data so
that callers can serialise, log, or replay portfolio runs without re-importing
the heavy machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from pairs._exceptions import InputError

__all__ = [
    "CapEvent",
    "KillswitchEvent",
    "OverlayConfig",
    "PortfolioDiagnostics",
    "PortfolioResult",
]


CapKind = Literal["max_pairs", "sector_gross", "asset_notional", "reselection"]
"""Discriminator for the kind of cap that fired during weight projection."""


KillswitchTrigger = Literal["dd_threshold", "recover_half", "recover_full"]
"""State transitions emitted by the drawdown killswitch state machine."""


@dataclass(frozen=True, slots=True)
class CapEvent:
    """Single audit record describing a cap that was applied to a weight vector.

    Cap events are logged whenever the projection step alters at least one
    weight, including soft scaling operations. They are intentionally rich so
    that downstream tooling can reconstruct which constraint was binding.

    Attributes
    ----------
    asof : pandas.Timestamp
        Timestamp at which the cap fired.
    kind : {"max_pairs", "sector_gross", "asset_notional", "reselection"}
        Identifier of the constraint that triggered the event.
    pair_id : str
        Identifier of the affected pair, or the empty string for events that
        do not pertain to a single pair (such as walk-forward reselections).
    pre_weight : float
        Weight before the projection step.
    post_weight : float
        Weight after the projection step.
    detail : dict
        Free-form payload with constraint-specific diagnostics (sector key,
        scale factor, etc.).
    """

    asof: pd.Timestamp
    kind: CapKind
    pair_id: str
    pre_weight: float
    post_weight: float
    detail: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in {"max_pairs", "sector_gross", "asset_notional", "reselection"}:
            msg = f"unknown CapEvent kind: {self.kind!r}"
            raise InputError(msg)
        if not np.isfinite(float(self.pre_weight)):
            msg = f"pre_weight must be finite, got {self.pre_weight!r}"
            raise InputError(msg)
        if not np.isfinite(float(self.post_weight)):
            msg = f"post_weight must be finite, got {self.post_weight!r}"
            raise InputError(msg)


@dataclass(frozen=True, slots=True)
class KillswitchEvent:
    """Audit record describing a transition of the drawdown killswitch."""

    asof: pd.Timestamp
    trigger: KillswitchTrigger
    drawdown: float
    gross_before: float
    gross_after: float

    def __post_init__(self) -> None:
        if self.trigger not in {"dd_threshold", "recover_half", "recover_full"}:
            msg = f"unknown KillswitchEvent trigger: {self.trigger!r}"
            raise InputError(msg)
        if not np.isfinite(float(self.drawdown)):
            msg = "drawdown must be finite"
            raise InputError(msg)


@dataclass(frozen=True, slots=True)
class PortfolioDiagnostics:
    """Per-step portfolio-level diagnostics for a multi-pair backtest.

    Attributes
    ----------
    n_eff : pandas.Series
        Daily effective number of independent pairs after accounting for
        cross-pair correlation.
    sector_gross : pandas.DataFrame
        Daily gross exposure broken down by sector key. Columns are sector
        labels; values are sums of absolute pair weights.
    max_pair_corr : pandas.Series
        Largest pairwise correlation observed in the active set on each day.
    avg_active_count : float
        Average number of active pairs across the backtest.
    """

    n_eff: pd.Series
    sector_gross: pd.DataFrame
    max_pair_corr: pd.Series
    avg_active_count: float

    def __post_init__(self) -> None:
        if not isinstance(self.n_eff, pd.Series):
            msg = "n_eff must be a pandas Series"
            raise InputError(msg)
        if not isinstance(self.sector_gross, pd.DataFrame):
            msg = "sector_gross must be a pandas DataFrame"
            raise InputError(msg)
        if not isinstance(self.max_pair_corr, pd.Series):
            msg = "max_pair_corr must be a pandas Series"
            raise InputError(msg)
        if float(self.avg_active_count) < 0.0:
            msg = f"avg_active_count must be non-negative, got {self.avg_active_count!r}"
            raise InputError(msg)


@dataclass(frozen=True, slots=True)
class PortfolioResult:
    """Aggregate output of :func:`pairs.portfolio.run_multi_pair_backtest`.

    Attributes
    ----------
    equity : pandas.Series
        Cumulative equity curve, normalised to ``1.0`` at the start.
    returns : pandas.Series
        Net portfolio returns after overlay and killswitch.
    gross_history : pandas.Series
        Total gross exposure on each step (sum of absolute weights).
    weights_history : pandas.DataFrame
        Daily weight matrix; columns are pair ids.
    cap_events : list of :class:`CapEvent`
        Ordered audit log of every cap that fired during the run.
    killswitch_events : list of :class:`KillswitchEvent`
        Ordered audit log of every drawdown-killswitch state change.
    metrics : dict
        Summary metrics (annualised return, Sharpe, max drawdown, etc.).
    diagnostics : :class:`PortfolioDiagnostics`
        Per-step diagnostics for the run.
    """

    equity: pd.Series
    returns: pd.Series
    gross_history: pd.Series
    weights_history: pd.DataFrame
    cap_events: list[CapEvent]
    killswitch_events: list[KillswitchEvent]
    metrics: dict[str, float]
    diagnostics: PortfolioDiagnostics

    def __post_init__(self) -> None:
        if not isinstance(self.equity, pd.Series):
            msg = "equity must be a pandas Series"
            raise InputError(msg)
        if not isinstance(self.returns, pd.Series):
            msg = "returns must be a pandas Series"
            raise InputError(msg)
        if not isinstance(self.weights_history, pd.DataFrame):
            msg = "weights_history must be a pandas DataFrame"
            raise InputError(msg)
        if len(self.equity) != len(self.returns):
            msg = "equity and returns must have the same length"
            raise InputError(msg)


@dataclass(frozen=True, slots=True)
class OverlayConfig:
    """Tuning parameters for the volatility-target overlay and killswitch.

    Attributes
    ----------
    target_vol : float
        Annualised volatility target for the overlay multiplier.
    vol_window : int
        Rolling window (in bars) used to estimate realised volatility.
    vol_clip : tuple of float
        Lower and upper bounds applied to the overlay multiplier.
    dd_threshold : float
        Drawdown fraction (positive number) that trips the killswitch.
    dd_window : int
        Rolling window (in bars) used to compute the drawdown.
    ladder_days : int
        Bars of no new low required to step up to the next recovery rung.
    """

    target_vol: float = 0.08
    vol_window: int = 20
    vol_clip: tuple[float, float] = (0.5, 1.5)
    dd_threshold: float = 0.08
    dd_window: int = 20
    ladder_days: int = 10

    def __post_init__(self) -> None:
        if float(self.target_vol) <= 0.0:
            msg = f"target_vol must be positive, got {self.target_vol!r}"
            raise InputError(msg)
        if int(self.vol_window) <= 1:
            msg = f"vol_window must be > 1, got {self.vol_window!r}"
            raise InputError(msg)
        lo, hi = float(self.vol_clip[0]), float(self.vol_clip[1])
        if not (0.0 < lo <= hi):
            msg = f"vol_clip must satisfy 0 < lo <= hi, got {self.vol_clip!r}"
            raise InputError(msg)
        if not (0.0 < float(self.dd_threshold) < 1.0):
            msg = f"dd_threshold must lie in (0, 1), got {self.dd_threshold!r}"
            raise InputError(msg)
        if int(self.dd_window) <= 1:
            msg = f"dd_window must be > 1, got {self.dd_window!r}"
            raise InputError(msg)
        if int(self.ladder_days) <= 0:
            msg = f"ladder_days must be positive, got {self.ladder_days!r}"
            raise InputError(msg)
