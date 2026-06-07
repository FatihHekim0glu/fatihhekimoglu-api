"""Portfolio-construction layer for the pairs-trading library.

This subpackage turns a collection of single-pair :class:`BacktestResult`
objects into a multi-pair book by stitching together allocators, hard caps,
correlation filtering, lifecycle bookkeeping, and risk overlays. The public
surface is intentionally narrow: callers normally only need
:func:`run_multi_pair_backtest` plus one of the :class:`Allocator`
implementations.
"""

from __future__ import annotations

from pairs.portfolio.allocators import (
    Allocator,
    EqualDollarAllocator,
    HRPAllocator,
    InverseVolAllocator,
)
from pairs.portfolio.caps import apply_caps
from pairs.portfolio.correlation import correlation_filter, effective_n
from pairs.portfolio.lifecycle import PairLifecycle
from pairs.portfolio.overlay import drawdown_killswitch, vol_target_overlay
from pairs.portfolio.results import (
    CapEvent,
    KillswitchEvent,
    OverlayConfig,
    PortfolioDiagnostics,
    PortfolioResult,
)
from pairs.portfolio.runner import run_multi_pair_backtest

__all__ = [
    "Allocator",
    "CapEvent",
    "EqualDollarAllocator",
    "HRPAllocator",
    "InverseVolAllocator",
    "KillswitchEvent",
    "OverlayConfig",
    "PairLifecycle",
    "PortfolioDiagnostics",
    "PortfolioResult",
    "apply_caps",
    "correlation_filter",
    "drawdown_killswitch",
    "effective_n",
    "run_multi_pair_backtest",
    "vol_target_overlay",
]
