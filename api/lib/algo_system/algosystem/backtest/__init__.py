"""Backtest subpackage: the vectorized no-lookahead engine + the bar-finality guard.

Exposes the reused transaction-cost model and walk-forward scaffold (copied from
the HRP infra) plus the algo-system-specific vectorized backtester
(:mod:`algosystem.backtest.engine`) and the only-act-on-closed-bars guard
(:mod:`algosystem.backtest.bar_finality`). Importing this subpackage has no side
effects.
"""

from __future__ import annotations
