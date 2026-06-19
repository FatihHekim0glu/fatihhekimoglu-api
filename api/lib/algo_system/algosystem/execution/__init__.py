"""Execution subpackage: the SIMULATED paper broker + the backtest<->live parity oracle.

Exposes the bar-by-bar simulated paper-broker engine
(:mod:`algosystem.execution.paper_broker`) — next-bar-open fills + costs + slippage,
the "live" path — and the LOAD-BEARING backtest<->live parity oracle
(:mod:`algosystem.execution.parity`) that asserts the vectorized backtest equity
curve equals the paper-broker equity curve to ``1e-10``.

EXECUTION IS SIMULATED. There is no live Alpaca / broker connection and no broker
key: the paper broker is an internal cost+slippage simulator, never a live order
router. Importing this subpackage has no side effects.
"""

from __future__ import annotations
