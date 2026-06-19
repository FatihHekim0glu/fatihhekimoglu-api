"""Env subpackage: the causal trading env, the vectorized backtester, and the parity oracle.

The trading env (:mod:`rltrader.env.trading_env`) is the step-by-step oracle; the
vectorized backtester (:mod:`rltrader.env.backtester`) is the fast equity-curve
evaluator; the parity module (:mod:`rltrader.env.parity`) asserts the two agree to
1e-10 (the load-bearing look-ahead guard). Importing this subpackage has no side
effects and pulls in nothing heavy (gymnasium is imported lazily inside
``TradingEnv.as_gym_env``).
"""

from __future__ import annotations
