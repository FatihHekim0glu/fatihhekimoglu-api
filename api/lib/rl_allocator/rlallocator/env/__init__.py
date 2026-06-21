"""Env subpackage: the causal multi-asset portfolio env + vectorized backtester + parity oracle.

- :mod:`rlallocator.env.portfolio_env` — the gymnasium-compatible MULTI-ASSET
  portfolio environment (weight-simplex action, strictly next-bar reward);
- :mod:`rlallocator.env.backtester` — the VECTORIZED multi-asset equity-curve
  evaluator over a weight path (pure numpy);
- :mod:`rlallocator.env.parity` — the parity oracle asserting the vectorized
  backtester equals the step-by-step env rollout to 1e-10 (the look-ahead catch),
  with a deliberately-leaky negative control it must catch.

Importing this subpackage has no side effects (gymnasium is imported lazily inside
:meth:`PortfolioEnv.as_gym_env`).
"""

from __future__ import annotations
