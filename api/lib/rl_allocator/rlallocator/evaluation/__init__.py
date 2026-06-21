"""Evaluation subpackage: the PURE honesty kernels.

The leakage-free, overfit-aware statistics that judge the RL allocator against the
baselines — all pure numpy / stdlib, no torch / sklearn / scipy at import or call,
so the serve path computes them live and torch-free:

- :mod:`rlallocator.evaluation.metrics` — OOS net Sharpe, max drawdown, turnover,
  net PnL, and the Newey-West HAC standard error (the DM denominator);
- :mod:`rlallocator.evaluation.diebold_mariano` — the DM test of the RL allocator
  vs. the best baseline on the per-bar net-return differential;
- :mod:`rlallocator.evaluation.dsr` — the Probabilistic / Deflated Sharpe ratios;
- :mod:`rlallocator.evaluation.pbo` — the CSCV Probability of Backtest Overfitting;
- :mod:`rlallocator.evaluation.seed_lottery` — the across-seed OOS-Sharpe dispersion;
- :mod:`rlallocator.evaluation.verdict` — the PURE ``rl_beats_baselines`` verdict.

Importing this subpackage has no side effects.
"""

from __future__ import annotations
