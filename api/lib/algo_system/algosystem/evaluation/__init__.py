"""Evaluation subpackage: OOS metrics, PBO/CSCV, Diebold-Mariano, DSR, and the verdict.

Exposes the honest-statistics layer used to judge the strategy out-of-sample net of
costs:

- :mod:`algosystem.evaluation.metrics` — OOS net Sharpe, max drawdown, turnover, net
  PnL;
- :mod:`algosystem.evaluation.pbo` — Probability of Backtest Overfitting via CSCV;
- :mod:`algosystem.evaluation.diebold_mariano` — the DM test of the system-vs-buy-hold
  per-bar net-return differential (fully implemented PURE kernel);
- :mod:`algosystem.evaluation.dsr` — the Probabilistic / Deflated Sharpe ratios
  (reused verbatim from the HRP infra);
- :mod:`algosystem.evaluation.hac` — the Newey-West HAC standard error (the DM
  denominator);
- :mod:`algosystem.evaluation.verdict` — the PURE ``system_has_edge`` verdict
  (fully implemented; gated at DM-significance AND DSR > 1-alpha AND PBO < 0.5).

Importing this subpackage has no side effects.
"""

from __future__ import annotations
