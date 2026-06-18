"""Honest-statistics evaluation layer for the cross-sectional graph comparison.

Everything here lives in RANKING space — the honest comparison for next-period
cross-sectional return prediction:

- :mod:`gnnstocks.evaluation.metrics` — per-period rank-IC (Spearman) + a
  long-short decile-spread return/Sharpe net of basis-point costs, with a HAC
  standard error on the IC/return differential;
- :mod:`gnnstocks.evaluation.diebold_mariano` — the Diebold-Mariano test on the
  per-period IC (or long-short return) differential of a GNN vs. the best
  baseline;
- :mod:`gnnstocks.evaluation.dsr` — the Probabilistic / Deflated Sharpe ratios
  (reused verbatim), with ``n_trials`` = graph-type x model x HP grid;
- :mod:`gnnstocks.evaluation.verdict` — the PURE ``gnn_beats_baseline`` verdict.

There is NO baseline-free in-sample R²/accuracy anywhere in this layer.
Importing this package has no side effects.
"""

from __future__ import annotations
