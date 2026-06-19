"""Evaluation subpackage: OOS metrics, the seed lottery, Diebold-Mariano, DSR, verdict.

The metrics (:mod:`rltrader.evaluation.metrics`) compute OOS net Sharpe / drawdown
/ turnover / net PnL; the seed lottery (:mod:`rltrader.evaluation.seed_lottery`)
summarizes the across-seed Sharpe dispersion; Diebold-Mariano
(:mod:`rltrader.evaluation.diebold_mariano`) tests the RL-vs-buy-hold differential;
the DSR (:mod:`rltrader.evaluation.dsr`) deflates the Sharpe for the honest
seed x HP multiplicity; and the verdict (:mod:`rltrader.evaluation.verdict`) is the
PURE ``rl_beats_baseline`` gate. Importing this subpackage has no side effects.
"""

from __future__ import annotations
