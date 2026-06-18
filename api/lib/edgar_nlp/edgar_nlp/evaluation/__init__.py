"""Statistical evaluation: Deflated Sharpe and HAC standard errors.

The honest-yardstick layer for the tone-spread panel study. The Deflated Sharpe
ratio (Bailey & Lopez de Prado, 2014) counts the full configuration grid as
``n_trials``; the Newey-West HAC standard error gives autocorrelation-robust
t-statistics for the overlapping-window tone spread. Importing this package has
no side effects.
"""

from __future__ import annotations

from edgar_nlp.evaluation.dsr import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from edgar_nlp.evaluation.hac import andrews_lag, newey_west_se

__all__ = [
    "andrews_lag",
    "deflated_sharpe_ratio",
    "newey_west_se",
    "probabilistic_sharpe_ratio",
]
