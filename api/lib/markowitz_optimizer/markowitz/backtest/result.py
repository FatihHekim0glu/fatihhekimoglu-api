"""Container for walk-forward backtest output."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import stats as _stats


@dataclass
class BacktestResult:
    """Aggregated output of a :class:`WalkForward` run.

    Attributes
    ----------
    returns_gross:
        Per-period gross returns, one column per strategy.
    returns_net:
        Per-period net-of-cost returns.
    weights:
        Mapping ``strategy_name -> DataFrame`` of post-rebalance weights.
    turnover:
        Per-period turnover, one column per strategy.
    rebalance_dates:
        Sorted list of dates on which weights were recomputed.
    """

    returns_gross: pd.DataFrame
    returns_net: pd.DataFrame
    weights: dict[str, pd.DataFrame]
    turnover: pd.DataFrame
    rebalance_dates: list[pd.Timestamp] = field(default_factory=list)

    def summary(self, *, ann: int = 12, gamma: float = 5.0) -> pd.DataFrame:
        """Standard performance grid (Sharpe, Sortino, MDD, Calmar, CEQ, TO)."""
        rows: list[dict[str, float]] = []
        for col in self.returns_net.columns:
            r = self.returns_net[col].dropna()
            rows.append(
                {
                    "Sharpe": _stats.sharpe_ratio(r, ann=ann),
                    "Sortino": _stats.sortino_ratio(r, ann=ann),
                    "MaxDD": _stats.max_drawdown(r),
                    "Calmar": _stats.calmar(r, ann=ann),
                    "CEQ": _stats.ceq(r, gamma=gamma),
                    "Turnover": float(self.turnover[col].mean()) if col in self.turnover else 0.0,
                }
            )
        return pd.DataFrame(rows, index=list(self.returns_net.columns))
