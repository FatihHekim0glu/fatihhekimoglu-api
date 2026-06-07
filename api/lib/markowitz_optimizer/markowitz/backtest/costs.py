"""Transaction-cost models applied to gross backtest returns."""

from __future__ import annotations

import pandas as pd


def apply_transaction_costs(
    gross_returns: pd.Series,
    turnover: pd.Series,
    *,
    bps: float = 10.0,
) -> pd.Series:
    """Subtract proportional transaction costs from ``gross_returns``.

    Parameters
    ----------
    gross_returns:
        Per-period simple returns *before* costs.
    turnover:
        Per-period one-sided turnover, aligned to ``gross_returns``.
    bps:
        Round-trip-equivalent cost in basis points. The deduction applied
        to period ``t`` is ``(bps / 10_000) * turnover_t``.
    """
    rate = float(bps) / 10_000.0
    aligned_to = turnover.reindex(gross_returns.index).fillna(0.0)
    return gross_returns.sub(aligned_to.mul(rate))
