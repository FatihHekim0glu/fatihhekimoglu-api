"""One-sided, drift-adjusted portfolio turnover."""

from __future__ import annotations

import numpy as np


def compute_turnover(
    weights_new: np.ndarray,
    weights_prev: np.ndarray,
    returns_between: np.ndarray,
) -> float:
    """Return one-sided drift-adjusted turnover.

    Let ``w_prev`` be the weights at the start of the period, ``r`` the
    realized asset returns over that period and ``w_new`` the target
    weights set at the rebalance date. The portfolio drifts to

    ``w_drift = w_prev * (1 + r) / (1 + w_prev . r)``

    and the turnover charged on the rebalance is

    ``TO = 0.5 * sum(|w_new - w_drift|)``.
    """
    w_prev = np.asarray(weights_prev, dtype=float).ravel()
    w_new = np.asarray(weights_new, dtype=float).ravel()
    r = np.asarray(returns_between, dtype=float).ravel()
    if w_prev.shape != w_new.shape or w_prev.shape != r.shape:
        raise ValueError("weights_prev, weights_new and returns_between must align")
    denom = 1.0 + float(np.dot(w_prev, r))
    if denom == 0.0 or not np.isfinite(denom):
        # Catastrophic drawdown: charge full rebalance.
        return 0.5 * float(np.sum(np.abs(w_new - w_prev)))
    w_drift = w_prev * (1.0 + r) / denom
    return 0.5 * float(np.sum(np.abs(w_new - w_drift)))
