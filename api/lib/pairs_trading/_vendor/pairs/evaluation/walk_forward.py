"""Anchored walk-forward evaluation harness.

At each fold the training partition is the entire history up to the
fold boundary (anchored / expanding), and the test partition is the
next ``test_period``. Training observations whose label horizon would
overlap the test set are purged, and the post-test embargo window is
removed from any subsequent training set.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from pairs._exceptions import InputError
from pairs._rng import default_rng

from ._purge import embargo_indices, purge_indices
from .bootstrap_ci import stationary_bootstrap_ci
from .results import WalkForwardResult

__all__ = ["walk_forward_anchored"]

PairSelector = Callable[[pd.DataFrame], Any]
PairBacktester = Callable[[pd.DataFrame, Any], pd.Series]


def _sharpe_of(arr: np.ndarray) -> float:
    finite = arr[np.isfinite(arr)]
    if finite.size < 2:
        return float("nan")
    std = float(finite.std(ddof=1))
    if std <= 0.0:
        return float("nan")
    return float(finite.mean() / std)


def walk_forward_anchored(
    prices: pd.DataFrame,
    *,
    train_min_years: float = 3.0,
    test_period: str = "63D",
    step: str = "63D",
    purge_days: int = 10,
    embargo_pct: float = 0.01,
    pair_selector: PairSelector,
    pair_backtester: PairBacktester,
    rng: np.random.Generator | None = None,
    bootstrap_replicates: int = 1000,
) -> WalkForwardResult:
    """Run an anchored walk-forward backtest.

    Parameters
    ----------
    prices : pandas.DataFrame
        Wide price panel indexed by trading dates.
    train_min_years : float, default ``3.0``
        Minimum history (in years of 365.25 days) before the first fold.
    test_period : str, default ``"63D"``
        Pandas frequency string for the OOS window length.
    step : str, default ``"63D"``
        Step between consecutive fold anchors.
    purge_days : int, default ``10``
        Label-horizon length (calendar days) to purge from training.
    embargo_pct : float, default ``0.01``
        Embargo length as a fraction of the full sample, converted to
        calendar days via the index span.
    pair_selector : callable
        ``(train_prices) -> selection``. The returned object is opaque
        to the harness and forwarded to ``pair_backtester``.
    pair_backtester : callable
        ``(test_prices, selection) -> pandas.Series`` of OOS returns.
    rng : numpy.random.Generator, optional
        Source of randomness for the OOS bootstrap. Defaults to
        :func:`pairs.default_rng`.
    bootstrap_replicates : int, default ``1000``
        Replicates for the OOS Sharpe confidence interval.

    Returns
    -------
    WalkForwardResult
        Concatenated OOS series, fold metadata and Sharpe CI.
    """
    if not isinstance(prices, pd.DataFrame):
        raise InputError("prices must be a pandas DataFrame")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise InputError("prices.index must be a pandas.DatetimeIndex")
    if not prices.index.is_monotonic_increasing:
        raise InputError("prices index must be sorted ascending")
    if train_min_years <= 0:
        raise InputError("train_min_years must be positive")
    if purge_days < 0:
        raise InputError("purge_days must be non-negative")
    if not (0.0 <= embargo_pct < 1.0):
        raise InputError("embargo_pct must lie in [0, 1)")

    idx: pd.DatetimeIndex = prices.index
    span_days = max(int((idx[-1] - idx[0]).days), 1)
    embargo_days = int(round(embargo_pct * span_days))

    test_delta = pd.Timedelta(test_period)
    step_delta = pd.Timedelta(step)
    train_anchor = idx[0] + pd.Timedelta(days=int(round(train_min_years * 365.25)))

    fold_oos: list[pd.Series] = []
    fold_starts: list[pd.Timestamp] = []
    fold_ends: list[pd.Timestamp] = []
    fold_count = 0
    cursor = train_anchor
    while cursor + test_delta <= idx[-1] + pd.Timedelta(days=1):
        train_mask = idx <= cursor
        test_mask = (idx > cursor) & (idx <= cursor + test_delta)
        train_idx = idx[train_mask]
        test_idx = idx[test_mask]
        if len(train_idx) == 0 or len(test_idx) == 0:
            cursor = cursor + step_delta
            continue
        purged_train = purge_indices(train_idx, test_idx, label_horizon_days=int(purge_days))
        # Embargo only affects observations *after* the test window, which is
        # irrelevant for the anchored expanding scheme; we still compute it so
        # that callers can audit the dropped set.
        _ = embargo_indices(idx, test_idx, embargo_days=int(embargo_days))
        train_prices = prices.loc[purged_train]
        test_prices = prices.loc[test_idx]
        if train_prices.shape[0] < 2 or test_prices.shape[0] < 1:
            cursor = cursor + step_delta
            continue
        selection = pair_selector(train_prices)
        oos = pair_backtester(test_prices, selection)
        if not isinstance(oos, pd.Series):
            raise InputError("pair_backtester must return a pandas Series")
        if not oos.empty:
            fold_oos.append(oos)
            fold_starts.append(test_idx[0])
            fold_ends.append(test_idx[-1])
            fold_count += 1
        cursor = cursor + step_delta

    if fold_oos:
        oos_returns = pd.concat(fold_oos).sort_index()
        oos_returns = oos_returns[~oos_returns.index.duplicated(keep="first")]
    else:
        oos_returns = pd.Series(dtype=float)

    if oos_returns.size >= 4 and oos_returns.std(ddof=1) > 0.0:
        gen = rng if rng is not None else default_rng()
        ci = stationary_bootstrap_ci(
            oos_returns,
            _sharpe_of,
            alpha=0.05,
            n_boot=int(bootstrap_replicates),
            rng=gen,
        )
        sr = _sharpe_of(oos_returns.to_numpy(dtype=float))
        ci_low, ci_high = ci.ci_low, ci.ci_high
    else:
        sr = float("nan") if oos_returns.empty else _sharpe_of(oos_returns.to_numpy(dtype=float))
        ci_low = float("-inf")
        ci_high = float("inf")

    return WalkForwardResult(
        oos_returns=oos_returns,
        fold_count=fold_count,
        fold_starts=tuple(fold_starts),
        fold_ends=tuple(fold_ends),
        oos_sharpe=float(sr) if np.isfinite(sr) else float("nan"),
        sharpe_ci_low=float(ci_low),
        sharpe_ci_high=float(ci_high),
        purge_days=int(purge_days),
        embargo_days=int(embargo_days),
    )
