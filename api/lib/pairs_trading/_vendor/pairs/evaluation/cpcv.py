"""Combinatorial Purged Cross-Validation (Lopez de Prado 2018, ch. 12).

The sample is divided into ``N`` contiguous groups. For every choice of
``k_test`` test groups (``C(N, k_test)`` combinations) the remaining
``N - k_test`` groups are used for training (with purge / embargo). The
test predictions are reassembled into ``C(N, k_test) * k_test / N``
synthetic paths so that each path corresponds to a full traversal of
the sample using one test slice per group.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from pairs._exceptions import InputError

from ._purge import embargo_indices, purge_indices
from .results import CPCVResult

__all__ = ["cpcv_paths"]

PairSelector = Callable[[pd.DataFrame], Any]
PairBacktester = Callable[[pd.DataFrame, Any], pd.Series]


def _sharpe(series: pd.Series) -> float:
    finite = series.dropna()
    if finite.size < 2:
        return float("nan")
    std = float(finite.std(ddof=1))
    if std <= 0.0:
        return float("nan")
    return float(finite.mean() / std)


def cpcv_paths(
    prices: pd.DataFrame,
    *,
    n_groups: int = 10,
    k_test: int = 2,
    purge_days: int = 10,
    embargo_pct: float = 0.01,
    pair_selector: PairSelector,
    pair_backtester: PairBacktester,
    rng: np.random.Generator | None = None,  # noqa: ARG001 - reserved for future use
) -> CPCVResult:
    """Run combinatorial purged cross-validation and reassemble paths.

    Parameters
    ----------
    prices : pandas.DataFrame
        Wide price panel indexed by trading dates.
    n_groups : int, default ``10``
        Number of contiguous groups.
    k_test : int, default ``2``
        Number of test groups per combination.
    purge_days : int, default ``10``
        Label horizon (calendar days) for purging.
    embargo_pct : float, default ``0.01``
        Embargo fraction of the full sample.
    pair_selector, pair_backtester : callable
        See :func:`walk_forward_anchored` for the contract.
    rng : numpy.random.Generator, optional
        Reserved for future stochastic extensions.

    Returns
    -------
    CPCVResult
        Path return series, per-path Sharpe and combination counts.
    """
    if not isinstance(prices, pd.DataFrame):
        raise InputError("prices must be a pandas DataFrame")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise InputError("prices.index must be a pandas.DatetimeIndex")
    if n_groups <= 1:
        raise InputError("n_groups must exceed 1")
    if not (0 < k_test < n_groups):
        raise InputError("k_test must satisfy 0 < k_test < n_groups")

    idx = prices.index
    t = len(idx)
    if t < n_groups * 2:
        raise InputError("not enough observations for the requested group count")

    span_days = max(int((idx[-1] - idx[0]).days), 1)
    embargo_days = int(round(embargo_pct * span_days))

    bounds = np.linspace(0, t, n_groups + 1, dtype=int)
    group_slices: list[pd.DatetimeIndex] = [idx[bounds[g] : bounds[g + 1]] for g in range(n_groups)]

    combos = list(combinations(range(n_groups), int(k_test)))
    n_combos = len(combos)
    # Path count formula (Lopez de Prado 2018): C(N, k) * k / N.
    n_paths = n_combos * k_test // n_groups
    if n_paths <= 0:  # pragma: no cover - defensive
        raise InputError("CPCV produced zero paths; check n_groups/k_test")

    # For each (combination, slot-within-combo) the produced test prediction
    # belongs to one of the synthetic paths. The mapping below tracks, for
    # each group g, how many times it has appeared as a test group; that
    # counter becomes its path index for that combination.
    group_appearance: list[int] = [0] * n_groups
    path_segments: dict[int, list[pd.Series]] = {p: [] for p in range(n_paths)}

    for combo in combos:
        test_groups = list(combo)
        test_index = pd.DatetimeIndex(
            np.concatenate([group_slices[g].to_numpy() for g in test_groups])
        )
        train_groups = [g for g in range(n_groups) if g not in set(test_groups)]
        train_index = pd.DatetimeIndex(
            np.concatenate([group_slices[g].to_numpy() for g in train_groups])
        )
        train_index = train_index.sort_values()
        purged = purge_indices(train_index, test_index, label_horizon_days=int(purge_days))
        embargoed = embargo_indices(purged, test_index, embargo_days=int(embargo_days))
        train_final = purged.difference(embargoed)
        if len(train_final) < 2 or len(test_index) < 1:
            continue
        train_prices = prices.loc[train_final]
        selection = pair_selector(train_prices)
        # Backtest each test group separately so the segment can be routed.
        for g in test_groups:
            group_idx = group_slices[g]
            if len(group_idx) == 0:
                continue
            segment = pair_backtester(prices.loc[group_idx], selection)
            if not isinstance(segment, pd.Series):
                raise InputError("pair_backtester must return a pandas Series")
            path_id = group_appearance[g] % n_paths
            group_appearance[g] += 1
            if not segment.empty:
                path_segments[path_id].append(segment)

    paths: list[pd.Series] = []
    sharpes: list[float] = []
    for p in range(n_paths):
        if not path_segments[p]:
            continue
        path_series = pd.concat(path_segments[p]).sort_index()
        path_series = path_series[~path_series.index.duplicated(keep="first")]
        paths.append(path_series)
        sharpes.append(_sharpe(path_series))

    finite_sharpes = [s for s in sharpes if np.isfinite(s)]
    median = float(np.median(finite_sharpes)) if finite_sharpes else float("nan")

    return CPCVResult(
        paths=tuple(paths),
        n_groups=int(n_groups),
        k_test=int(k_test),
        n_combinations=int(n_combos),
        path_sharpes=tuple(sharpes),
        median_path_sharpe=median,
    )
