"""Pair-level capital allocators.

An allocator turns a panel of spread P&Ls plus an "active" mask into a weight
vector that sums to one across active pairs. Allocators are deliberately
stateless and side-effect-free so they can be composed inside a daily loop or
re-used inside ensembles.

Three concrete allocators are provided:

* :class:`EqualDollarAllocator` -- the naive 1/N benchmark.
* :class:`InverseVolAllocator` -- inverse rolling-volatility weights.
* :class:`HRPAllocator` -- Lopez de Prado's Hierarchical Risk Parity.

All three implement the :class:`Allocator` protocol so they are interchangeable
at the call site.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from pairs._exceptions import InputError

__all__ = [
    "Allocator",
    "EqualDollarAllocator",
    "HRPAllocator",
    "InverseVolAllocator",
]


@runtime_checkable
class Allocator(Protocol):
    """Structural type for any object that can produce pair weights."""

    def weights(self, spread_pnls: pd.DataFrame, active_mask: pd.Series) -> pd.Series:
        """Return a weight vector indexed by the columns of ``spread_pnls``."""
        ...


def _validate_inputs(spread_pnls: pd.DataFrame, active_mask: pd.Series) -> None:
    if not isinstance(spread_pnls, pd.DataFrame):
        msg = "spread_pnls must be a pandas DataFrame"
        raise InputError(msg)
    if not isinstance(active_mask, pd.Series):
        msg = "active_mask must be a pandas Series"
        raise InputError(msg)
    if active_mask.dtype != bool:
        msg = f"active_mask must be boolean, got {active_mask.dtype!r}"
        raise InputError(msg)
    if not active_mask.index.equals(pd.Index(spread_pnls.columns)):
        msg = "active_mask index must match spread_pnls columns"
        raise InputError(msg)


def _zero_weights(columns: pd.Index) -> pd.Series:
    return pd.Series(0.0, index=columns, dtype=float)


@dataclass(frozen=True, slots=True)
class EqualDollarAllocator:
    """Assign ``1/N`` to every active pair.

    The simplest possible allocator. Useful both as a baseline and as a
    fallback when other allocators cannot run (insufficient history, etc.).
    """

    def weights(self, spread_pnls: pd.DataFrame, active_mask: pd.Series) -> pd.Series:
        _validate_inputs(spread_pnls, active_mask)
        out = _zero_weights(pd.Index(spread_pnls.columns))
        active = active_mask[active_mask]
        if active.empty:
            return out
        share = 1.0 / float(len(active))
        out.loc[active.index] = share
        return out


@dataclass(frozen=True, slots=True)
class InverseVolAllocator:
    """Weight inversely proportional to trailing standard deviation.

    Pairs whose rolling volatility cannot be estimated (insufficient observed
    history within the window) receive zero weight. The remaining weights are
    normalised to sum to one across the active set.

    Parameters
    ----------
    window : int, default ``60``
        Rolling window used to estimate volatility.
    min_periods : int, default ``20``
        Minimum non-null observations required inside the window.
    """

    window: int = 60
    min_periods: int = 20

    def weights(self, spread_pnls: pd.DataFrame, active_mask: pd.Series) -> pd.Series:
        _validate_inputs(spread_pnls, active_mask)
        if int(self.window) <= 1:
            msg = f"window must be > 1, got {self.window!r}"
            raise InputError(msg)
        if int(self.min_periods) <= 0:
            msg = f"min_periods must be positive, got {self.min_periods!r}"
            raise InputError(msg)
        out = _zero_weights(pd.Index(spread_pnls.columns))
        active = active_mask[active_mask].index
        if len(active) == 0:
            return out
        recent = spread_pnls.loc[:, active].tail(self.window)
        vol = recent.std(ddof=1, skipna=True)
        valid = recent.count() >= self.min_periods
        vol = vol.where(valid & vol.gt(0.0) & vol.notna())
        inv = 1.0 / vol
        inv = inv.replace([np.inf, -np.inf], np.nan).dropna()
        if inv.empty:
            return out
        weights = inv / inv.sum()
        out.loc[weights.index] = weights.to_numpy(dtype=float)
        return out


def _corr_to_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """Convert a correlation matrix to Lopez de Prado's distance metric."""
    clipped = corr.clip(-1.0, 1.0)
    dist = np.sqrt(0.5 * (1.0 - clipped))
    arr = dist.to_numpy(copy=True)
    np.fill_diagonal(arr, 0.0)
    return pd.DataFrame(arr, index=dist.index, columns=dist.columns)


def _quasi_diag(link: np.ndarray) -> list[int]:
    """Return the leaf ordering implied by a scipy linkage matrix."""
    n_links = link.shape[0]
    n_items = n_links + 1
    order: list[int] = [int(link[-1, 0]), int(link[-1, 1])]
    while max(order) >= n_items:
        new_order: list[int] = []
        for item in order:
            if item < n_items:
                new_order.append(item)
                continue
            row = link[item - n_items]
            new_order.append(int(row[0]))
            new_order.append(int(row[1]))
        order = new_order
    return order


def _cluster_variance(cov: pd.DataFrame, indices: list[int]) -> float:
    sub = cov.iloc[indices, indices]
    diag = np.diag(sub.values).astype(float)
    diag = np.where(diag > 0.0, diag, 1e-12)
    ivp = 1.0 / diag
    ivp = ivp / ivp.sum()
    return float(ivp @ sub.values @ ivp)


def _recursive_bisection(cov: pd.DataFrame, sort_ix: list[int]) -> pd.Series:
    labels = list(cov.index[sort_ix])
    weights = pd.Series(1.0, index=labels)
    clusters: list[list[int]] = [sort_ix]
    while clusters:
        new_clusters: list[list[int]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            left = cluster[:mid]
            right = cluster[mid:]
            var_left = _cluster_variance(cov, left)
            var_right = _cluster_variance(cov, right)
            total = var_left + var_right
            alpha = 1.0 - var_left / total if total > 0.0 else 0.5
            for idx in left:
                weights.loc[cov.index[idx]] *= alpha
            for idx in right:
                weights.loc[cov.index[idx]] *= 1.0 - alpha
            new_clusters.append(left)
            new_clusters.append(right)
        clusters = new_clusters
    return weights


@dataclass(frozen=True, slots=True)
class HRPAllocator:
    """Hierarchical Risk Parity allocator following Lopez de Prado (2016).

    The algorithm is:

    1. Compute the correlation matrix of active spread P&Ls.
    2. Convert correlations to distances ``sqrt(0.5 * (1 - corr))``.
    3. Build a hierarchical clustering via ``scipy.cluster.hierarchy.linkage``.
    4. Quasi-diagonalise the covariance matrix using the cluster ordering.
    5. Assign weights via recursive bisection with inverse-variance allocation.

    Parameters
    ----------
    linkage : {"single", "complete", "average", "ward"}, default ``"single"``
        Linkage method passed to ``scipy.cluster.hierarchy.linkage``.
    max_weight : float or None, default ``None``
        Optional per-pair cap applied post-bisection; the excess mass is
        redistributed proportionally across the remaining pairs.
    min_history : int, default ``60``
        Minimum observations required before the allocator will run; fewer
        observations falls back to equal-dollar weighting on the active set.
    """

    linkage: str = "single"
    max_weight: float | None = None
    min_history: int = 60

    def weights(self, spread_pnls: pd.DataFrame, active_mask: pd.Series) -> pd.Series:
        _validate_inputs(spread_pnls, active_mask)
        out = _zero_weights(pd.Index(spread_pnls.columns))
        active = active_mask[active_mask].index
        if len(active) == 0:
            return out
        sub = spread_pnls.loc[:, active].dropna(how="any")
        if len(sub) < int(self.min_history) or sub.shape[1] < 2:
            return EqualDollarAllocator().weights(spread_pnls, active_mask)
        cov = sub.cov()
        corr = sub.corr()
        if corr.isna().any().any():
            return EqualDollarAllocator().weights(spread_pnls, active_mask)
        dist = _corr_to_distance(corr)
        condensed = squareform(dist.values, checks=False)
        link = linkage(condensed, method=self.linkage)
        sort_ix = _quasi_diag(link)
        raw = _recursive_bisection(cov, sort_ix)
        raw = raw.reindex(active).fillna(0.0)
        total = float(raw.sum())
        if total <= 0.0:
            return EqualDollarAllocator().weights(spread_pnls, active_mask)
        weights = raw / total
        if self.max_weight is not None:
            cap = float(self.max_weight)
            for _ in range(10):
                over = weights > cap
                if not over.any():
                    break
                excess = float((weights[over] - cap).sum())
                weights.loc[over] = cap
                room = ~over & (weights > 0.0)
                if not room.any():
                    break
                weights.loc[room] += excess * (weights.loc[room] / float(weights.loc[room].sum()))
            weights = weights / float(weights.sum())
        out.loc[weights.index] = weights.to_numpy(dtype=float)
        return out
