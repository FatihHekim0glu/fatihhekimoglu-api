"""Probability of Backtest Overfitting via Combinatorially Symmetric CV.

Reference: Bailey, D. H., Borwein, J., Lopez de Prado, M. and Zhu, Q. J.
(2017), "The Probability of Backtest Overfitting", *Journal of
Computational Finance*, 20(4).

Given a ``(T, N)`` matrix of trial returns the procedure:

1. Partitions the ``T`` observations into ``S`` contiguous slabs.
2. Enumerates the ``C(S, S/2)`` partitions of slabs into IS and OOS halves.
3. For each split: selects the trial maximising the IS Sharpe ratio,
   records its OOS rank ``omega`` in ``(0, 1)`` and the logit
   ``lambda = log(omega / (1 - omega))``.
4. Reports ``PBO = mean(lambda <= 0)``.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pairs._exceptions import InputError

from .results import PBOResult

__all__ = ["pbo_cscv"]


def _sharpe(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per-column Sharpe ratios, returning ``-inf`` for degenerate columns."""
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=1)
    result = np.full_like(mean, -np.inf, dtype=float)
    safe = std > 0.0
    result[safe] = mean[safe] / std[safe]
    return result


def pbo_cscv(
    returns_matrix: pd.DataFrame,
    *,
    s: int = 16,
    rng: np.random.Generator | None = None,  # noqa: ARG001 - deterministic enumeration
) -> PBOResult:
    """Compute the Probability of Backtest Overfitting (CSCV).

    Parameters
    ----------
    returns_matrix : pandas.DataFrame
        ``(T, N)`` table where each column is a trial strategy's return
        series. ``T`` must be at least ``s``.
    s : int, default ``16``
        Number of contiguous slabs the sample is divided into. Must be a
        positive even integer.
    rng : numpy.random.Generator, optional
        Unused; the enumeration is exhaustive and deterministic.

    Returns
    -------
    PBOResult
        ``pbo`` plus the per-split logits and bookkeeping.
    """
    if not isinstance(returns_matrix, pd.DataFrame):
        raise InputError("returns_matrix must be a pandas DataFrame")
    if s <= 0 or s % 2 != 0:
        raise InputError(f"s must be a positive even integer; got {s}")
    arr = returns_matrix.to_numpy(dtype=float, copy=False)
    t, n = arr.shape
    if t < s:
        raise InputError(f"T={t} must be at least s={s}")
    if n < 2:
        raise InputError("returns_matrix must have at least two strategies")

    # Partition rows into S contiguous slabs of as-equal sizes as possible.
    bounds = np.linspace(0, t, s + 1, dtype=int)
    slabs: list[NDArray[np.float64]] = [arr[bounds[i] : bounds[i + 1]] for i in range(s)]

    half = s // 2
    logits: list[float] = []
    for is_combo in combinations(range(s), half):
        is_mask = set(is_combo)
        is_rows = np.concatenate([slabs[i] for i in range(s) if i in is_mask], axis=0)
        oos_rows = np.concatenate([slabs[i] for i in range(s) if i not in is_mask], axis=0)
        if is_rows.shape[0] < 2 or oos_rows.shape[0] < 2:
            continue
        is_sr = _sharpe(is_rows)
        oos_sr = _sharpe(oos_rows)
        finite_is = np.isfinite(is_sr)
        if not np.any(finite_is):
            continue
        best = int(np.argmax(np.where(finite_is, is_sr, -np.inf)))
        # Rank of best in OOS, normalised to (0, 1) exclusive.
        order = np.argsort(oos_sr, kind="mergesort")
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(1, n + 1, dtype=float)
        omega = ranks[best] / (n + 1.0)
        omega = float(np.clip(omega, 1e-12, 1.0 - 1e-12))
        logits.append(float(np.log(omega / (1.0 - omega))))
    if not logits:  # pragma: no cover - defensive
        raise InputError("CSCV produced no usable splits")
    pbo = float(np.mean(np.asarray(logits) <= 0.0))
    return PBOResult(
        pbo=pbo,
        logit_lambdas=tuple(logits),
        n_splits=len(logits),
        s_partitions=int(s),
    )
