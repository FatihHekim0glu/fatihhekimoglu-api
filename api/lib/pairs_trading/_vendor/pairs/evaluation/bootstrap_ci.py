"""Stationary-bootstrap confidence intervals for scalar statistics.

Implements the Politis-Romano (1994) stationary block bootstrap. Block
lengths are drawn from a geometric distribution with mean
``expected_block`` and starting indices are drawn uniformly with
wraparound so that the resampled series is stationary.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pairs._exceptions import InputError

from .results import BootstrapCI

__all__ = ["stationary_bootstrap_ci"]


def _default_block_length(n: int) -> int:
    return max(2, int(round(n ** (1.0 / 3.0))))


def _stationary_indices(
    n: int,
    expected_block: int,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    p = 1.0 / float(expected_block)
    indices = np.empty(n, dtype=np.int64)
    filled = 0
    while filled < n:
        start = int(rng.integers(0, n))
        length = int(rng.geometric(p))
        length = min(length, n - filled)
        for k in range(length):
            indices[filled + k] = (start + k) % n
        filled += length
    return indices


def stationary_bootstrap_ci(
    returns: pd.Series | NDArray[np.float64],
    statistic: Callable[[NDArray[np.float64]], float],
    *,
    alpha: float = 0.05,
    n_boot: int = 2000,
    expected_block: int | None = None,
    rng: np.random.Generator | None = None,
) -> BootstrapCI:
    """Compute a stationary-bootstrap confidence interval.

    Parameters
    ----------
    returns : pandas.Series or numpy.ndarray
        Observed returns or other one-dimensional series.
    statistic : callable
        Function mapping a 1-D ``ndarray`` to a scalar.
    alpha : float, default ``0.05``
        Two-sided significance level. The returned interval covers
        ``1 - alpha`` of the bootstrap distribution.
    n_boot : int, default ``2000``
        Number of bootstrap replicates.
    expected_block : int, optional
        Expected geometric block length. Defaults to
        ``max(2, round(n**(1/3)))``.
    rng : numpy.random.Generator, optional
        Source of randomness; defaults to :func:`numpy.random.default_rng`.

    Returns
    -------
    BootstrapCI
        Point estimate, lower / upper percentile bounds and bookkeeping.
    """
    if not (0.0 < alpha < 1.0):
        raise InputError(f"alpha must lie in (0, 1); got {alpha}")
    if n_boot <= 0:
        raise InputError(f"n_boot must be positive; got {n_boot}")
    arr = (
        returns.to_numpy(dtype=float, copy=False)
        if isinstance(returns, pd.Series)
        else np.asarray(returns, dtype=float)
    )
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 2:
        raise InputError("need at least two finite observations")
    block = expected_block if expected_block is not None else _default_block_length(n)
    if block <= 0:
        raise InputError(f"expected_block must be positive; got {block}")
    generator = rng if rng is not None else np.random.default_rng()
    point = float(statistic(arr))
    replicates = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = _stationary_indices(n, block, generator)
        try:
            replicates[i] = float(statistic(arr[idx]))
        except Exception:  # pragma: no cover - defensive
            replicates[i] = np.nan
    finite = replicates[np.isfinite(replicates)]
    if finite.size == 0:  # pragma: no cover - defensive
        raise InputError("bootstrap produced no finite replicates")
    low = float(np.quantile(finite, alpha / 2.0))
    high = float(np.quantile(finite, 1.0 - alpha / 2.0))
    return BootstrapCI(
        point_estimate=point,
        ci_low=low,
        ci_high=high,
        alpha=float(alpha),
        n_boot=int(n_boot),
        expected_block=int(block),
    )
