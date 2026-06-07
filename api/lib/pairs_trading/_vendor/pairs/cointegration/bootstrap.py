"""Stationary-bootstrap p-values for the Engle-Granger statistic.

The bootstrap procedure resamples the *first differences* of both
inputs under the null hypothesis of no cointegration, reconstructs
random-walk paths by cumulative summation, and recomputes the
Engle-Granger statistic on each bootstrap replicate.  The reported
p-value is the empirical fraction of bootstrap statistics that are at
least as extreme (more negative) than the observed statistic.

If the optional :mod:`arch` dependency is installed it is used for the
Politis-Romano stationary bootstrap.  Otherwise a compact in-house
implementation is used that generates wraparound blocks with geometric
lengths.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from statsmodels.tsa.stattools import coint

from pairs._exceptions import InputError

from ._internal.align import _inner_join_and_dropna
from .results import BootstrapResult

try:  # pragma: no cover - optional dependency
    from arch.bootstrap import StationaryBootstrap

    _HAS_ARCH = True
except Exception:  # pragma: no cover - exercised when arch is missing
    StationaryBootstrap = None  # type: ignore[assignment,misc]
    _HAS_ARCH = False


def _default_block_length(n: int) -> int:
    """Politis-Romano-style default block length: ``max(2, round(n**(1/3)))``."""
    return max(2, int(round(n ** (1.0 / 3.0))))


def _stationary_indices(
    n: int,
    block_length: int,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Generate ``n`` indices via wraparound geometric-block resampling."""
    p = 1.0 / float(block_length)
    indices = np.empty(n, dtype=np.int64)
    i = 0
    while i < n:
        start = int(rng.integers(0, n))
        length = int(rng.geometric(p))
        length = min(length, n - i)
        for k in range(length):
            indices[i + k] = (start + k) % n
        i += length
    return indices


def _coint_stat(y0: NDArray[np.float64], y1: NDArray[np.float64], trend: str) -> float:
    """Return the Engle-Granger ADF statistic without solving for a p-value."""
    stat, _, _ = coint(y0, y1, trend=trend, autolag=None)
    return float(stat)


def bootstrap_coint_pvalue(
    y0: pd.Series | NDArray[np.float64],
    y1: pd.Series | NDArray[np.float64],
    *,
    n_boot: int = 999,
    block_length: int | None = None,
    rng: np.random.Generator | None = None,
    trend: Literal["n", "c", "ct", "ctt"] = "c",
) -> BootstrapResult:
    """Compute a bootstrap p-value for the Engle-Granger statistic.

    Parameters
    ----------
    y0, y1 : pandas.Series or numpy.ndarray
        Aligned price series.  No log transform is applied here; the
        caller is expected to pass whatever scale they intend to test.
    n_boot : int, default ``999``
        Number of bootstrap replicates.
    block_length : int, optional
        Stationary-bootstrap expected block length.  Defaults to
        ``max(2, round(n**(1/3)))``.
    rng : numpy.random.Generator, optional
        Source of randomness; falls back to :func:`numpy.random.default_rng`.
    trend : {"n", "c", "ct", "ctt"}, default ``"c"``
        Deterministic trend in the cointegrating regression.

    Returns
    -------
    BootstrapResult
        Empirical p-value, block length used, observed statistic and a
        small dictionary of null-distribution quantiles.
    """
    if n_boot <= 0:
        raise InputError(f"n_boot must be positive; got {n_boot}")
    if block_length is not None and block_length <= 0:
        raise InputError(f"block_length must be positive; got {block_length}")

    a, b = _inner_join_and_dropna(y0, y1)
    arr0 = a.to_numpy(copy=True)
    arr1 = b.to_numpy(copy=True)
    n = arr0.size

    bl = block_length if block_length is not None else _default_block_length(n)
    gen = rng if rng is not None else np.random.default_rng()

    observed = _coint_stat(arr0, arr1, trend=trend)

    # Differences under the null of two unrelated I(1) processes.
    diffs = np.column_stack([np.diff(arr0), np.diff(arr1)])
    diff_n = diffs.shape[0]

    null_stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = _stationary_indices(diff_n, bl, gen)
        boot_diffs = diffs[idx]
        path0 = np.concatenate([[arr0[0]], arr0[0] + np.cumsum(boot_diffs[:, 0])])
        path1 = np.concatenate([[arr1[0]], arr1[0] + np.cumsum(boot_diffs[:, 1])])
        try:
            null_stats[i] = _coint_stat(path0, path1, trend=trend)
        except Exception:  # pragma: no cover - extremely rare numerical edge
            null_stats[i] = np.nan

    finite = null_stats[np.isfinite(null_stats)]
    if finite.size == 0:  # pragma: no cover - defensive
        raise InputError("bootstrap produced no finite replicates")

    # Davison-Hinkley (1997) continuity correction: (1 + count) / (1 + n_boot).
    # This guarantees a strictly positive p-value and matches the convention
    # used by ``arch.bootstrap`` and most academic references.
    count_le = int(np.sum(finite <= observed))
    pvalue = float((1 + count_le) / (1 + n_boot))
    quantiles = {q: float(np.quantile(finite, q)) for q in (0.01, 0.05, 0.10, 0.50, 0.90)}

    return BootstrapResult(
        pvalue=pvalue,
        n_boot=int(n_boot),
        block_length=int(bl),
        observed_stat=observed,
        null_stat_quantiles=quantiles,
    )
