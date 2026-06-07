"""Romano-Wolf step-down multiple-testing procedure for OOS Sharpe ratios.

This implementation evaluates one-sided tests of the form

    H0_k:  E[r_k - r_bench] <= 0
    H1_k:  E[r_k - r_bench]  > 0

where ``r_k`` is the OOS return series of strategy ``k`` and ``r_bench`` is
a common benchmark return series (set to zero when no benchmark is
supplied). For each strategy we compute a ``sqrt(T)`` t-statistic on the
differential mean using a Newey-West HAC standard error with the
Andrews (1991) plug-in bandwidth.

The null distribution is built via a stationary bootstrap on the *centred*
differential returns. ``arch.bootstrap.StationaryBootstrap`` is used when
available; otherwise we fall back to a hand-rolled stationary bootstrap
with geometric block lengths -- adequate for unit testing but slower.

The step-down loop follows Romano & Wolf (2005): take the largest observed
t-stat, compare against the bootstrap distribution of the maximum, reject
if it exceeds the ``1 - alpha`` quantile, then repeat on the remaining
hypotheses. ``adjusted_pvalues`` are constructed monotonically so they can
be inspected without re-running the bootstrap.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pairs._exceptions import InputError
from pairs._rng import default_rng, derive_rng
from pairs.selection.results import RWResult

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

__all__ = ["romano_wolf_stepdown"]


def _try_import_arch() -> object | None:
    """Return the ``arch.bootstrap`` module if installed, else ``None``."""
    try:
        from arch import bootstrap as arch_bootstrap

        return arch_bootstrap
    except ImportError:
        return None


def _newey_west_var(x: NDArray[np.float64], lag: int) -> float:
    """Compute the Newey-West HAC variance estimator with Bartlett kernel.

    Parameters
    ----------
    x
        One-dimensional array of (already de-meaned or raw) observations.
    lag
        Number of lags. ``0`` yields the plain sample variance.
    """
    n = x.size
    if n == 0:
        return 0.0
    centred = x - x.mean()
    gamma0 = float(np.dot(centred, centred) / n)
    out = gamma0
    for l_ in range(1, lag + 1):
        if l_ >= n:
            break
        weight = 1.0 - l_ / (lag + 1)
        cov = float(np.dot(centred[l_:], centred[:-l_]) / n)
        out += 2.0 * weight * cov
    return max(out, 0.0)


def _auto_lag(n: int) -> int:
    """Andrews-style automatic lag selection: ``floor(4 * (n/100)^(2/9))``."""
    if n <= 1:
        return 0
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def _tstat(diff: NDArray[np.float64]) -> float:
    """Return ``sqrt(T) * mean(diff) / NW_se(diff)`` or 0 when degenerate."""
    n = diff.size
    if n < 2:
        return 0.0
    lag = _auto_lag(n)
    var = _newey_west_var(diff, lag)
    if var <= 0.0:
        return 0.0
    se = np.sqrt(var)
    return float(np.sqrt(n) * diff.mean() / se)


def _stationary_block_length(n: int) -> int:
    """Default block length ``floor(n^(1/3))`` clipped to ``[1, n]``."""
    if n <= 1:
        return 1
    return max(1, min(n, int(np.floor(n ** (1.0 / 3.0)))))


def _stationary_bootstrap_indices(
    n: int,
    block_length: int,
    rng: np.random.Generator,
) -> NDArray[np.intp]:
    """Hand-rolled stationary bootstrap: sample indices with geometric blocks.

    Returns a length-``n`` array of indices into the original sample.
    Vectorised: pre-draws all start-block flips and uses ``np.where`` to
    select either the previous index (incremented) or a fresh random
    starting index.
    """
    if block_length <= 0:
        msg = "block_length must be positive"
        raise InputError(msg)
    p = 1.0 / block_length
    starts = rng.integers(0, n, size=n)
    flips = rng.random(size=n) < p
    flips[0] = True  # first index is always a fresh draw
    idx = np.empty(n, dtype=np.intp)
    current = int(starts[0])
    for t in range(n):
        if flips[t]:
            current = int(starts[t])
        idx[t] = current
        current = (current + 1) % n
    return idx


def _bootstrap_with_arch(
    centred: NDArray[np.float64],
    block_length: int,
    n_boot: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Use ``arch.bootstrap.StationaryBootstrap`` to draw resampled t-stats."""
    arch_bootstrap = _try_import_arch()
    if arch_bootstrap is None:
        msg = "arch is not installed"
        raise RuntimeError(msg)
    bs = arch_bootstrap.StationaryBootstrap(  # type: ignore[attr-defined]
        block_length,
        centred,
        seed=int(rng.integers(0, 2**31 - 1)),
    )
    out = np.empty((n_boot, centred.shape[1]), dtype=np.float64)
    for b, data in enumerate(bs.bootstrap(n_boot)):
        sample = data[0][0]
        for k in range(sample.shape[1]):
            out[b, k] = _tstat(sample[:, k])
    return out


def _bootstrap_handrolled(
    centred: NDArray[np.float64],
    block_length: int,
    n_boot: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Pure-numpy stationary bootstrap used when ``arch`` is unavailable."""
    n, k = centred.shape
    out = np.empty((n_boot, k), dtype=np.float64)
    for b in range(n_boot):
        idx = _stationary_bootstrap_indices(n, block_length, rng)
        sample = centred[idx]
        for j in range(k):
            out[b, j] = _tstat(sample[:, j])
    return out


def _align_inputs(
    oos_sharpes: dict[str, pd.Series],
    benchmark: pd.Series | None,
) -> tuple[list[str], NDArray[np.float64], pd.DatetimeIndex]:
    """Inner-join the strategy returns and benchmark onto a common index.

    Returns
    -------
    keys
        Ordered list of strategy identifiers retained after alignment.
    diff
        ``(T, K)`` array of differential returns ``r_k - r_bench``.
    index
        Common time index after the inner join.
    """
    if not oos_sharpes:
        msg = "oos_sharpes must contain at least one strategy"
        raise InputError(msg)
    frame = pd.DataFrame(dict(oos_sharpes)).dropna(how="any")
    if benchmark is not None:
        bench = benchmark.reindex(frame.index)
        # Use only the intersection so we don't drop the whole frame.
        common = frame.index.intersection(bench.dropna().index)
        frame = frame.loc[common]
        bench = bench.loc[common]
    else:
        bench = pd.Series(0.0, index=frame.index)
    if frame.empty:
        msg = "no overlapping observations between strategies and benchmark"
        raise InputError(msg)
    diff = frame.to_numpy(dtype=float) - bench.to_numpy(dtype=float).reshape(-1, 1)
    return list(frame.columns.astype(str)), diff, pd.DatetimeIndex(frame.index)


def romano_wolf_stepdown(
    oos_sharpes: dict[str, pd.Series],
    benchmark: pd.Series | None,
    *,
    n_boot: int = 999,
    block_length: int | None = None,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> RWResult:
    """Romano-Wolf step-down procedure controlling FWER.

    Parameters
    ----------
    oos_sharpes
        Mapping from pair identifier to OOS return series. The series are
        inner-joined onto a common date index before testing.
    benchmark
        Optional benchmark return series. ``None`` is treated as the zero
        benchmark (i.e. testing whether each strategy has a positive mean
        return).
    n_boot
        Number of bootstrap replications. Must be ``>= 1``.
    block_length
        Stationary-bootstrap mean block length. When ``None`` the function
        auto-selects ``floor(T^(1/3))``.
    alpha
        Family-wise error rate target.
    rng
        Optional NumPy ``Generator``. Defaults to the library-wide
        deterministic generator so calls are reproducible.

    Returns
    -------
    RWResult
        Aggregated step-down output. ``surviving_pair_ids`` lists the
        rejected hypotheses in the order they were rejected.

    Raises
    ------
    InputError
        If ``oos_sharpes`` is empty, ``alpha`` is out of range, ``n_boot``
        is non-positive, or the alignment leaves no observations.
    """
    if n_boot <= 0:
        msg = f"n_boot must be positive; got {n_boot}"
        raise InputError(msg)
    if not (0.0 < alpha < 1.0):
        msg = f"alpha must lie in (0, 1); got {alpha}"
        raise InputError(msg)

    keys, diff, _index = _align_inputs(oos_sharpes, benchmark)
    n_obs, k_hyp = diff.shape

    if block_length is None:
        block_length = _stationary_block_length(n_obs)
    if block_length <= 0:
        msg = f"block_length must be positive; got {block_length}"
        raise InputError(msg)

    # Observed t-statistics, one per hypothesis.
    observed = np.array([_tstat(diff[:, k]) for k in range(k_hyp)], dtype=float)

    # Centre each column to enforce the null when bootstrapping.
    centred = diff - diff.mean(axis=0, keepdims=True)
    parent = rng if rng is not None else default_rng()
    bs_rng = derive_rng(parent, "romano_wolf.bootstrap")

    if _try_import_arch() is not None:
        try:
            boot_tstats = _bootstrap_with_arch(centred, block_length, n_boot, bs_rng)
        except RuntimeError:
            boot_tstats = _bootstrap_handrolled(centred, block_length, n_boot, bs_rng)
    else:
        boot_tstats = _bootstrap_handrolled(centred, block_length, n_boot, bs_rng)

    # Step-down loop.
    remaining = list(range(k_hyp))
    rejected_in_order: list[int] = []
    adjusted = np.ones(k_hyp, dtype=float)
    null_max_full = boot_tstats.max(axis=1)

    while remaining:
        sub_t = observed[remaining]
        top_local = int(np.argmax(sub_t))
        top_global = remaining[top_local]
        top_value = sub_t[top_local]
        null_max = boot_tstats[:, remaining].max(axis=1)
        # Right-tail empirical p-value of the maximum.
        p_max = float((1.0 + np.sum(null_max >= top_value)) / (1.0 + n_boot))
        # Monotone adjustment: never smaller than the previous adjusted p.
        prev = adjusted[rejected_in_order[-1]] if rejected_in_order else 0.0
        p_adj = max(p_max, prev)
        adjusted[top_global] = p_adj
        if p_adj <= alpha:
            rejected_in_order.append(top_global)
            remaining.remove(top_global)
        else:
            # All remaining hypotheses inherit the current adjusted p as a
            # lower bound; we still set them and break out.
            for idx in remaining:
                adjusted[idx] = max(adjusted[idx], p_adj)
            break

    surviving_pair_ids = [keys[i] for i in rejected_in_order]
    adjusted_pvalues = pd.Series(adjusted, index=keys, name="adjusted_pvalue")
    return RWResult(
        surviving_pair_ids=surviving_pair_ids,
        adjusted_pvalues=adjusted_pvalues,
        null_distribution=np.asarray(null_max_full, dtype=np.float64),
        block_length=int(block_length),
        n_boot=int(n_boot),
    )
