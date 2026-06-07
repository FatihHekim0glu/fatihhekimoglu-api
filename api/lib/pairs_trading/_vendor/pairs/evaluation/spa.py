"""Hansen (2005) Superior Predictive Ability test.

Tests the composite null that no model in the candidate set has a
positive expected loss differential against a benchmark. If the
optional ``arch`` package is installed its :class:`arch.bootstrap.SPA`
implementation is used; otherwise a self-contained stationary-bootstrap
implementation of the studentized maximum statistic is used.

Reference: Hansen, P. R. (2005), "A Test for Superior Predictive
Ability", *Journal of Business and Economic Statistics*, 23(4), 365-380.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pairs._exceptions import InputError

from .bootstrap_ci import _stationary_indices
from .results import SPAResult

try:  # pragma: no cover - optional dependency
    from arch.bootstrap import SPA as _ArchSPA

    _HAS_ARCH = True
except Exception:  # pragma: no cover - exercised when arch missing
    _ArchSPA = None  # type: ignore[assignment,misc]
    _HAS_ARCH = False

__all__ = ["hansen_spa"]


def _block_length(n: int) -> int:
    return max(2, int(round(n ** (1.0 / 3.0))))


def _fallback_spa(
    excess: NDArray[np.float64],
    columns: list[str],
    n_boot: int,
    rng: np.random.Generator,
) -> SPAResult:
    t, m = excess.shape
    mu = excess.mean(axis=0)
    sigma = excess.std(axis=0, ddof=1)
    sigma_safe = np.where(sigma > 0.0, sigma, 1.0)
    studentised = np.sqrt(t) * mu / sigma_safe
    studentised = np.where(sigma > 0.0, studentised, -np.inf)
    observed = float(np.max(np.concatenate([studentised, [0.0]])))

    block = _block_length(t)
    boot_max_cons = np.empty(n_boot, dtype=float)
    boot_max_lower = np.empty(n_boot, dtype=float)
    boot_max_upper = np.empty(n_boot, dtype=float)
    # Hansen's recentring thresholds for consistent / lower / upper p-values.
    threshold_consistent = -np.sqrt(2.0 * np.log(np.log(max(t, 3))) / t) * sigma_safe
    centring_lower = np.minimum(mu, 0.0)  # only re-centre non-positive means
    for i in range(n_boot):
        idx = _stationary_indices(t, block, rng)
        sample = excess[idx]
        sample_mean = sample.mean(axis=0)
        boot_stat_upper = np.sqrt(t) * (sample_mean - mu) / sigma_safe
        boot_stat_lower = np.sqrt(t) * (sample_mean - centring_lower) / sigma_safe
        keep_cons = mu >= threshold_consistent
        centring_consistent = np.where(keep_cons, mu, 0.0)
        boot_stat_cons = np.sqrt(t) * (sample_mean - centring_consistent) / sigma_safe
        boot_stat_upper = np.where(sigma > 0.0, boot_stat_upper, -np.inf)
        boot_stat_lower = np.where(sigma > 0.0, boot_stat_lower, -np.inf)
        boot_stat_cons = np.where(sigma > 0.0, boot_stat_cons, -np.inf)
        boot_max_upper[i] = max(float(np.max(boot_stat_upper)), 0.0)
        boot_max_lower[i] = max(float(np.max(boot_stat_lower)), 0.0)
        boot_max_cons[i] = max(float(np.max(boot_stat_cons)), 0.0)

    p_upper = float(np.mean(boot_max_upper >= observed))
    p_lower = float(np.mean(boot_max_lower >= observed))
    p_cons = float(np.mean(boot_max_cons >= observed))
    best_idx = int(np.argmax(studentised))
    return SPAResult(
        p_value_consistent=float(np.clip(p_cons, 0.0, 1.0)),
        p_value_lower=float(np.clip(p_lower, 0.0, 1.0)),
        p_value_upper=float(np.clip(p_upper, 0.0, 1.0)),
        best_model=columns[best_idx],
        n_models=int(m),
        n_boot=int(n_boot),
    )


def hansen_spa(
    strategy_returns: pd.DataFrame,
    benchmark: pd.Series,
    *,
    n_boot: int = 999,
    rng: np.random.Generator | None = None,
) -> SPAResult:
    """Run Hansen's SPA test on a panel of candidate strategies.

    Parameters
    ----------
    strategy_returns : pandas.DataFrame
        ``(T, M)`` matrix of candidate-strategy returns.
    benchmark : pandas.Series
        Length-``T`` benchmark return series, aligned to
        ``strategy_returns``.
    n_boot : int, default ``999``
        Number of bootstrap replicates.
    rng : numpy.random.Generator, optional
        Source of randomness. Required only by the in-house fallback.

    Returns
    -------
    SPAResult
        Consistent, lower and upper p-values plus the best-performing
        column label.
    """
    if not isinstance(strategy_returns, pd.DataFrame):
        raise InputError("strategy_returns must be a pandas DataFrame")
    if not isinstance(benchmark, pd.Series):
        raise InputError("benchmark must be a pandas Series")
    if strategy_returns.shape[1] == 0:
        raise InputError("strategy_returns must have at least one column")
    if n_boot <= 0:
        raise InputError(f"n_boot must be positive; got {n_boot}")
    aligned = strategy_returns.align(benchmark, axis=0, join="inner")
    strat_df, bench_s = aligned[0], aligned[1]
    strat_df = strat_df.dropna(how="any")
    common_idx = strat_df.index.intersection(bench_s.dropna().index)
    strat_df = strat_df.loc[common_idx]
    bench_s = bench_s.loc[common_idx]
    if strat_df.shape[0] < 8:
        raise InputError("need at least eight aligned observations")
    excess = strat_df.to_numpy(dtype=float) - bench_s.to_numpy(dtype=float)[:, None]
    columns = [str(c) for c in strat_df.columns]
    generator = rng if rng is not None else np.random.default_rng()
    if _HAS_ARCH:  # pragma: no cover - depends on optional package
        try:
            loss = -excess  # SPA in `arch` works in loss space
            spa = _ArchSPA(
                loss[:, 0],
                loss[:, 1:],
                reps=int(n_boot),
                seed=int(generator.integers(0, 2**31 - 1)),
            )
            spa.compute()
            pvals = spa.pvalues
            best_idx = int(np.argmax(excess.mean(axis=0)))
            return SPAResult(
                p_value_consistent=float(pvals["consistent"]),
                p_value_lower=float(pvals["lower"]),
                p_value_upper=float(pvals["upper"]),
                best_model=columns[best_idx],
                n_models=excess.shape[1],
                n_boot=int(n_boot),
            )
        except Exception:
            # Fall back to the hand-rolled implementation on any failure.
            pass
    return _fallback_spa(excess, columns, int(n_boot), generator)
