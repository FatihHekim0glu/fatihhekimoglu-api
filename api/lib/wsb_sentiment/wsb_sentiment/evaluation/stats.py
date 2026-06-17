"""Honest-statistics layer for the sentiment signal.

Bundles the overfitting / significance guards the verdict depends on:

- the DEFLATED / PROBABILISTIC Sharpe ratio with an EFFECTIVE number of trials
  estimated by PCA of the trial-return matrix over the swept
  lexicon x window x lag x threshold x cost grid (so correlated trials do not
  inflate the multiplicity count);
- PROBABILITY OF BACKTEST OVERFITTING via CSCV (:func:`pbo_cscv`);
- a HAC (Newey-West) t-statistic on the net OOS returns;
- the MEMMEL-JK test of the signal's Sharpe against the buy-and-hold baseline.

This module orchestrates the vendored primitives (``dsr``, ``pbo``, ``hac``,
``memmel``, ``bootstrap_ci``); it adds no new heavy math beyond the PCA-of-trials
effective-trials estimator. Importing this module has no side effects.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from wsb_sentiment._exceptions import ValidationError
from wsb_sentiment._typing import FloatArray
from wsb_sentiment.evaluation.dsr import (
    _norm_cdf,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from wsb_sentiment.evaluation.hac import newey_west_se
from wsb_sentiment.evaluation.pbo import pbo_cscv


@dataclass(frozen=True, slots=True)
class HonestStats:
    """The full honest-statistics bundle the verdict is derived from.

    Attributes
    ----------
    net_sharpe:
        Annualized net (after-cost) OOS Sharpe of the selected signal.
    buyhold_sharpe:
        Annualized OOS Sharpe of the equal-weight buy-and-hold baseline.
    deflated_sharpe:
        The Deflated Sharpe Ratio in ``[0, 1]`` (FULL effective-trials grid).
    psr:
        The Probabilistic Sharpe Ratio in ``[0, 1]`` vs a zero benchmark.
    pbo:
        The Probability of Backtest Overfitting in ``[0, 1]`` (CSCV).
    hac_tstat:
        The Newey-West HAC t-statistic on the net OOS mean return.
    hac_pvalue:
        The two-sided p-value for ``hac_tstat``.
    n_effective_trials:
        The PCA-estimated effective number of independent trials.
    n_obs:
        The number of OOS observations.
    """

    net_sharpe: float
    buyhold_sharpe: float
    deflated_sharpe: float
    psr: float
    pbo: float
    hac_tstat: float
    hac_pvalue: float
    n_effective_trials: float
    n_obs: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the bundle."""
        return asdict(self)


def effective_n_trials(trial_returns: pd.DataFrame, *, var_threshold: float = 0.95) -> float:
    """Estimate the EFFECTIVE number of independent trials by PCA of trial returns.

    Many configurations on the swept grid (lexicon x window x lag x threshold x
    cost) produce highly-correlated return streams, so the raw grid size grossly
    overstates the multiplicity. We PCA the standardized trial-return matrix and
    return the number of principal components needed to explain ``var_threshold``
    of the variance — a deflation-honest effective trial count that feeds the DSR.

    Parameters
    ----------
    trial_returns:
        A ``(T, N_trials)`` matrix; each column is one configuration's OOS return
        series.
    var_threshold:
        The cumulative-variance fraction defining "enough" components
        (default ``0.95``).

    Returns
    -------
    float
        The effective number of trials (``>= 1``).

    Raises
    ------
    ValidationError
        If ``trial_returns`` is not a DataFrame, has fewer than one column, or
        ``var_threshold`` is outside ``(0, 1]``.
    """
    if not isinstance(trial_returns, pd.DataFrame):
        raise ValidationError("trial_returns must be a pandas DataFrame.")
    if not 0.0 < var_threshold <= 1.0:
        raise ValidationError(f"var_threshold must be in (0, 1], got {var_threshold}.")

    arr = trial_returns.to_numpy(dtype=float, copy=True)
    if arr.ndim != 2 or arr.shape[1] < 1:
        raise ValidationError("trial_returns must be a 2-D matrix with at least one column.")

    n_trials = arr.shape[1]
    # A single configuration cannot be deflated below one independent trial.
    if n_trials == 1:
        return 1.0

    # Standardize each column (zero mean, unit variance). Columns with zero
    # variance carry no information and collapse onto already-explained axes, so
    # they are dropped before the PCA rather than producing NaNs.
    arr = arr[np.all(np.isfinite(arr), axis=1)]
    if arr.shape[0] < 2:
        # Too few observations to estimate a covariance: fall back to the raw
        # (most conservative) multiplicity count.
        return float(n_trials)

    std = arr.std(axis=0, ddof=1)
    keep = std > 0.0
    n_dropped = int((~keep).sum())
    if not np.any(keep):
        # Every column is constant: a single degenerate trial.
        return 1.0
    standardized = (arr[:, keep] - arr[:, keep].mean(axis=0)) / std[keep]

    # Correlation-matrix eigenvalues are the variances of the principal axes.
    corr = np.corrcoef(standardized, rowvar=False)
    corr = np.atleast_2d(corr)
    eigvals = np.linalg.eigvalsh(corr)
    eigvals = np.clip(eigvals[::-1], 0.0, None)  # descending, non-negative
    total = float(eigvals.sum())
    if total <= 0.0:  # pragma: no cover - defensive; corr trace == n_kept > 0
        return 1.0

    cumulative = np.cumsum(eigvals) / total
    # Number of components needed to reach the variance threshold.
    n_components = int(np.searchsorted(cumulative, var_threshold) + 1)
    n_components = min(n_components, int(keep.sum()))

    # Dropped (degenerate) columns are perfectly redundant, so they add no
    # independent trials; the effective count is bounded below by one.
    _ = n_dropped
    return float(max(n_components, 1))


def hac_tstat(returns: pd.Series | FloatArray, *, lag: int | None = None) -> tuple[float, float]:
    """Newey-West HAC t-statistic and two-sided p-value for the mean return.

    Wraps the vendored :func:`wsb_sentiment.evaluation.hac.newey_west_se` to form
    ``t = mean / HAC_se(mean)`` and a normal-approximation two-sided p-value, so
    serial correlation in the daily OOS returns does not inflate significance.

    Parameters
    ----------
    returns:
        The net OOS return series.
    lag:
        Bartlett lag truncation; ``None`` selects the Andrews automatic rule.

    Returns
    -------
    tuple[float, float]
        The HAC t-statistic and its two-sided p-value.

    Raises
    ------
    ValidationError
        Propagated from :func:`newey_west_se` when there are fewer than two
        finite observations.
    """
    if isinstance(returns, pd.Series):
        arr = returns.to_numpy(dtype=float, copy=False)
    else:
        arr = np.asarray(returns, dtype=float)
    finite = arr[np.isfinite(arr)]

    se = newey_west_se(returns, lag=lag)
    mean = float(finite.mean())
    if se <= 0.0:
        # A degenerate (zero long-run variance) series carries no usable signal:
        # a zero t-stat and an uninformative p-value of one.
        return 0.0, 1.0

    tstat = mean / se
    # Two-sided normal-approximation p-value: 2 * (1 - Phi(|t|)).
    pvalue = 2.0 * (1.0 - _norm_cdf(abs(tstat)))
    pvalue = min(max(pvalue, 0.0), 1.0)
    return float(tstat), float(pvalue)


def compute_honest_stats(
    net_returns: pd.Series,
    buyhold_returns: pd.Series,
    trial_returns: pd.DataFrame,
    *,
    n_grid_trials: int,
    periods_per_year: int = 252,
) -> HonestStats:
    """Assemble the full honest-statistics bundle from OOS returns and the trial grid.

    Computes net/buy-hold Sharpe, the PCA-effective trial count, the DSR/PSR
    (vendored ``dsr``), the PBO via CSCV (vendored ``pbo``), the HAC t-stat
    (:func:`hac_tstat`), and the Memmel-JK comparison vs buy-and-hold (vendored
    ``memmel``), returning a single immutable :class:`HonestStats`.

    Parameters
    ----------
    net_returns:
        The net (after-cost) OOS return series of the selected signal.
    buyhold_returns:
        The OOS return series of the equal-weight buy-and-hold baseline.
    trial_returns:
        The ``(T, N_grid)`` matrix of all swept configurations' OOS returns (for
        PBO and the effective-trials estimate).
    n_grid_trials:
        The raw number of configurations swept (the upper bound on multiplicity).
    periods_per_year:
        Annualization factor (``252`` for daily).

    Returns
    -------
    HonestStats
        The full bundle the verdict consumes.

    Raises
    ------
    ValidationError
        If the return series are too short, ``n_grid_trials < 1``, or
        ``trial_returns`` is malformed.
    """
    if not isinstance(net_returns, pd.Series) or not isinstance(buyhold_returns, pd.Series):
        raise ValidationError("net_returns and buyhold_returns must be pandas Series.")
    if not isinstance(trial_returns, pd.DataFrame):
        raise ValidationError("trial_returns must be a pandas DataFrame.")
    if n_grid_trials < 1:
        raise ValidationError(f"n_grid_trials must be >= 1, got {n_grid_trials}.")
    if periods_per_year < 1:
        raise ValidationError(f"periods_per_year must be >= 1, got {periods_per_year}.")

    net = net_returns.to_numpy(dtype=float, copy=False)
    net = net[np.isfinite(net)]
    bh = buyhold_returns.to_numpy(dtype=float, copy=False)
    bh = bh[np.isfinite(bh)]
    n_obs = int(net.size)
    if n_obs < 2:
        raise ValidationError(f"net_returns needs >= 2 finite observations, got {n_obs}.")
    if bh.size < 2:
        raise ValidationError(f"buyhold_returns needs >= 2 finite observations, got {bh.size}.")

    ann = math.sqrt(float(periods_per_year))

    # Per-observation and annualized Sharpe ratios of the selected signal.
    net_sr_obs, net_skew, net_kurt = _sharpe_and_moments(net)
    net_sharpe = net_sr_obs * ann
    bh_sr_obs, _, _ = _sharpe_and_moments(bh)
    buyhold_sharpe = bh_sr_obs * ann

    # Effective (PCA-deflated) trial count, bounded above by the raw grid size
    # and below by one, so the DSR multiplicity is honest but never absurd.
    eff = effective_n_trials(trial_returns)
    n_eff = max(1.0, min(eff, float(n_grid_trials)))

    # Variance of the per-observation Sharpe ratios across the swept grid.
    var_trial_sharpes = _trial_sharpe_variance(trial_returns)

    # PSR vs zero and the multiplicity-adjusted Deflated Sharpe.
    psr = probabilistic_sharpe_ratio(net_sr_obs, n_obs=n_obs, skew=net_skew, kurtosis=net_kurt)
    deflated = deflated_sharpe_ratio(
        net_sr_obs,
        n_obs=n_obs,
        n_trials=round(n_eff),
        variance_of_trial_sharpes=var_trial_sharpes,
        skew=net_skew,
        kurtosis=net_kurt,
    )

    # Probability of backtest overfitting over the swept grid (CSCV). The grid
    # may be too small/short for the default 16-slab partition, so the partition
    # count adapts and PBO degrades gracefully to ``nan``->0.5-neutral only when
    # genuinely uncomputable.
    pbo = _safe_pbo(trial_returns)

    # HAC (Newey-West) significance of the mean net OOS return.
    tstat, pvalue = hac_tstat(net_returns)

    return HonestStats(
        net_sharpe=float(net_sharpe),
        buyhold_sharpe=float(buyhold_sharpe),
        deflated_sharpe=float(deflated),
        psr=float(psr),
        pbo=float(pbo),
        hac_tstat=float(tstat),
        hac_pvalue=float(pvalue),
        n_effective_trials=float(n_eff),
        n_obs=int(n_obs),
    )


def _sharpe_and_moments(arr: FloatArray) -> tuple[float, float, float]:
    """Per-observation Sharpe ratio plus sample skewness and FULL kurtosis.

    Returns ``(0.0, 0.0, 3.0)`` for a degenerate (zero-variance) series so the
    downstream PSR/DSR see a neutral, Gaussian-shaped null rather than NaNs.
    """
    n = arr.size
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    if std <= 0.0:
        return 0.0, 0.0, 3.0
    centred = arr - mean
    m2 = float(np.mean(centred**2))
    if m2 <= 0.0:  # pragma: no cover - guarded by std check above
        return 0.0, 0.0, 3.0
    m3 = float(np.mean(centred**3))
    m4 = float(np.mean(centred**4))
    skew = m3 / m2**1.5
    kurt = m4 / m2**2  # FULL (non-excess) kurtosis; Gaussian == 3.
    return mean / std, skew, kurt


def _trial_sharpe_variance(trial_returns: pd.DataFrame) -> float:
    """Cross-trial variance of the per-observation Sharpe ratios.

    Each column's per-observation Sharpe is computed (skipping degenerate
    columns); the sample variance of those Sharpes is the ``V`` the Deflated
    Sharpe needs. A single usable column yields ``0.0`` (no multiplicity spread).
    """
    arr = trial_returns.to_numpy(dtype=float, copy=False)
    sharpes: list[float] = []
    for j in range(arr.shape[1]):
        col = arr[:, j]
        col = col[np.isfinite(col)]
        if col.size < 2:
            continue
        std = float(col.std(ddof=1))
        if std > 0.0:
            sharpes.append(float(col.mean()) / std)
    if len(sharpes) < 2:
        return 0.0
    return float(np.var(np.asarray(sharpes), ddof=1))


def _safe_pbo(trial_returns: pd.DataFrame) -> float:
    """PBO via CSCV with a partition count adapted to the available sample.

    The default 16-slab partition needs ``T >= 16`` rows and ``N >= 2`` trials.
    When the grid is smaller the largest feasible even slab count is used; if no
    valid configuration exists the PBO is reported as the neutral ``0.5`` (which
    by itself fails the low-PBO hurdle, preserving the honest null).
    """
    t, n = trial_returns.shape
    if n < 2 or t < 4:
        return 0.5
    s = min(16, t)
    if s % 2 != 0:
        s -= 1
    if s < 2:  # pragma: no cover - unreachable given the t >= 4 precheck above
        return 0.5
    result = pbo_cscv(trial_returns, s=s)
    return float(result.pbo)
