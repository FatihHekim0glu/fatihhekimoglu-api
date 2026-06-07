"""Probabilistic and deflated Sharpe ratios (Bailey-Lopez de Prado).

Reference: Bailey, D. H. and Lopez de Prado, M. (2014),
"The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
Overfitting and Non-Normality", *Journal of Portfolio Management*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from pairs._exceptions import InputError

from .results import DSRResult

__all__ = [
    "deflated_sharpe_ratio",
    "effective_n_trials",
    "probabilistic_sharpe_ratio",
]


_EULER_MASCHERONI: float = 0.5772156649015329


def probabilistic_sharpe_ratio(
    sr_hat: float,
    sr_benchmark: float,
    n: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probability that the true SR exceeds ``sr_benchmark``.

    Implements equation (7) of Bailey-Lopez de Prado (2014). The
    distributional correction uses the sample skew and (raw, not excess)
    kurtosis of the underlying return distribution.

    Parameters
    ----------
    sr_hat : float
        Realised Sharpe ratio (same frequency as ``n``).
    sr_benchmark : float
        Reference Sharpe ratio to beat. Use ``0.0`` for the standard
        null of "no skill".
    n : int
        Number of return observations used to estimate ``sr_hat``.
    skew : float, default ``0.0``
        Sample skewness of returns.
    kurtosis : float, default ``3.0``
        *Raw* sample kurtosis of returns (i.e., 3 under Gaussianity).

    Returns
    -------
    float
        ``P(SR_true > sr_benchmark) in [0, 1]``.
    """
    if n <= 1:
        raise InputError(f"n must exceed 1; got {n}")
    if kurtosis <= 1.0:
        raise InputError(f"kurtosis must exceed 1; got {kurtosis}")
    denom_sq = 1.0 - skew * sr_hat + ((kurtosis - 1.0) / 4.0) * sr_hat * sr_hat
    if denom_sq <= 0.0:
        raise InputError("PSR denominator is non-positive; check skew/kurtosis inputs")
    z = (sr_hat - sr_benchmark) * np.sqrt(n - 1) / np.sqrt(denom_sq)
    return float(stats.norm.cdf(z))


def effective_n_trials(trial_returns: pd.DataFrame) -> float:
    """Effective number of independent trials via PCA participation ratio.

    For a return matrix with ``N`` columns the correlation matrix has
    eigenvalues ``lambda_i``. The participation ratio
    ``(sum lambda)**2 / sum(lambda**2)`` lies in ``[1, N]`` and counts
    the number of effectively independent dimensions.

    Parameters
    ----------
    trial_returns : pandas.DataFrame
        Returns of candidate trials in columns. Constant columns are
        dropped before computing correlations.

    Returns
    -------
    float
        Participation ratio in ``[1, N_effective]``.
    """
    if not isinstance(trial_returns, pd.DataFrame):
        raise InputError("trial_returns must be a pandas DataFrame")
    if trial_returns.shape[1] == 0:
        raise InputError("trial_returns must have at least one column")
    arr = trial_returns.to_numpy(dtype=float, copy=True)
    # Drop columns that are entirely NaN or constant.
    std = np.nanstd(arr, axis=0, ddof=0)
    keep = std > 0.0
    if not np.any(keep):
        return 1.0
    arr = arr[:, keep]
    arr = arr - np.nanmean(arr, axis=0, keepdims=True)
    arr = np.nan_to_num(arr, nan=0.0)
    arr = arr / np.where(std[keep] > 0.0, std[keep], 1.0)
    # Correlation via X'X / T
    t = max(arr.shape[0], 1)
    corr = (arr.T @ arr) / t
    eigvals = np.linalg.eigvalsh(corr)
    eigvals = np.clip(eigvals, 0.0, None)
    total = float(eigvals.sum())
    sq = float((eigvals * eigvals).sum())
    if sq <= 0.0:
        return 1.0
    n_eff = (total * total) / sq
    return float(np.clip(n_eff, 1.0, arr.shape[1]))


def deflated_sharpe_ratio(
    realized_sr: float,
    n_trials_eff: float,
    sr_trial_variance: float,
    *,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    sample_size: int,
    periods_per_year: int = 252,  # noqa: ARG001 - retained for downstream callers
) -> DSRResult:
    """Deflate the PSR by the expected maximum SR over correlated trials.

    Parameters
    ----------
    realized_sr : float
        The headline Sharpe ratio under consideration.
    n_trials_eff : float
        Effective independent trial count (see :func:`effective_n_trials`).
    sr_trial_variance : float
        Variance of the candidate Sharpe ratios across trials.
    skew : float, default ``0.0``
        Skewness of returns for the PSR denominator.
    kurtosis : float, default ``3.0``
        Raw kurtosis of returns for the PSR denominator.
    sample_size : int
        Number of return observations used to compute ``realized_sr``.
    periods_per_year : int, default ``252``
        Kept for symmetry with the rest of the library; not used inside
        the deflation formula but echoed back to the caller via the
        result object's metadata where applicable.

    Returns
    -------
    DSRResult
        Realised SR, deflated threshold, PSR(threshold), the implied
        DSR value (alias of PSR(threshold)) and its complement p-value.
    """
    if sr_trial_variance < 0.0:
        raise InputError("sr_trial_variance must be non-negative")
    if n_trials_eff < 1.0:
        raise InputError("n_trials_eff must be at least 1")
    if sample_size <= 1:
        raise InputError("sample_size must exceed 1")
    n_eff = max(float(n_trials_eff), 1.0 + 1e-12)
    # Expected maximum of N IID standard-normals.
    q_main = stats.norm.ppf(1.0 - 1.0 / n_eff)
    inner = 1.0 / (n_eff * np.e)
    inner = float(np.clip(inner, 1e-300, 1.0 - 1e-15))
    q_aux = stats.norm.ppf(1.0 - inner)
    expected_max_z = (1.0 - _EULER_MASCHERONI) * q_main + _EULER_MASCHERONI * q_aux
    sr_star = float(np.sqrt(sr_trial_variance)) * float(expected_max_z)
    psr_threshold = probabilistic_sharpe_ratio(
        sr_hat=float(realized_sr),
        sr_benchmark=sr_star,
        n=int(sample_size),
        skew=float(skew),
        kurtosis=float(kurtosis),
    )
    p_value = float(np.clip(1.0 - psr_threshold, 0.0, 1.0))
    return DSRResult(
        realized_sr=float(realized_sr),
        deflated_threshold=sr_star,
        psr_of_threshold=psr_threshold,
        dsr=psr_threshold,
        p_value=p_value,
        n_trials_effective=float(n_eff),
        sample_size=int(sample_size),
    )
