"""Performance statistics for backtest return series.

All functions accept a ``pandas.Series`` of *per-period* simple returns and
return a scalar. Annualization is controlled by the ``ann`` argument (12 for
monthly, 252 for daily, etc.).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as _sps


def _to_excess(r: pd.Series, rf: pd.Series | float) -> pd.Series:
    if isinstance(rf, pd.Series):
        rf_a = rf.reindex(r.index).fillna(0.0)
        return r.sub(rf_a)
    return r.sub(float(rf))


def sharpe_ratio(r: pd.Series, rf: pd.Series | float = 0.0, ann: int = 12) -> float:
    """Annualized Sharpe ratio of an excess-return series.

    Returns 0.0 if the standard deviation is exactly zero (degenerate case).
    """
    excess = _to_excess(r, rf)
    sd = float(excess.std(ddof=1))
    mu = float(excess.mean())
    scale = max(abs(mu), 1.0)
    if not np.isfinite(sd) or sd <= 1e-12 * scale:
        return 0.0
    return float(mu / sd * np.sqrt(ann))


def sortino_ratio(r: pd.Series, rf: pd.Series | float = 0.0, ann: int = 12) -> float:
    """Annualized Sortino ratio using downside semi-deviation (target = 0)."""
    excess = _to_excess(r, rf)
    downside = excess.clip(upper=0.0)
    dd_sd = float(np.sqrt((downside**2).mean()))
    if not np.isfinite(dd_sd) or dd_sd == 0.0:
        return 0.0
    mu = float(excess.mean())
    return float(mu / dd_sd * np.sqrt(ann))


def max_drawdown(r: pd.Series) -> float:
    """Maximum drawdown of the wealth path implied by ``r``.

    Returns a non-positive number (``0.0`` if the path is monotonically
    non-decreasing).
    """
    if len(r) == 0:
        return 0.0
    wealth = (1.0 + r.astype(float)).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return float(dd.min())


def calmar(r: pd.Series, ann: int = 12) -> float:
    """Calmar ratio: annualized arithmetic mean / |max drawdown|.

    Returns 0.0 when the drawdown is zero.
    """
    mdd = max_drawdown(r)
    if mdd == 0.0:
        return 0.0
    ann_ret = float(r.mean()) * ann
    return float(ann_ret / abs(mdd))


def ceq(r: pd.Series, gamma: float) -> float:
    """Per-period certainty-equivalent return under quadratic utility.

    ``CEQ = mu - 0.5 * gamma * sigma^2``.
    """
    mu = float(r.mean())
    var = float(r.var(ddof=1))
    return float(mu - 0.5 * gamma * var)


def jobson_korkie_memmel(r_i: pd.Series, r_n: pd.Series) -> tuple[float, float]:
    """Jobson-Korkie test with the Memmel (2003) correction.

    Tests :math:`H_0`: Sharpe(:math:`r_i`) = Sharpe(:math:`r_n`) against a
    two-sided alternative. Returns ``(z, p_value)``.

    Identical input series yield ``z == 0`` and ``p_value == 1``.
    """
    joined = pd.concat([r_i, r_n], axis=1, join="inner").dropna()
    if joined.shape[0] < 3:
        return 0.0, 1.0
    a = joined.iloc[:, 0].to_numpy(dtype=float)
    b = joined.iloc[:, 1].to_numpy(dtype=float)
    t_obs = a.shape[0]

    mu_i = float(np.mean(a))
    mu_n = float(np.mean(b))
    sig_i = float(np.std(a, ddof=1))
    sig_n = float(np.std(b, ddof=1))
    cov_in = float(np.cov(a, b, ddof=1)[0, 1])

    if sig_i == 0.0 or sig_n == 0.0:
        return 0.0, 1.0

    delta = sig_n * mu_i - sig_i * mu_n
    theta = (1.0 / t_obs) * (
        2.0 * (sig_i**2) * (sig_n**2)
        - 2.0 * sig_i * sig_n * cov_in
        + 0.5 * (mu_i**2) * (sig_n**2)
        + 0.5 * (mu_n**2) * (sig_i**2)
        - (mu_i * mu_n / (sig_i * sig_n)) * (cov_in**2)
    )
    if not np.isfinite(theta) or theta <= 0.0:
        return 0.0, 1.0
    z = delta / float(np.sqrt(theta))
    p = 2.0 * (1.0 - float(_sps.norm.cdf(abs(z))))
    p = max(0.0, min(1.0, p))
    return float(z), float(p)


def ledoit_wolf_bootstrap_pvalue(
    r_i: pd.Series,
    r_n: pd.Series,
    *,
    B: int = 999,
    block_length: int = 5,
    seed: int = 0,
) -> float:
    """Stationary-block-bootstrap p-value for the Sharpe-difference test.

    Uses Politis-Romano fixed-block sampling under the null that the two
    Sharpe ratios are equal. Returns a two-sided p-value in ``[0, 1]``.
    """
    joined = pd.concat([r_i, r_n], axis=1, join="inner").dropna()
    if joined.shape[0] < block_length + 1:
        return 1.0
    a = joined.iloc[:, 0].to_numpy(dtype=float)
    b = joined.iloc[:, 1].to_numpy(dtype=float)
    n = a.shape[0]

    def _sharpe_diff(x: np.ndarray, y: np.ndarray) -> float:
        sx = float(np.std(x, ddof=1))
        sy = float(np.std(y, ddof=1))
        if sx == 0.0 or sy == 0.0:
            return 0.0
        return float(np.mean(x) / sx - np.mean(y) / sy)

    obs = _sharpe_diff(a, b)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_length))

    count = 0
    for _ in range(B):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(s + np.arange(block_length)) % n for s in starts])[:n]
        a_b = a[idx]
        b_b = b[idx]
        # Recenter under H0: equal Sharpes -> compare against observed.
        diff_boot = _sharpe_diff(a_b, b_b) - obs
        if abs(diff_boot) >= abs(obs):
            count += 1
    p = (count + 1) / (B + 1)
    return float(max(0.0, min(1.0, p)))
