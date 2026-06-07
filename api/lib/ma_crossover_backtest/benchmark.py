"""Statistical comparison of a strategy against a benchmark.

Three tests, each answering a distinct question:

1. CAPM regression with Newey-West HAC standard errors  ->  alpha and beta
   "Does the strategy earn return unexplained by exposure to the benchmark?"

2. Information ratio + implied t-stat (Grinold's identity)
   "Is active return per unit of active risk meaningful?"

3. Memmel-corrected Jobson-Korkie Sharpe difference test
   "Are the two Sharpe ratios statistically distinguishable?"

HAC standard errors are non-negotiable: MA-crossover strategies hold positions
for days/weeks, so the residuals are serially correlated and naive OLS
standard errors are biased downward (over-rejection of the null alpha = 0).
The Newey-West bandwidth follows Andrews (1991): floor(4*(T/100)^(2/9)).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from ma_backtester.config import TRADING_DAYS_PER_YEAR
from ma_backtester.results import BenchmarkComparison


def _andrews_hac_lags(n_obs: int) -> int:
    return max(1, math.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))


def capm_regression(
    *,
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_annual: float = 0.0,
    hac_lags: int | None = None,
) -> tuple[float, float, float, float, float, int]:
    """Run excess-return CAPM with Newey-West HAC standard errors.

    Returns
    -------
    (alpha_annual, alpha_t, alpha_p, beta, beta_se, hac_lags_used)
    """
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1, join="inner").dropna()
    aligned.columns = ["strat", "bench"]

    n_obs = len(aligned)
    if n_obs < 30:
        raise ValueError(f"Need at least 30 aligned observations, got {n_obs}")

    rf_daily = (1.0 + risk_free_annual) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    y = aligned["strat"] - rf_daily
    x = sm.add_constant(aligned["bench"] - rf_daily)

    lags = hac_lags if hac_lags is not None else _andrews_hac_lags(n_obs)
    model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": lags})

    alpha_daily = float(model.params.iloc[0])
    beta = float(model.params.iloc[1])
    alpha_t = float(model.tvalues.iloc[0])
    alpha_p = float(model.pvalues.iloc[0])
    beta_se = float(model.bse.iloc[1])
    alpha_annual = alpha_daily * TRADING_DAYS_PER_YEAR

    return alpha_annual, alpha_t, alpha_p, beta, beta_se, lags


def information_ratio(
    *,
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> tuple[float, float, float]:
    """Returns (IR, tracking_error_annual, active_return_annual)."""
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1, join="inner").dropna()
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    te_daily = float(active.std(ddof=1))
    if te_daily < 1e-12:
        return float("nan"), 0.0, float(active.mean() * TRADING_DAYS_PER_YEAR)
    ar_annual = float(active.mean() * TRADING_DAYS_PER_YEAR)
    te_annual = float(te_daily * math.sqrt(TRADING_DAYS_PER_YEAR))
    ir = ar_annual / te_annual
    return ir, te_annual, ar_annual


def memmel_sharpe_difference_test(
    *,
    returns_a: pd.Series,
    returns_b: pd.Series,
) -> tuple[float, float]:
    """Memmel-corrected Jobson-Korkie test for equal Sharpe ratios.

    Memmel (2003) corrected a typo in Jobson & Korkie (1981). Tests
    H0: SR_A = SR_B against a two-sided alternative.

    Asymptotic variance per Memmel (2003) / Ledoit & Wolf (2008) eq. 1::

        theta = ( 2*(1 - rho) + 0.5*(SR_a^2 + SR_b^2)
                  - 0.5*SR_a*SR_b*(1 + rho^2) ) / n

    where rho = cov(a,b) / (sig_a * sig_b). At rho = 1 with SR_a = SR_b
    the expression collapses to zero, as required.

    Note: assumes iid-normal returns. Under heavy autocorrelation the test
    is liberal; a stationary-bootstrap (Ledoit & Wolf 2008) is more robust
    but materially more code. Memmel-JK is the standard analytic choice.

    Returns
    -------
    (sharpe_diff_annualised, two_sided_p_value)
    """
    aligned = pd.concat([returns_a, returns_b], axis=1, join="inner").dropna()
    a = aligned.iloc[:, 0]
    b = aligned.iloc[:, 1]
    n = len(a)
    if n < 30:
        raise ValueError(f"Need at least 30 aligned observations, got {n}")

    mu_a = float(a.mean())
    mu_b = float(b.mean())
    var_a = float(a.var(ddof=1))
    var_b = float(b.var(ddof=1))
    cov_ab = float(np.cov(a, b, ddof=1)[0, 1])
    sig_a = math.sqrt(var_a)
    sig_b = math.sqrt(var_b)
    if (
        sig_a < 1e-12 or sig_b < 1e-12
    ):  # pragma: no cover — defensive guard, unreachable from valid samples
        return float("nan"), float("nan")

    sr_a_daily = mu_a / sig_a
    sr_b_daily = mu_b / sig_b
    diff_daily = sr_a_daily - sr_b_daily
    rho = cov_ab / (sig_a * sig_b)

    theta = (
        2.0 * (1.0 - rho)
        + 0.5 * (sr_a_daily**2 + sr_b_daily**2)
        - 0.5 * sr_a_daily * sr_b_daily * (1.0 + rho**2)
    ) / n

    diff_annualised = diff_daily * math.sqrt(TRADING_DAYS_PER_YEAR)

    # Degenerate cases: zero diff and zero variance both come from identical
    # (or near-identical) series. The two-sided p-value of "no difference"
    # against itself is 1.0 — flag it as such, not NaN.
    if abs(diff_daily) < 1e-15 and theta <= 0:
        return diff_annualised, 1.0
    if (
        theta <= 0
    ):  # pragma: no cover — non-zero diff but zero variance is mathematically pathological
        return diff_annualised, float("nan")

    z = diff_daily / math.sqrt(theta)
    p_two_sided = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return diff_annualised, float(p_two_sided)


def compare_strategies(
    *,
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_annual: float = 0.0,
) -> BenchmarkComparison:
    """Run CAPM alpha, information ratio, and Memmel-JK Sharpe-diff jointly.

    Aligns the two series on inner-join, then runs all three tests and
    bundles the results into a single dataclass. HAC bandwidth follows
    Andrews (1991).

    Parameters
    ----------
    strategy_returns, benchmark_returns : pd.Series
        Daily simple returns. Misaligned or NaN rows are dropped.
    risk_free_annual : float
        Used only by the CAPM regression's excess-return transform.

    Returns
    -------
    BenchmarkComparison
        Alpha (annualised), t-statistic, p-value, beta, beta SE,
        information ratio, tracking error, active return, Sharpe difference
        + p-value, ``n_observations`` and ``hac_lags``.

    Raises
    ------
    ValueError
        Fewer than 30 aligned observations (propagated from the underlying
        regression / test functions).

    References
    ----------
    Newey & West (1987); Andrews (1991); Memmel (2003); Grinold (1989).
    """
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1, join="inner").dropna()
    n_obs = len(aligned)

    alpha_ann, alpha_t, alpha_p, beta, beta_se, lags = capm_regression(
        strategy_returns=strategy_returns,
        benchmark_returns=benchmark_returns,
        risk_free_annual=risk_free_annual,
    )
    ir, te_annual, ar_annual = information_ratio(
        strategy_returns=strategy_returns,
        benchmark_returns=benchmark_returns,
    )
    sharpe_diff, sharpe_p = memmel_sharpe_difference_test(
        returns_a=strategy_returns,
        returns_b=benchmark_returns,
    )

    return BenchmarkComparison(
        alpha_annual=alpha_ann,
        alpha_t_stat=alpha_t,
        alpha_p_value=alpha_p,
        beta=beta,
        beta_se=beta_se,
        information_ratio=ir,
        tracking_error_annual=te_annual,
        active_return_annual=ar_annual,
        sharpe_diff=sharpe_diff,
        sharpe_diff_p_value=sharpe_p,
        n_observations=n_obs,
        hac_lags=lags,
    )
