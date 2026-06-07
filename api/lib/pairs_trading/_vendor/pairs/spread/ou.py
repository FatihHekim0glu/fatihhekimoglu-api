"""Discrete-time MLE of an Ornstein-Uhlenbeck process on a spread series."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from pairs._exceptions import DegenerateSeriesError, InputError, InsufficientDataError
from pairs.spread.results import OUResult

__all__ = ["fit_ou"]

_MIN_OBS: int = 50


def fit_ou(spread: pd.Series, *, dt: float = 1.0) -> OUResult:
    """Fit ``dS_t = theta * (mu - S_t) dt + sigma dW_t`` to ``spread``.

    The discrete-time exact AR(1) representation is
    ``S_t = c + phi * S_{t-1} + eps_t`` with ``phi = exp(-theta * dt)`` and
    ``c = mu * (1 - phi)``. The function regresses ``S_t`` on ``S_{t-1}`` via
    :func:`statsmodels.api.OLS` and back-transforms to ``(theta, mu, sigma)``.

    Parameters
    ----------
    spread : pandas.Series
        Spread series. Leading and trailing NaNs are dropped before fitting;
        internal NaNs raise :class:`pairs.InputError` because the AR(1) lag
        structure would otherwise be ambiguous.
    dt : float, default ``1.0``
        Sampling step. Use ``1.0`` for "per observation" units, ``1/252`` for
        annualised time on daily data.

    Returns
    -------
    OUResult
        Fully populated result with both AR(1) and OU representations.

    Raises
    ------
    pairs.InputError
        Non-Series input, non-positive ``dt``, or internal NaNs.
    pairs.InsufficientDataError
        Fewer than 50 usable observations.
    pairs.DegenerateSeriesError
        Estimated ``phi`` outside ``(0, 1)`` -- the spread is not mean-reverting
        in the OU sense and the back-transform would be ill-defined. Also raised
        when ``phi >= 0.999``, indicating a near-unit-root (random-walk-like)
        series for which the OU fit is not meaningful.

    Notes
    -----
    The OLS / AR(1) estimator of ``phi`` is known to be biased downward in
    finite samples, with the bias growing as ``phi -> 1`` (slow mean reversion).
    For slow-reversion regimes (e.g. half-life on the order of 40+ observations)
    on short samples (n ~ 500), the recovered ``half_life`` may carry a
    relative error of 30-40%. No bias correction is applied here; consumers
    needing tight half-life accuracy in the slow-reversion regime should
    supply longer samples or apply a downstream Yu-Phillips style correction.
    """

    if not isinstance(spread, pd.Series):
        msg = "spread must be a pandas Series"
        raise InputError(msg)
    if float(dt) <= 0.0:
        msg = f"dt must be positive, got {dt!r}"
        raise InputError(msg)
    series = spread.dropna()
    if series.isna().any():
        msg = "spread must not contain interior NaNs"
        raise InputError(msg)
    if series.shape[0] < _MIN_OBS:
        msg = f"fit_ou needs at least {_MIN_OBS} observations, got {series.shape[0]}"
        raise InsufficientDataError(msg)

    values = series.to_numpy(dtype=np.float64)
    s_lag = values[:-1]
    s_now = values[1:]
    design = sm.add_constant(s_lag, has_constant="add")
    fit = sm.OLS(s_now, design).fit()
    intercept = float(fit.params[0])
    phi = float(fit.params[1])
    if not 0.0 < phi < 1.0:
        msg = (
            f"AR(1) coefficient phi={phi:.6f} outside (0, 1); spread is not "
            "OU-mean-reverting on this sample"
        )
        raise DegenerateSeriesError(msg)
    if phi >= 0.99:
        msg = "Series appears non-stationary; OU fit not meaningful"
        raise DegenerateSeriesError(msg)
    # In the suspect zone (phi very close to 1) corroborate with ADF on levels;
    # a series that fails to reject the unit-root null is non-stationary and
    # fitting OU would yield a meaningless half-life. The phi-gate avoids the
    # well-known low power of ADF on legitimately slow-reverting OU samples.
    if phi >= 0.97:
        try:
            from statsmodels.tsa.stattools import adfuller  # local import keeps OU module cheap

            adf_pvalue = float(adfuller(values, autolag="BIC", regression="c")[1])
        except Exception:
            adf_pvalue = 0.0
        if adf_pvalue > 0.30:
            msg = (
                f"Series fails ADF stationarity test (p={adf_pvalue:.3f}) at "
                f"phi={phi:.4f}; OU fit would produce a meaningless half-life"
            )
            raise DegenerateSeriesError(msg)

    theta = -np.log(phi) / float(dt)
    mu = intercept / (1.0 - phi)
    resid_arr = np.asarray(fit.resid, dtype=np.float64)
    resid_var = float(resid_arr.var(ddof=1))
    sigma_sq = resid_var * 2.0 * theta / (1.0 - phi * phi)
    sigma_sq = max(sigma_sq, 1e-30)
    sigma = float(np.sqrt(sigma_sq))
    sigma_eq = float(sigma / np.sqrt(2.0 * theta))
    half_life = float(np.log(2.0) / theta)
    residuals = pd.Series(resid_arr, index=series.index[1:], name=f"resid_ou({series.name})")
    return OUResult(
        theta=float(theta),
        mu=float(mu),
        sigma=float(sigma),
        sigma_eq=float(sigma_eq),
        half_life=float(half_life),
        phi=float(phi),
        intercept=float(intercept),
        residuals=residuals,
        log_likelihood=float(fit.llf),
        dt=float(dt),
        n_obs=int(s_now.shape[0]),
    )
