"""Static hedge-ratio estimation by ordinary and total least squares.

Two estimators are exposed:

* :func:`ols_hedge` -- classical OLS, asymmetric in ``(y, x)``.
* :func:`tls_hedge` -- total least squares (orthogonal regression), symmetric
  up to sign, useful when both legs of the pair carry measurement noise.

Both functions return a :class:`pairs.spread.results.HedgeResult`. The TLS
estimator is sign-aligned to OLS post-fit so downstream code (sign of beta,
direction of spread) is consistent across estimators.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import statsmodels.api as sm

from pairs._exceptions import DegenerateSeriesError, InputError, InsufficientDataError
from pairs.spread.results import HedgeResult

if TYPE_CHECKING:
    pass

__all__ = ["ols_hedge", "tls_hedge"]

_MIN_OBS: int = 3


def _prepare(y: pd.Series, x: pd.Series, *, use_log: bool) -> tuple[pd.Series, pd.Series]:
    """Align ``y`` and ``x``, optionally take logs, drop NaNs, validate.

    Parameters
    ----------
    y, x : pandas.Series
        Input price series. Indexed alignment is enforced via ``concat``.
    use_log : bool
        When ``True``, returns ``log(y), log(x)`` after asserting positivity.

    Returns
    -------
    tuple of pandas.Series
        Aligned, optionally log-transformed series with shared index.
    """

    if not isinstance(y, pd.Series) or not isinstance(x, pd.Series):
        msg = "y and x must be pandas Series"
        raise InputError(msg)
    frame = pd.concat([y.rename("y"), x.rename("x")], axis=1, join="inner").dropna()
    if frame.shape[0] < _MIN_OBS:
        msg = f"need at least {_MIN_OBS} aligned observations, got {frame.shape[0]}"
        raise InsufficientDataError(msg)
    y_a = frame["y"]
    x_a = frame["x"]
    if use_log:
        if (y_a <= 0).any() or (x_a <= 0).any():
            msg = "use_log=True requires strictly positive prices in y and x"
            raise InputError(msg)
        y_a = np.log(y_a)
        x_a = np.log(x_a)
    if float(np.std(x_a.to_numpy())) == 0.0 or float(np.std(y_a.to_numpy())) == 0.0:
        msg = "input series is constant; hedge ratio is undefined"
        raise DegenerateSeriesError(msg)
    return y_a, x_a


def ols_hedge(y: pd.Series, x: pd.Series, *, use_log: bool = True) -> HedgeResult:
    """Estimate a hedge ratio by ordinary least squares.

    Fits the model ``y_t = alpha + beta * x_t + eps_t`` using
    :func:`statsmodels.api.OLS` with an explicit intercept column.

    Parameters
    ----------
    y, x : pandas.Series
        Dependent and independent price series respectively. Aligned on their
        index; NaN rows are dropped.
    use_log : bool, default ``True``
        Regress on log prices. Pairs trading almost always uses log space so
        the spread is interpretable as a return.

    Returns
    -------
    HedgeResult
        Fitted hedge ratio, residuals, and R-squared.

    Raises
    ------
    pairs.InputError
        If ``y`` or ``x`` is not a :class:`pandas.Series`, or if ``use_log`` is
        requested on non-positive data.
    pairs.InsufficientDataError
        Fewer than three aligned observations.
    pairs.DegenerateSeriesError
        Either series is constant.
    """

    y_p, x_p = _prepare(y, x, use_log=use_log)
    design = sm.add_constant(x_p.to_numpy(), has_constant="add")
    fit = sm.OLS(y_p.to_numpy(), design).fit()
    alpha = float(fit.params[0])
    beta = float(fit.params[1])
    residuals = pd.Series(fit.resid, index=y_p.index, name=f"resid_ols({y.name},{x.name})")
    r_squared = float(min(max(float(fit.rsquared), 0.0), 1.0))
    return HedgeResult(
        alpha=alpha,
        beta=beta,
        residuals=residuals,
        r_squared=r_squared,
        method="ols",
        direction=f"{y.name}~{x.name}",
        use_log=use_log,
        n_obs=int(y_p.shape[0]),
    )


def tls_hedge(y: pd.Series, x: pd.Series, *, use_log: bool = True) -> HedgeResult:
    """Estimate a hedge ratio by total (orthogonal) least squares.

    The estimator minimises perpendicular distance from ``(x_i, y_i)`` to the
    fitted line. It is computed via the SVD of the centred design matrix
    ``M = [x_c, y_c]``: the smallest right-singular vector points along the
    direction orthogonal to the regression line and so encodes ``beta``.

    Sign convention: the raw SVD solution is sign-aligned to the OLS slope on
    the same data so that ``sign(beta_tls) == sign(beta_ols)`` always holds.

    Parameters
    ----------
    y, x : pandas.Series
        Dependent and independent price series.
    use_log : bool, default ``True``
        Regress on log prices.

    Returns
    -------
    HedgeResult
        Fitted hedge ratio with orthogonal-projection residuals.
    """

    y_p, x_p = _prepare(y, x, use_log=use_log)
    x_arr = x_p.to_numpy(dtype=np.float64)
    y_arr = y_p.to_numpy(dtype=np.float64)
    x_mean = float(x_arr.mean())
    y_mean = float(y_arr.mean())
    x_c = x_arr - x_mean
    y_c = y_arr - y_mean
    m = np.column_stack([x_c, y_c])
    _, _, vt = np.linalg.svd(m, full_matrices=False)
    v_last = vt[-1]
    if abs(float(v_last[1])) < 1e-15:
        msg = "TLS minor right-singular vector has zero y-component; degenerate"
        raise DegenerateSeriesError(msg)
    beta = -float(v_last[0]) / float(v_last[1])
    # Orthogonal projection residuals: signed perpendicular distance.
    norm = float(np.hypot(v_last[0], v_last[1]))
    raw_resid = (v_last[0] * x_c + v_last[1] * y_c) / norm
    # Sign-align to OLS so downstream sign conventions stay stable.
    ols_fit = ols_hedge(y, x, use_log=use_log)
    sign = 1.0 if float(np.sign(ols_fit.beta)) >= 0.0 else -1.0
    if float(np.sign(beta)) != sign:
        beta = -beta
        raw_resid = -raw_resid
    alpha = y_mean - beta * x_mean
    residuals = pd.Series(raw_resid, index=y_p.index, name=f"resid_tls({y.name},{x.name})")
    # Pseudo-R^2 for TLS: 1 - SS_orth / SS_total_y.
    ss_total = float(np.dot(y_c, y_c))
    ss_resid = float(np.dot(raw_resid, raw_resid))
    if ss_total == 0.0:
        r_squared = 0.0
    else:
        r_squared = float(min(max(1.0 - ss_resid / ss_total, 0.0), 1.0))
    return HedgeResult(
        alpha=float(alpha),
        beta=float(beta),
        residuals=residuals,
        r_squared=r_squared,
        method="tls",
        direction=f"{y.name}~{x.name}",
        use_log=use_log,
        n_obs=int(y_p.shape[0]),
    )
