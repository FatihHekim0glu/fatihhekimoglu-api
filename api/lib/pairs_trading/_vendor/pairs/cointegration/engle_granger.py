"""Engle-Granger two-step cointegration test.

The public :func:`engle_granger` helper runs the standard Engle-Granger
procedure in both directions, keeps the regression with the smaller
p-value, and exposes the slope, intercept and residuals of the chosen
direction.  The p-value of the other direction is preserved alongside so
that callers can detect direction-sensitive specifications.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import coint

from pairs._exceptions import DegenerateSeriesError, InputError

from ._internal.align import _inner_join_and_dropna
from .results import CointegrationResult, TestDirection

_VALID_TRENDS: frozenset[str] = frozenset({"n", "c", "ct", "ctt"})
_VALID_AUTOLAG: frozenset[str] = frozenset({"aic", "bic", "t-stat"})


def _safe_log(x: pd.Series, name: str) -> pd.Series:
    """Apply :func:`numpy.log` after asserting positivity."""
    if (x <= 0).any():
        raise InputError(
            f"use_log=True requires {name} to be strictly positive; "
            "pass use_log=False if you are already supplying log prices",
        )
    return pd.Series(np.log(x.to_numpy()), index=x.index, name=x.name)


def _coint_one_direction(
    dep: pd.Series,
    indep: pd.Series,
    *,
    trend: str,
    autolag: str | None,
) -> tuple[float, float, tuple[float, float, float], pd.Series, float, float]:
    """Run ``statsmodels.tsa.stattools.coint`` for a single ordering.

    Returns
    -------
    tuple
        ``(stat, pvalue, crit_values, residuals, beta, alpha)`` where
        ``residuals = dep - alpha - beta * indep``.
    """
    stat, pvalue, crit = coint(
        dep.to_numpy(),
        indep.to_numpy(),
        trend=trend,
        autolag=autolag,
    )
    # Second-stage OLS so the caller can inspect beta / alpha / residuals.
    if trend == "n":
        design = indep.to_numpy().reshape(-1, 1)
        model = OLS(dep.to_numpy(), design).fit()
        intercept = 0.0
        slope = float(model.params[0])
    else:
        design = add_constant(indep.to_numpy(), has_constant="add")
        model = OLS(dep.to_numpy(), design).fit()
        intercept = float(model.params[0])
        slope = float(model.params[1])

    residuals = pd.Series(
        dep.to_numpy() - intercept - slope * indep.to_numpy(),
        index=dep.index,
        name="resid",
    )
    return (
        float(stat),
        float(pvalue),
        (float(crit[0]), float(crit[1]), float(crit[2])),
        residuals,
        slope,
        intercept,
    )


def engle_granger(
    y0: pd.Series | np.ndarray,
    y1: pd.Series | np.ndarray,
    *,
    trend: Literal["n", "c", "ct", "ctt"] = "c",
    autolag: Literal["aic", "bic", "t-stat"] | None = "bic",
    use_log: bool = True,
    require_i1: bool = True,
) -> CointegrationResult:
    """Run the Engle-Granger two-step cointegration test in both directions.

    Parameters
    ----------
    y0, y1 : pandas.Series or numpy.ndarray
        Aligned price (or log-price) series.  If both supply a
        :class:`~pandas.Index`, the intersection is used.
    trend : {"n", "c", "ct", "ctt"}, default ``"c"``
        Deterministic regressors in the cointegrating regression.
    autolag : {"aic", "bic", "t-stat"} or None, default ``"bic"``
        Lag-length selection for the ADF stage.  Pass ``None`` to use a
        fixed lag of zero.
    use_log : bool, default ``True``
        Apply :func:`numpy.log` to both inputs before testing.
    require_i1 : bool, default ``True``
        Reserved for future enforcement; currently honoured by
        :func:`pairs.cointegration.full_pipeline`.

    Returns
    -------
    CointegrationResult
        The direction with the lower p-value, plus the p-value from the
        reverse regression.

    Raises
    ------
    InputError
        If inputs are malformed or non-positive when ``use_log=True``.
    DegenerateSeriesError
        If either input is constant after alignment.
    """
    if trend not in _VALID_TRENDS:
        raise InputError(f"trend must be one of {sorted(_VALID_TRENDS)}; got {trend!r}")
    if autolag is not None and autolag not in _VALID_AUTOLAG:
        raise InputError(f"autolag must be None or one of {sorted(_VALID_AUTOLAG)}")
    _ = require_i1  # honoured by full_pipeline; placeholder for future logic

    a, b = _inner_join_and_dropna(y0, y1)
    if use_log:
        a = _safe_log(a, "y0")
        b = _safe_log(b, "y1")

    try:
        forward = _coint_one_direction(a, b, trend=trend, autolag=autolag)
        reverse = _coint_one_direction(b, a, trend=trend, autolag=autolag)
    except (np.linalg.LinAlgError, ValueError) as exc:  # pragma: no cover - guard
        raise DegenerateSeriesError(f"OLS solve failed during Engle-Granger: {exc}") from exc

    fwd_stat, fwd_p, fwd_crit, fwd_resid, fwd_beta, fwd_alpha = forward
    rev_stat, rev_p, rev_crit, rev_resid, rev_beta, rev_alpha = reverse

    if fwd_p <= rev_p:
        direction = TestDirection.Y0_ON_Y1
        chosen = (fwd_stat, fwd_p, fwd_crit, fwd_resid, fwd_beta, fwd_alpha)
        other_p = rev_p
    else:
        direction = TestDirection.Y1_ON_Y0
        chosen = (rev_stat, rev_p, rev_crit, rev_resid, rev_beta, rev_alpha)
        other_p = fwd_p

    stat, pvalue, crit_values, residuals, beta, alpha = chosen
    return CointegrationResult(
        stat=stat,
        pvalue=pvalue,
        crit_values=crit_values,
        direction_used=direction,
        beta=beta,
        alpha=alpha,
        residuals=residuals,
        autolag_used=autolag,
        n_obs=len(a),
        pvalue_other_direction=other_p,
    )
