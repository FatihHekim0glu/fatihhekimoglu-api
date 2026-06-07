"""KPSS stationarity test wrapper for candidate spreads."""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import kpss

from pairs._exceptions import DegenerateSeriesError, InputError, InsufficientDataError

from .results import KPSSResult

_BOUNDARY_PVALUES: tuple[float, float] = (0.01, 0.10)
_VALID_REGRESSION: frozenset[str] = frozenset({"c", "ct"})
_MIN_OBS: int = 30


def kpss_spread(
    spread: pd.Series | np.ndarray,
    *,
    regression: Literal["c", "ct"] = "c",
    nlags: int | Literal["auto", "legacy"] = "auto",
    alpha: float = 0.05,
) -> KPSSResult:
    """Run the KPSS test on a candidate cointegrating spread.

    Parameters
    ----------
    spread : pandas.Series or numpy.ndarray
        Residual / spread series to be tested for stationarity.
    regression : {"c", "ct"}, default ``"c"``
        Whether to include a constant (``"c"``) or constant plus trend
        (``"ct"``).
    nlags : int or {"auto", "legacy"}, default ``"auto"``
        Lag-truncation rule forwarded to :func:`statsmodels.tsa.stattools.kpss`.
    alpha : float, default ``0.05``
        Significance level used to derive :attr:`KPSSResult.is_stationary`.

    Returns
    -------
    KPSSResult
        Statistic, p-value (with interpolation flag) and stationarity
        decision.
    """
    if regression not in _VALID_REGRESSION:
        raise InputError(f"regression must be in {sorted(_VALID_REGRESSION)}")
    if not (0.0 < alpha < 1.0):
        raise InputError(f"alpha must lie in (0, 1); got {alpha}")

    arr = np.asarray(spread, dtype=float)
    if arr.ndim != 1:
        raise InputError(f"spread must be 1-D; got shape {arr.shape}")
    arr = arr[~np.isnan(arr)]
    if arr.size < _MIN_OBS:
        raise InsufficientDataError(
            f"kpss requires at least {_MIN_OBS} observations; got {arr.size}",
        )
    if float(np.std(arr)) == 0.0:
        raise DegenerateSeriesError("spread is constant; KPSS undefined")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InterpolationWarning)
        stat, pvalue, nlags_used, crit_dict = kpss(arr, regression=regression, nlags=nlags)
    interpolated = any(issubclass(w.category, InterpolationWarning) for w in caught)
    if pvalue in _BOUNDARY_PVALUES:
        interpolated = True

    return KPSSResult(
        stat=float(stat),
        pvalue=float(pvalue),
        pvalue_interpolated=bool(interpolated),
        crit_values={str(k): float(v) for k, v in crit_dict.items()},
        nlags_used=int(nlags_used),
        is_stationary=bool(pvalue > alpha),
    )
