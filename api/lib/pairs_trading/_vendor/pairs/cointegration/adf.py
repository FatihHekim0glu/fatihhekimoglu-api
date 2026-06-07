"""Per-leg I(1) integration check via ADF (with optional DF-GLS).

A series is considered I(1) when the unit-root null cannot be rejected
on the levels but can be rejected on the first differences at the
chosen significance level.  Series that already look stationary on
levels raise an :class:`~pairs._exceptions.InputError` so that callers
do not silently regress returns on returns.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from pairs._exceptions import DegenerateSeriesError, InputError, InsufficientDataError

from .results import UnitRootResult

_VALID_METHODS: frozenset[str] = frozenset({"auto", "adf", "dfgls"})
_MIN_OBS: int = 30


def _adf_pvalue(arr: np.ndarray) -> float:
    """Return only the p-value from :func:`statsmodels.tsa.stattools.adfuller`."""
    _, pvalue, *_ = adfuller(arr, autolag="AIC")
    return float(pvalue)


def _dfgls_pvalue(arr: np.ndarray) -> tuple[float, str]:
    """Run DF-GLS via :mod:`arch`; fall back to ADF when the dep is missing."""
    try:  # pragma: no cover - exercised only when arch is present
        from arch.unitroot import DFGLS

        return float(DFGLS(arr).pvalue), "dfgls"
    except Exception:  # pragma: no cover - exercised when arch missing
        warnings.warn(
            "arch is not installed; falling back to ADF for small-sample unit-root test",
            RuntimeWarning,
            stacklevel=3,
        )
        return _adf_pvalue(arr), "adf"


def unit_root_check(
    x: pd.Series | np.ndarray,
    *,
    method: Literal["auto", "adf", "dfgls"] = "auto",
    alpha: float = 0.05,
    min_obs_for_dfgls: int = 100,
    leg_name: str = "leg",
) -> UnitRootResult:
    """Test whether ``x`` is consistent with an I(1) process.

    Parameters
    ----------
    x : pandas.Series or numpy.ndarray
        Candidate series (typically log prices).
    method : {"auto", "adf", "dfgls"}, default ``"auto"``
        Which unit-root test to use.  ``"auto"`` prefers DF-GLS for
        samples shorter than ``min_obs_for_dfgls`` and falls back to ADF
        when :mod:`arch` is unavailable.
    alpha : float, default ``0.05``
        Significance level used for both the levels rejection guard and
        the differenced-series rejection requirement.
    min_obs_for_dfgls : int, default ``100``
        Sample-size cut-off below which the DF-GLS test is preferred.
    leg_name : str, default ``"leg"``
        Identifier carried into the resulting :class:`UnitRootResult`.

    Returns
    -------
    UnitRootResult
        Levels p-value, differenced-series p-value and I(1) decision.

    Raises
    ------
    InputError
        If the levels series already looks stationary (p < alpha), which
        usually means the caller accidentally passed returns.
    """
    if method not in _VALID_METHODS:
        raise InputError(f"method must be one of {sorted(_VALID_METHODS)}")
    if not (0.0 < alpha < 1.0):
        raise InputError(f"alpha must lie in (0, 1); got {alpha}")
    if min_obs_for_dfgls <= 0:
        raise InputError(f"min_obs_for_dfgls must be positive; got {min_obs_for_dfgls}")

    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise InputError(f"x must be 1-D; got shape {arr.shape}")
    arr = arr[~np.isnan(arr)]
    if arr.size < _MIN_OBS:
        raise InsufficientDataError(
            f"unit_root_check requires at least {_MIN_OBS} observations; got {arr.size}",
        )
    if float(np.std(arr)) == 0.0:
        raise DegenerateSeriesError("x is constant; unit-root test undefined")

    if method == "auto":
        use_dfgls = arr.size < min_obs_for_dfgls
        effective = "dfgls" if use_dfgls else "adf"
    else:
        effective = method

    if effective == "dfgls":
        levels_pvalue, effective = _dfgls_pvalue(arr)
    else:
        levels_pvalue = _adf_pvalue(arr)

    if levels_pvalue < alpha:
        raise InputError(
            "Series appears already stationary; cointegration requires I(1) inputs",
        )

    diff_pvalue = _adf_pvalue(np.diff(arr))
    is_i1 = bool(diff_pvalue < alpha)
    return UnitRootResult(
        leg_name=leg_name,
        levels_pvalue=float(levels_pvalue),
        diff_pvalue=float(diff_pvalue),
        is_i1=is_i1,
        method=effective,
        n_obs=int(arr.size),
    )
