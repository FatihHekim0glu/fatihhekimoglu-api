"""Internal helpers for aligning pairs of price series.

This module exposes a small set of utilities that the public cointegration
functions use to validate and align two input series prior to any
statistical estimation.  Centralising the validation logic keeps the
behaviour consistent across :func:`engle_granger`, :func:`johansen`,
:func:`unit_root_check` and :func:`full_pipeline`.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from pairs._exceptions import DegenerateSeriesError, InputError, InsufficientDataError

_MIN_OBS: Final[int] = 30


def _to_series(x: object, name: str) -> pd.Series:
    """Coerce ``x`` into a :class:`pandas.Series` while preserving its index.

    Parameters
    ----------
    x : array-like
        A NumPy array, list, tuple or pandas Series.
    name : str
        Human-readable label used in any raised exception.

    Returns
    -------
    pandas.Series
        The coerced series, dtype-promoted to ``float64`` when numeric.

    Raises
    ------
    InputError
        If ``x`` is not a recognised one-dimensional numeric container.
    """
    if isinstance(x, pd.Series):
        s = x.astype(float, copy=False)
    elif isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise InputError(f"{name} must be 1-D; got DataFrame with {x.shape[1]} columns")
        s = x.iloc[:, 0].astype(float, copy=False)
    else:
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 1:
            raise InputError(f"{name} must be 1-D; got array with shape {arr.shape}")
        s = pd.Series(arr, name=name)
    return s


def _inner_join_and_dropna(y0: object, y1: object) -> tuple[pd.Series, pd.Series]:
    """Inner-join two series and drop rows where either is missing.

    The result is a pair of aligned :class:`pandas.Series` that share a
    common ascending index.  Both series are required to have at least
    :data:`_MIN_OBS` overlapping non-null observations, otherwise an
    :class:`InsufficientDataError` is raised.

    Parameters
    ----------
    y0, y1 : array-like
        Input price (or log-price) series.

    Returns
    -------
    tuple of pandas.Series
        The aligned ``(y0, y1)`` pair with NaNs dropped.

    Raises
    ------
    InputError
        If either series cannot be coerced or the inputs are not 1-D.
    DegenerateSeriesError
        If either series is constant after alignment.
    InsufficientDataError
        If fewer than 30 overlapping observations remain.
    """
    s0 = _to_series(y0, "y0")
    s1 = _to_series(y1, "y1")

    # Reset positional indices when the user supplied bare arrays so that
    # an alignment can happen on shared positions rather than label-based.
    if not isinstance(y0, pd.Series) and not isinstance(y1, pd.Series):
        if len(s0) != len(s1):
            raise InputError(
                f"y0 and y1 must have the same length when no index is provided; "
                f"got {len(s0)} and {len(s1)}",
            )
        s0 = s0.reset_index(drop=True)
        s1 = s1.reset_index(drop=True)

    df = pd.concat({"y0": s0, "y1": s1}, axis=1, join="inner").dropna(how="any")
    if df.empty:
        raise InputError("y0 and y1 share no overlapping observations after alignment")
    if len(df) < _MIN_OBS:
        raise InsufficientDataError(
            f"need at least {_MIN_OBS} aligned observations; got {len(df)}",
        )

    a, b = df["y0"], df["y1"]
    if float(a.std(ddof=0)) == 0.0:
        raise DegenerateSeriesError("y0 is constant after alignment")
    if float(b.std(ddof=0)) == 0.0:
        raise DegenerateSeriesError("y1 is constant after alignment")
    return a, b
