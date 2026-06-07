"""Johansen rank test wrapper.

This is a thin wrapper around
:func:`statsmodels.tsa.vector_ar.vecm.coint_johansen` that normalises
the output into a :class:`~pairs.cointegration.results.JohansenResult`
and applies a simple, conservative rank decision rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from pairs._exceptions import DegenerateSeriesError, InputError, InsufficientDataError

from .results import JohansenResult

_VALID_DET_ORDERS: frozenset[int] = frozenset({-1, 0, 1})
_MIN_OBS: int = 30


def _to_endog_frame(endog: object) -> pd.DataFrame:
    """Coerce ``endog`` into a 2-D :class:`pandas.DataFrame`."""
    if isinstance(endog, pd.DataFrame):
        df = endog.astype(float, copy=False)
    else:
        arr = np.asarray(endog, dtype=float)
        if arr.ndim != 2:
            raise InputError(f"endog must be 2-D; got shape {arr.shape}")
        df = pd.DataFrame(arr)
    df = df.dropna(how="any")
    if df.shape[1] < 2:
        raise InputError(f"johansen requires at least 2 columns; got {df.shape[1]}")
    if df.shape[0] < _MIN_OBS:
        raise InsufficientDataError(
            f"johansen requires at least {_MIN_OBS} observations; got {df.shape[0]}",
        )
    if (df.std(ddof=0) == 0).any():
        raise DegenerateSeriesError("one or more columns of endog are constant")
    return df


def johansen(
    endog: pd.DataFrame | NDArray[np.float64],
    *,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> JohansenResult:
    """Run the Johansen trace / max-eigenvalue rank tests.

    Parameters
    ----------
    endog : pandas.DataFrame or array-like
        Wide-format matrix where each column is one I(1) series.
    det_order : {-1, 0, 1}, default ``0``
        Deterministic-trend specification forwarded to statsmodels.
    k_ar_diff : int, default ``1``
        Number of lagged differences in the VECM representation.

    Returns
    -------
    JohansenResult
        Trace and max-eigenvalue statistics with their 5% critical
        values, the implied cointegration rank, eigenvectors and
        sample size.  For two-asset baskets the rank is capped at 1.
    """
    if det_order not in _VALID_DET_ORDERS:
        raise InputError(f"det_order must be in {sorted(_VALID_DET_ORDERS)}")
    if k_ar_diff < 0:
        raise InputError(f"k_ar_diff must be non-negative; got {k_ar_diff}")

    df = _to_endog_frame(endog)
    res = coint_johansen(df.to_numpy(), det_order, k_ar_diff)

    trace_stats = np.asarray(res.lr1, dtype=float)
    trace_crit = np.asarray(res.cvt[:, 1], dtype=float)  # 5% column
    max_eig_stats = np.asarray(res.lr2, dtype=float)
    max_eig_crit = np.asarray(res.cvm[:, 1], dtype=float)
    eigenvectors = np.asarray(res.evec, dtype=float)

    rank = 0
    for r in range(len(trace_stats)):
        if trace_stats[r] > trace_crit[r]:
            rank = r + 1
    rank = min(rank, df.shape[1] - 1) if df.shape[1] == 2 else rank

    return JohansenResult(
        trace_stats=trace_stats,
        trace_crit_95=trace_crit,
        max_eig_stats=max_eig_stats,
        max_eig_crit_95=max_eig_crit,
        rank=rank,
        eigenvectors=eigenvectors,
        n_obs=df.shape[0],
    )
