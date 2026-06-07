"""Leak-free z-score transformations of a spread series."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pairs._exceptions import InputError
from pairs.spread.results import OUResult

__all__ = ["zscore"]


def zscore(
    spread: pd.Series,
    window: int | None = None,
    *,
    use_ou: bool = False,
    ou_result: OUResult | None = None,
) -> pd.Series:
    """Standardise a spread using either rolling statistics or an OU fit.

    Three modes are supported:

    * ``use_ou=True`` and ``ou_result`` provided:
      ``(spread - mu) / sigma_eq``, the canonical OU-based z-score.
    * ``window`` provided (with or without ``ou_result``): trailing rolling
      mean and standard deviation with ``min_periods=window``.
    * ``window is None`` and ``ou_result`` provided (``use_ou=False``):
      window is auto-picked as ``max(2, round(2 * half_life))``.

    Parameters
    ----------
    spread : pandas.Series
        Spread series.
    window : int, optional
        Rolling window length. If ``None`` and an ``ou_result`` is provided,
        the window is derived from the half-life as ``2 * H``.
    use_ou : bool, default ``False``
        Force the OU mode. Requires ``ou_result``.
    ou_result : OUResult, optional
        Fitted OU dynamics, used either for the OU-mode standardisation or to
        auto-pick the rolling window.

    Returns
    -------
    pandas.Series
        Z-scored spread on the same index. Initial observations within the
        warm-up window are ``NaN``.

    Raises
    ------
    pairs.InputError
        If neither ``window`` nor ``ou_result`` is provided, or if
        ``use_ou=True`` without an ``ou_result``, or if ``window`` is < 2.

    Notes
    -----
    Rolling statistics use the default trailing semantics of
    :meth:`pandas.Series.rolling` -- the window covers ``[t - w + 1, t]`` and
    so includes the current observation only, with no future leakage.
    ``min_periods=window`` blanks the warm-up period so that early z-scores
    are not biased by a partial sample.
    """

    if not isinstance(spread, pd.Series):
        msg = "spread must be a pandas Series"
        raise InputError(msg)
    if use_ou:
        if ou_result is None:
            msg = "use_ou=True requires an ou_result"
            raise InputError(msg)
        sigma_eq = float(ou_result.sigma_eq)
        if sigma_eq <= 0.0:
            msg = f"ou_result.sigma_eq must be positive, got {sigma_eq!r}"
            raise InputError(msg)
        return (spread - float(ou_result.mu)) / sigma_eq

    if window is None:
        if ou_result is None:
            msg = (
                "zscore needs either a window or an ou_result (with use_ou=True "
                "for stationary z-scores, or use_ou=False to auto-pick a window)"
            )
            raise InputError(msg)
        window = max(2, int(round(2.0 * float(ou_result.half_life))))
    if int(window) < 2:
        msg = f"window must be at least 2, got {window!r}"
        raise InputError(msg)
    roll = spread.rolling(window=int(window), min_periods=int(window))
    mean = roll.mean()
    std = roll.std(ddof=1)
    out = (spread - mean) / std.replace(0.0, np.nan)
    out.name = spread.name
    return out
