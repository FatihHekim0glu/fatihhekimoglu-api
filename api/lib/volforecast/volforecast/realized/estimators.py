"""Realized-volatility estimators and the forward RV-window target builder.

This module computes daily realized-volatility (RV) proxies from an OHLC frame
(Parkinson and Garman-Klass range estimators) and from a close-to-close return
series, plus the *forward* RV target used by every model: the realized volatility
over a strictly future window ``(t + gap, t + gap + h]``. The explicit ``gap``
between the feature timestamp and the target window is the leakage guard — a
feature observed at ``t`` may never see a return inside its own target window.

All estimators are scale-aware (RV scales linearly with the return scale) and
contain no fit/network/randomness. Importing this module has no side effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from volforecast._exceptions import ValidationError
from volforecast._validation import ensure_dataframe, ensure_series

#: The OHLC columns every range-based estimator requires (lower-cased on input).
OHLC_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")

#: Parkinson scaling constant ``1 / (4 ln 2)`` applied to the squared log range.
_PARKINSON_FACTOR: float = 1.0 / (4.0 * np.log(2.0))

#: Garman-Klass close-minus-open term coefficient ``2 ln 2 - 1``.
_GK_CO_FACTOR: float = 2.0 * np.log(2.0) - 1.0

#: The estimators exposed via :func:`realized_volatility`.
_ESTIMATORS: tuple[str, ...] = ("close_to_close", "parkinson", "garman_klass")


def _validate_window(window: int) -> None:
    """Raise :class:`ValidationError` unless ``window`` is an integer ``>= 1``."""
    if not isinstance(window, int) or isinstance(window, bool):
        raise ValidationError(f"window must be an int, got {type(window).__name__}.")
    if window < 1:
        raise ValidationError(f"window must be >= 1, got {window}.")


def _log_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Return ``ln(numerator / denominator)`` as a strictly-typed float64 Series.

    ``np.log`` on a Series is typed ``Any`` by pandas-stubs; this rebuilds a typed
    Series at a single boundary so the range estimators stay strict downstream.
    """
    ratio = numerator.to_numpy(dtype="float64") / denominator.to_numpy(dtype="float64")
    return pd.Series(np.log(ratio), index=numerator.index, dtype="float64")


def _rolling_rms(squared: pd.Series, window: int, *, name: str) -> pd.Series:
    """Root of the rolling mean of a per-day variance contribution.

    ``squared`` is a per-day (non-negative) variance proxy; this returns the
    square root of its trailing mean over ``window`` observations, NaN over the
    warm-up. The result is a volatility (same units as a return), so it scales
    linearly with the underlying return scale.
    """
    mean_var = squared.rolling(window=window, min_periods=window).mean()
    # ``np.sqrt`` of a Series returns ``Any`` under pandas-stubs; rebuild a typed
    # Series at this single boundary so the rest of the module stays strict.
    rv = pd.Series(np.sqrt(mean_var.to_numpy(dtype="float64")), index=squared.index)
    return rv.rename(name)


def _require_ohlc(ohlc: pd.DataFrame, columns: tuple[str, ...], *, name: str) -> pd.DataFrame:
    """Coerce, lower-case, and validate the OHLC columns a range estimator needs.

    Returns a float64 frame restricted to ``columns`` (in canonical order) with
    a checked positivity and ``high >= low`` invariant. NaN is permitted so that
    estimators can run on series with warm-up / missing bars at the boundary.
    """
    frame = ensure_dataframe(ohlc, name=name, allow_nan=True)
    frame = frame.rename(columns={c: str(c).lower() for c in frame.columns})

    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValidationError(
            f"{name} is missing required column(s): {missing}. "
            f"Expected (case-insensitive): {list(columns)}."
        )
    frame = frame.loc[:, list(columns)]

    finite = frame.dropna()
    if not finite.empty:
        if not bool((finite.to_numpy() > 0.0).all()):
            raise ValidationError(f"{name} contains non-positive prices.")
        if (
            "high" in columns
            and "low" in columns
            and not bool((finite["high"] >= finite["low"]).all())
        ):
            raise ValidationError(f"{name} has a bar with high < low.")
    return frame


def close_to_close_rv(close: pd.Series, *, window: int = 1) -> pd.Series:
    r"""Close-to-close realized volatility over a trailing ``window``.

    Computes log returns ``r_t = ln(C_t / C_{t-1})`` and returns the rolling
    root-mean-square return over ``window`` observations (the simplest RV proxy):

    .. math::

        \mathrm{RV}^{cc}_t = \sqrt{\frac{1}{\text{window}}
                              \sum_{i=0}^{\text{window}-1} r_{t-i}^2}.

    With ``window=1`` this is ``|r_t|`` (absolute log return), the squared form of
    which is the canonical noisy one-day variance proxy.

    Parameters
    ----------
    close:
        A 1-D series of close prices indexed by date.
    window:
        Number of trailing observations to aggregate (``>= 1``).

    Returns
    -------
    pandas.Series
        The close-to-close RV series, named ``"rv_cc"``, NaN over the warm-up.

    Raises
    ------
    ValidationError
        If ``close`` is empty, contains non-positive prices, or ``window < 1``.
    """
    _validate_window(window)
    prices = ensure_series(close, name="close", allow_nan=True)

    finite = prices.dropna()
    if not finite.empty and not bool((finite.to_numpy() > 0.0).all()):
        raise ValidationError("close contains non-positive prices.")

    log_prices = pd.Series(np.log(prices.to_numpy(dtype="float64")), index=prices.index)
    log_ret = log_prices.diff()
    return _rolling_rms(log_ret**2, window, name="rv_cc")


def parkinson_rv(ohlc: pd.DataFrame, *, window: int = 1) -> pd.Series:
    r"""Parkinson (1980) high-low range realized volatility.

    The Parkinson estimator uses the daily high-low range and is ~5x more
    efficient than close-to-close under a zero-drift GBM:

    .. math::

        \mathrm{RV}^{park}_t = \sqrt{\frac{1}{4\ln 2}\,
                               \frac{1}{\text{window}}
                               \sum_{i=0}^{\text{window}-1}
                               \left(\ln \frac{H_{t-i}}{L_{t-i}}\right)^2}.

    Parameters
    ----------
    ohlc:
        A frame with ``high`` and ``low`` columns (case-insensitive) indexed by
        date.
    window:
        Number of trailing observations to aggregate (``>= 1``).

    Returns
    -------
    pandas.Series
        The Parkinson RV series, named ``"rv_parkinson"``.

    Raises
    ------
    ValidationError
        If required columns are missing, prices are non-positive, any
        ``high < low``, or ``window < 1``.
    """
    _validate_window(window)
    frame = _require_ohlc(ohlc, ("high", "low"), name="ohlc")

    log_hl = _log_ratio(frame["high"], frame["low"])
    daily_var = _PARKINSON_FACTOR * log_hl**2
    return _rolling_rms(daily_var, window, name="rv_parkinson")


def garman_klass_rv(ohlc: pd.DataFrame, *, window: int = 1) -> pd.Series:
    r"""Garman-Klass (1980) OHLC realized volatility.

    Uses the full open-high-low-close bar and is more efficient still than
    Parkinson:

    .. math::

        \mathrm{RV}^{gk}_t = \sqrt{\frac{1}{\text{window}}
                             \sum_{i=0}^{\text{window}-1}
                             \left[ \tfrac{1}{2}\!\left(\ln\tfrac{H}{L}\right)^2
                             - (2\ln 2 - 1)\!\left(\ln\tfrac{C}{O}\right)^2
                             \right]_{t-i}}.

    Parameters
    ----------
    ohlc:
        A frame with ``open``, ``high``, ``low``, ``close`` columns
        (case-insensitive) indexed by date.
    window:
        Number of trailing observations to aggregate (``>= 1``).

    Returns
    -------
    pandas.Series
        The Garman-Klass RV series, named ``"rv_garman_klass"``.

    Raises
    ------
    ValidationError
        If required columns are missing, prices are non-positive, any
        ``high < low``, or ``window < 1``.
    """
    _validate_window(window)
    frame = _require_ohlc(ohlc, OHLC_COLUMNS, name="ohlc")

    log_hl = _log_ratio(frame["high"], frame["low"])
    log_co = _log_ratio(frame["close"], frame["open"])
    daily_var = 0.5 * log_hl**2 - _GK_CO_FACTOR * log_co**2
    # The per-bar GK variance is non-negative in expectation but a single noisy
    # bar can dip slightly below zero; clip to zero so the sqrt is well-defined.
    daily_var = daily_var.clip(lower=0.0)
    return _rolling_rms(daily_var, window, name="rv_garman_klass")


def realized_volatility(
    ohlc: pd.DataFrame,
    *,
    estimator: str = "garman_klass",
    window: int = 1,
) -> pd.Series:
    """Dispatch to a named RV estimator.

    Parameters
    ----------
    ohlc:
        An OHLC frame (case-insensitive columns) indexed by date.
    estimator:
        One of ``"close_to_close"``, ``"parkinson"``, ``"garman_klass"``.
    window:
        Trailing aggregation window (``>= 1``).

    Returns
    -------
    pandas.Series
        The chosen RV series.

    Raises
    ------
    ValidationError
        If ``estimator`` is unknown or the inputs are malformed.
    """
    if estimator == "close_to_close":
        frame = _require_ohlc(ohlc, ("close",), name="ohlc")
        return close_to_close_rv(frame["close"], window=window)
    if estimator == "parkinson":
        return parkinson_rv(ohlc, window=window)
    if estimator == "garman_klass":
        return garman_klass_rv(ohlc, window=window)
    raise ValidationError(f"unknown estimator {estimator!r}; expected one of {list(_ESTIMATORS)}.")


def forward_rv_target(
    rv: pd.Series,
    *,
    horizon: int,
    gap: int = 1,
) -> pd.Series:
    r"""Build the strictly-forward realized-volatility target with an explicit gap.

    The target attached to timestamp ``t`` is the (annualization-free) realized
    volatility aggregated over the FUTURE window ``(t + gap, t + gap + horizon]``:

    .. math::

        y_t = \sqrt{\frac{1}{\text{horizon}}
              \sum_{k=1}^{\text{horizon}} \mathrm{RV}_{\,t + \text{gap} + k}^2 }.

    LEAKAGE GUARD: the window starts strictly after ``t + gap``, so for any model
    whose features are observable at ``t`` the feature index ``{<= t}`` and the
    target window ``{> t + gap}`` are DISJOINT. The trailing ``horizon + gap``
    rows have an incomplete future window and are returned as NaN (callers drop
    them). A unit test asserts this disjointness directly.

    Parameters
    ----------
    rv:
        A per-day realized-volatility series indexed by date.
    horizon:
        The forward window length in trading days (``>= 1``).
    gap:
        The number of days skipped between the feature timestamp ``t`` and the
        first day of the target window (``>= 0``; default ``1`` so the target
        never includes the same-day return).

    Returns
    -------
    pandas.Series
        The forward RV target aligned to ``rv.index``, named ``"rv_target"``,
        NaN where the future window is incomplete.

    Raises
    ------
    ValidationError
        If ``horizon < 1``, ``gap < 0``, or ``rv`` is empty.
    """
    if not isinstance(horizon, int) or isinstance(horizon, bool):
        raise ValidationError(f"horizon must be an int, got {type(horizon).__name__}.")
    if horizon < 1:
        raise ValidationError(f"horizon must be >= 1, got {horizon}.")
    if not isinstance(gap, int) or isinstance(gap, bool):
        raise ValidationError(f"gap must be an int, got {type(gap).__name__}.")
    if gap < 0:
        raise ValidationError(f"gap must be >= 0, got {gap}.")

    series = ensure_series(rv, name="rv", allow_nan=True)
    var = series.to_numpy(dtype="float64") ** 2

    n = var.size
    target = np.full(n, np.nan, dtype="float64")
    # The window for t is positions (t + gap, t + gap + horizon], i.e. the
    # indices [t + gap + 1, t + gap + horizon] in 0-based array terms.
    for t in range(n):
        first = t + gap + 1
        last = t + gap + horizon  # inclusive
        if last >= n:
            break  # all subsequent t have an even shorter (more incomplete) window
        window_var = var[first : last + 1]
        if np.isnan(window_var).any():
            continue
        target[t] = float(np.sqrt(window_var.mean()))

    return pd.Series(target, index=series.index, name="rv_target", dtype="float64")
