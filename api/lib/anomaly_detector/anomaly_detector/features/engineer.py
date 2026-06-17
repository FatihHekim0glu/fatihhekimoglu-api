"""Causal per-day feature engineering for anomaly detection.

Turns a price (and optional volume) series into a per-day feature matrix that a
detector can score. Every feature is CAUSAL: the value at day ``t`` is built
only from information available strictly before ``t``, via ``.shift(1)``-safe
rolling windows. Returns are differenced with ``pct_change(fill_method=None)``
(never forward-filling prices, which would manufacture spurious zero returns and
leak information across gaps).

Feature set (per day):

- ``log_return``        — log of the gross return ``ln(p_t / p_{t-1})``.
- ``realized_vol``      — rolling realized volatility of log-returns.
- ``return_zscore``     — return standardized by its rolling mean/std.
- ``range_atr``         — rolling average true range proxy (or |return| ATR).
- ``volume_zscore``     — volume standardized by its rolling mean/std (0 if no
  volume supplied).
- ``return_autocorr``   — short-lag rolling autocorrelation of returns.

NO-LOOKAHEAD REQUIREMENT: the rolling statistics that *standardize* day ``t``
must exclude day ``t`` itself (a ``.shift(1)`` on the rolling mean/std), so the
z-scores cannot peek at the very return they normalize. This is the upstream
half of the ``.shift(1)`` flag chokepoint. The whole assembled feature frame is
then shifted by one row so that EVERY feature at day ``t`` is a function of data
strictly before ``t`` (mutating any bar at or after ``t`` cannot change the
row ``t`` feature vector).

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from anomaly_detector._constants import EPS
from anomaly_detector._validation import ensure_series

if TYPE_CHECKING:
    from anomaly_detector._typing import PricesLike

#: The ordered feature-column names produced by :func:`engineer_features`.
FEATURE_NAMES: tuple[str, ...] = (
    "log_return",
    "realized_vol",
    "return_zscore",
    "range_atr",
    "volume_zscore",
    "return_autocorr",
)

#: Default rolling window (in trading days) for the realized-vol / z-score /
#: autocorrelation statistics.
DEFAULT_WINDOW: int = 21

#: Short lag (in trading days) for the rolling return-autocorrelation feature.
_AUTOCORR_LAG: int = 1


def _validate_window(window: int) -> None:
    """Raise :class:`ValidationError` unless ``window >= 2``."""
    # Imported lazily-by-reference: ValidationError lives in _exceptions, but we
    # re-raise the same type the validation helpers use so callers catch one base.
    from anomaly_detector._exceptions import ValidationError

    if not isinstance(window, int) or isinstance(window, bool):
        raise ValidationError(f"window must be an int, got {type(window).__name__}.")
    if window < 2:
        raise ValidationError(f"window must be >= 2, got {window}.")


def _coerce_prices(prices: PricesLike) -> pd.Series:
    """Coerce a price-like input to a 1-D float64 price Series.

    A single-column DataFrame is squeezed to its column; an ndarray/sequence is
    wrapped. Delegated NaN/shape checks come from :func:`ensure_series`.
    """
    if isinstance(prices, pd.DataFrame):
        from anomaly_detector._exceptions import ValidationError

        if prices.shape[1] != 1:
            raise ValidationError(
                f"prices DataFrame must have exactly one column, got {prices.shape[1]}."
            )
        prices = prices.iloc[:, 0]
    return ensure_series(prices, name="prices")


def _log_returns(prices: pd.Series) -> pd.Series:
    """Log-returns ``ln(p_t / p_{t-1})`` via ``pct_change(fill_method=None)``.

    The price path is differenced without forward-filling (no manufactured zero
    returns across gaps); the leading NaN row stays in place so the result keeps
    the price index. Scaling the whole price path by a positive constant leaves
    log-returns exactly unchanged (the scale cancels in the ratio).
    """
    gross = prices.pct_change(fill_method=None) + 1.0
    return pd.Series(np.log(gross), index=prices.index, name="log_return")


def causal_return_zscore(returns: pd.Series, *, window: int = DEFAULT_WINDOW) -> pd.Series:
    """Standardize ``returns`` by a strictly-trailing rolling mean and std.

    The mean and std used to standardize day ``t`` are computed over the window
    ENDING AT ``t - 1`` (a ``.shift(1)`` on the rolling statistics), so the
    z-score never sees the return it normalizes. Exposed separately because it
    is reused by the transparent proxy label (``|z-return| > 3``) in the
    evaluation layer.

    Parameters
    ----------
    returns:
        A per-day return series indexed by date.
    window:
        Rolling window length (must be ``>= 2``).

    Returns
    -------
    pandas.Series
        The causal return z-score, indexed like ``returns`` (leading warm-up
        rows are NaN).

    Raises
    ------
    ValidationError
        If ``returns`` is malformed or ``window < 2``.
    """
    _validate_window(window)
    series = ensure_series(returns, name="returns", allow_nan=True)

    roll = series.rolling(window=window, min_periods=window)
    mean_prev = roll.mean().shift(1)
    std_prev = roll.std(ddof=1).shift(1)

    zscore = (series - mean_prev) / std_prev.where(std_prev > EPS)
    return pd.Series(zscore, index=series.index, name="return_zscore")


def _rolling_autocorr(returns: pd.Series, *, window: int, lag: int) -> pd.Series:
    """Trailing rolling lag-``lag`` autocorrelation of ``returns``.

    The correlation at day ``t`` is computed over the trailing ``window`` ending
    at ``t``; the assembled frame is later ``.shift(1)``-ed so the published row
    ``t`` only sees data strictly before ``t``. Degenerate (zero-variance)
    windows yield ``0.0`` rather than NaN, so a flat calm stretch is "no
    autocorrelation signal" instead of a dropped row.
    """
    lagged = returns.shift(lag)

    def _corr(idx_window: np.ndarray) -> float:
        # ``idx_window`` holds the *positions* of the current window (raw=True on
        # an arange carrier), letting us pair returns with their lagged values.
        positions = idx_window.astype(int)
        a = returns.to_numpy()[positions]
        b = lagged.to_numpy()[positions]
        mask = np.isfinite(a) & np.isfinite(b)
        if int(mask.sum()) < 3:
            return 0.0
        a_v = a[mask]
        b_v = b[mask]
        if np.std(a_v) <= EPS or np.std(b_v) <= EPS:
            return 0.0
        return float(np.corrcoef(a_v, b_v)[0, 1])

    carrier = pd.Series(np.arange(len(returns), dtype="float64"), index=returns.index, name="_pos")
    out = carrier.rolling(window=window, min_periods=window).apply(_corr, raw=True)
    return pd.Series(out, index=returns.index, name="return_autocorr")


def engineer_features(
    prices: PricesLike,
    *,
    volume: pd.Series | None = None,
    window: int = DEFAULT_WINDOW,
) -> pd.DataFrame:
    """Build the causal per-day feature matrix from a price (and volume) series.

    Each column is one of :data:`FEATURE_NAMES`, computed from
    ``.shift(1)``-safe rolling windows so that day ``t``'s features use only
    data strictly before ``t``. Rows that cannot be formed without lookahead
    (the leading ``window`` rows whose statistics would otherwise peek) are
    dropped, so the returned frame is shorter than ``prices`` by the warm-up.

    Parameters
    ----------
    prices:
        A price level series (or single-column price panel), indexed by date.
    volume:
        Optional volume series aligned to ``prices``; when ``None`` the
        ``volume_zscore`` column is filled with zeros.
    window:
        Rolling window length (trading days) for the realized-vol, z-score, and
        autocorrelation statistics. Must be ``>= 2``.

    Returns
    -------
    pandas.DataFrame
        A per-day feature matrix (rows = date, columns = :data:`FEATURE_NAMES`),
        free of NaN and with no lookahead.

    Raises
    ------
    ValidationError
        If ``prices`` is malformed or ``window < 2``.
    """
    _validate_window(window)
    price_s = _coerce_prices(prices)

    log_ret = _log_returns(price_s)

    # Trailing rolling stats on log-returns. min_periods == window so a window is
    # only emitted once it is fully populated; the frame-wide ``.shift(1)`` below
    # then enforces the strict ``<= t-1`` causality on top.
    roll = log_ret.rolling(window=window, min_periods=window)
    realized_vol = roll.std(ddof=1)

    # Causal z-score: standardize by stats ending at t-1 (its own internal shift).
    return_z = causal_return_zscore(log_ret, window=window)

    # Range/ATR proxy: rolling mean of absolute log-returns (a true range proxy
    # when only closes are available). Scale-invariant for the same reason
    # log-returns are.
    range_atr = log_ret.abs().rolling(window=window, min_periods=window).mean()

    # Volume z-score (causal): standardize by trailing stats ending at t-1.
    if volume is None:
        volume_z = pd.Series(0.0, index=price_s.index, name="volume_zscore")
    else:
        vol_s = ensure_series(volume, name="volume", allow_nan=True).reindex(price_s.index)
        v_roll = vol_s.rolling(window=window, min_periods=window)
        v_mean_prev = v_roll.mean().shift(1)
        v_std_prev = v_roll.std(ddof=1).shift(1)
        volume_z = (vol_s - v_mean_prev) / v_std_prev.where(v_std_prev > EPS)
        volume_z = pd.Series(volume_z, index=price_s.index, name="volume_zscore")

    return_ac = _rolling_autocorr(log_ret, window=window, lag=_AUTOCORR_LAG)

    raw = pd.concat(
        {
            "log_return": log_ret,
            "realized_vol": realized_vol,
            "return_zscore": return_z,
            "range_atr": range_atr,
            "volume_zscore": volume_z,
            "return_autocorr": return_ac,
        },
        axis=1,
    )[list(FEATURE_NAMES)]

    # FRAME-WIDE CAUSAL CHOKEPOINT: every feature at row t becomes the value
    # engineered from data strictly before t. Mutating any bar at or after t can
    # no longer change the row-t feature vector (future-perturbation invariance).
    raw = raw.shift(1)

    # volume_zscore is intentionally 0.0 (not NaN) when no volume is supplied, so
    # do not let it gate the warm-up dropna; the other features already encode
    # the warm-up via their NaNs.
    feature_frame = raw.dropna(how="any")
    feature_frame = feature_frame.astype("float64")
    feature_frame.columns = pd.Index(FEATURE_NAMES)
    return feature_frame
