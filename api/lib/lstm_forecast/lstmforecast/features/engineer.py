"""Per-fold technical feature engineering on strictly-past bars.

THE DE-LEAK (part 1): every feature here is a function of bars STRICTLY EARLIER
than the bar it is attached to. Concretely:

- price changes use ``close.pct_change(fill_method=None)`` (no ffill-then-diff);
- every indicator is lagged one bar with ``.shift(1)`` so the feature row aligned
  to the target ``r_{t+1}`` is computed from information available at the close of
  day ``t`` and earlier — NEVER the same-bar close level (which would leak the
  label's scale);
- rolling / EWM windows have a finite warm-up that is dropped, and (in the
  walk-forward engine) are recomputed per fold so a warm-up never straddles a
  train/test boundary.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from lstmforecast._constants import EPS, TRADING_DAYS
from lstmforecast._exceptions import ValidationError
from lstmforecast._validation import ensure_monotonic_index, ensure_series

# quantcore-candidate: lagged technical features mirror factorlab feature kernels.


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Immutable specification of the technical-feature set to engineer.

    Attributes
    ----------
    momentum_windows:
        Look-back windows (in bars) for lagged past-return momentum features.
    vol_windows:
        Look-back windows for rolling realized-volatility features.
    rsi_window:
        Window for the lagged Relative Strength Index.
    include_lagged_return:
        Whether to include the one-bar-lagged return ``r_{t}`` itself as a
        feature for predicting ``r_{t+1}`` (still strictly past).
    """

    momentum_windows: tuple[int, ...] = (5, 10, 20)
    vol_windows: tuple[int, ...] = (10, 20)
    rsi_window: int = 14
    include_lagged_return: bool = True
    feature_names: tuple[str, ...] = field(default=(), compare=False)

    def __post_init__(self) -> None:
        """Validate that every window is a positive integer.

        Raises
        ------
        ValidationError
            If any window is non-positive or no feature group is selected.
        """
        windows = (*self.momentum_windows, *self.vol_windows, self.rsi_window)
        if any(int(w) < 1 for w in windows):
            raise ValidationError(f"FeatureSpec: all windows must be >= 1, got {windows!r}.")
        if not (self.momentum_windows or self.vol_windows or self.include_lagged_return):
            raise ValidationError(
                "FeatureSpec: at least one of momentum_windows, vol_windows, or "
                "include_lagged_return must be enabled (the RSI alone is too coarse)."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this spec."""
        return asdict(self)

    @property
    def max_window(self) -> int:
        """Return the largest look-back window across every feature group.

        The engineered frame loses ``max_window`` warm-up rows (plus the one row
        consumed by the universal ``.shift(1)`` lag), so callers size their panels
        against this value.
        """
        return max(
            (*self.momentum_windows, *self.vol_windows, self.rsi_window),
            default=1,
        )


def engineer_features(prices: pd.Series, spec: FeatureSpec | None = None) -> pd.DataFrame:
    """Build a no-lookahead technical-feature matrix from a price series.

    Every returned column is lagged so that the feature row indexed at date ``t``
    uses only bars at or before ``t`` and is therefore safe to pair with the
    target ``r_{t+1}``. Rows with any NaN from indicator warm-up are dropped, so
    the returned frame is dense and ready to scale.

    DE-LEAK GUARANTEE: there is NO same-bar close-level feature; price-derived
    features are differences/ratios (returns, momentum, volatility, RSI), never
    the raw level, so the feature scale cannot encode the label's scale.

    Parameters
    ----------
    prices:
        A strictly-positive, time-indexed price series (sorted ascending).
    spec:
        The feature specification; defaults to :class:`FeatureSpec` defaults.

    Returns
    -------
    pandas.DataFrame
        A dense, float64 feature matrix indexed by date, columns named after the
        engineered features.

    Raises
    ------
    ValidationError
        If ``prices`` is malformed or too short for the requested windows.
    """
    spec = spec or FeatureSpec()

    series = ensure_series(prices, name="prices")
    series = ensure_monotonic_index(series, name="prices")
    if bool((series <= 0.0).any()):
        raise ValidationError("engineer_features: prices must be strictly positive.")

    # Past one-bar log-return, observable at the close of each bar. ``pct_change`` is
    # taken with ``fill_method=None`` so gaps stay NaN (never forward-filled, which
    # would manufacture spurious zero returns and leak across missing bars).
    pct = series.pct_change(fill_method=None)
    log_ret = pd.Series(np.log1p(pct.to_numpy()), index=series.index)

    columns: dict[str, pd.Series] = {}

    # Momentum: cumulative past return over each window (a difference of log-prices),
    # never the close *level* — so the feature scale cannot encode the label scale.
    for window in spec.momentum_windows:
        columns[f"mom_{window}"] = log_ret.rolling(window, min_periods=window).sum()

    # Realized volatility: rolling std of past returns over each window.
    for window in spec.vol_windows:
        columns[f"vol_{window}"] = log_ret.rolling(window, min_periods=window).std(
            ddof=0
        ) * np.sqrt(float(TRADING_DAYS))

    # Relative Strength Index on past returns (bounded [0, 100]; scale-free).
    columns[f"rsi_{spec.rsi_window}"] = _rsi(log_ret, spec.rsi_window)

    if spec.include_lagged_return:
        # The one-bar return r_t itself (still strictly past relative to r_{t+1}).
        columns["ret_lag1"] = log_ret

    frame = pd.DataFrame(columns, index=series.index)

    # THE .shift(1) DISCIPLINE: lag every column by one bar so the feature row indexed
    # at date t is a function of bars STRICTLY EARLIER than t. This makes the row at t
    # independent of P_t (and everything after), so perturbing bar t or any later bar
    # never changes feature row t — the future-perturbation invariance the property
    # tests pin, and the headline fix for accidental same-bar leakage.
    frame = frame.shift(1)

    # Drop indicator warm-up rows so the returned matrix is dense and ready to scale.
    frame = frame.dropna(axis="index", how="any")

    ordered = (
        [f"mom_{w}" for w in spec.momentum_windows]
        + [f"vol_{w}" for w in spec.vol_windows]
        + [f"rsi_{spec.rsi_window}"]
        + (["ret_lag1"] if spec.include_lagged_return else [])
    )
    frame = frame[ordered].astype("float64")

    if frame.empty:
        raise ValidationError(
            f"engineer_features: price series of length {len(series)} is too short for "
            f"windows {spec.to_dict()!r}; no dense feature row survives warm-up."
        )
    return frame


def _rsi(log_ret: pd.Series, window: int) -> pd.Series:
    r"""Compute Wilder's Relative Strength Index from a past log-return series.

    Gains and losses are the positive and negative parts of each past return; the
    RSI is ``100 - 100 / (1 + avg_gain / avg_loss)`` over a trailing ``window``.
    The result is bounded in ``[0, 100]`` and depends only on past returns, so it
    carries momentum information without encoding the price level. A flat window
    (zero average loss) maps to ``100`` via the epsilon-floored denominator.

    Parameters
    ----------
    log_ret:
        The one-bar past log-return series (leading NaN allowed; propagated).
    window:
        The RSI averaging window in bars (``>= 1``).

    Returns
    -------
    pandas.Series
        The RSI series aligned to ``log_ret`` (warm-up rows are NaN).
    """
    gains = log_ret.clip(lower=0.0)
    losses = (-log_ret).clip(lower=0.0)
    avg_gain = gains.rolling(window, min_periods=window).mean()
    avg_loss = losses.rolling(window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + EPS)
    return 100.0 - 100.0 / (1.0 + rs)
