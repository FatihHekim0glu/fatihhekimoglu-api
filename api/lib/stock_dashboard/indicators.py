"""Technical indicators — hand-rolled in pandas/numpy.

All implementations are verified against TA-Lib in tests/test_indicators.py.
Five non-obvious rules to bake into every implementation:

  1. RSI uses Wilder's RMA smoothing (alpha = 1/n, recursive),
     NOT the common-but-wrong ewm(span=n) (which is alpha = 2/(n+1)).
  2. EMA uses adjust=False and is seeded with an SMA over the first n bars
     so the early values agree with TA-Lib (default pandas seeding diverges
     for ~50–100 bars before converging).
  3. Bollinger Bands use population std (ddof=0) to match TA-Lib and
     Bollinger's own definition — pandas' .std() defaults to ddof=1 (sample),
     which makes the bands ~3 % narrower at the 20-period default.
  4. All indicators leave warmup periods as NaN. Do not extrapolate, do not
     fill with zero — the charts rely on NaN to skip the warmup region.
  5. MACD returns three series — be explicit about (macd_line, signal,
     histogram) ordering; different libraries disagree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, length: int = 20) -> pd.Series:
    """Simple moving average.

    Rolling mean with min_periods=length so the warmup region (first
    length-1 bars) is NaN rather than the partial mean of fewer values.
    """
    if length < 1:
        raise ValueError(f"length must be >= 1, got {length}")
    return close.rolling(window=length, min_periods=length).mean()


def ema(close: pd.Series, length: int = 12) -> pd.Series:
    """Exponential moving average — SMA-seeded, adjust=False.

    Pre-warmup region (first `length-1` bars) is NaN; position `length-1` is
    primed with the SMA of the first `length` closes; from there on the standard
    `EMA_t = alpha * close_t + (1 - alpha) * EMA_{t-1}` recursion runs with
    `alpha = 2/(length+1)`. Matches TA-Lib's seeding convention.
    """
    if length < 1:
        raise ValueError(f"length must be >= 1, got {length}")
    if len(close) < length:
        return pd.Series(np.nan, index=close.index, dtype=float)
    seed = close.rolling(window=length, min_periods=length).mean()
    seeded = close.astype(float).copy()
    seeded.iloc[: length - 1] = np.nan
    seeded.iloc[length - 1] = seed.iloc[length - 1]
    result = seeded.ewm(span=length, adjust=False, ignore_na=False).mean()
    result.iloc[: length - 1] = np.nan
    return result


def bollinger_bands(close: pd.Series, length: int = 20, k: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands — middle = SMA(length), upper/lower = middle ± k * std.

    Returns a DataFrame with columns ['upper', 'middle', 'lower']. Uses
    population std (ddof=0) to match TA-Lib and Bollinger's original
    definition; pandas default ddof=1 (sample) would produce bands ~3%
    narrower at the 20-period default.
    """
    if length < 1:
        raise ValueError(f"length must be >= 1, got {length}")
    middle = sma(close, length)
    std = close.rolling(window=length, min_periods=length).std(ddof=0)
    upper = middle + k * std
    lower = middle - k * std
    return pd.DataFrame(
        {"upper": upper, "middle": middle, "lower": lower},
        index=close.index,
    )[["upper", "middle", "lower"]]


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index — Wilder's smoothing.

    Separates up- and down-moves, applies Wilder's RMA to each
    (`ewm(alpha=1/length, adjust=False)` primed with the SMA of the first
    `length` gains/losses). RS = avg_gain / avg_loss; RSI = 100 - 100/(1 + RS).
    Do NOT use `ewm(span=length)` — that is exponential smoothing with the
    wrong decay constant and disagrees with TA-Lib / TradingView by 1–3 points.
    """
    if length < 1:
        raise ValueError(f"length must be >= 1, got {length}")
    if len(close) <= length:
        return pd.Series(np.nan, index=close.index, dtype=float)
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)

    # Wilder's seed: SMA of the first `length` gains/losses lives at index `length`
    # (because delta is NaN at index 0, the first `length` real diffs span 1..length).
    def _wilder_rma(series: pd.Series) -> pd.Series:
        # Position of the seed: first `length` valid diffs are at positions 1..length,
        # so the SMA seed sits at index position `length`.
        seed_value = series.iloc[1 : length + 1].mean()
        seeded = series.astype(float).copy()
        seeded.iloc[:length] = np.nan
        seeded.iloc[length] = seed_value
        # Wilder's RMA: alpha = 1/length (NOT 2/(length+1)).
        rma = seeded.ewm(alpha=1.0 / length, adjust=False, ignore_na=False).mean()
        rma.iloc[:length] = np.nan
        return rma

    avg_gain = _wilder_rma(gains)
    avg_loss = _wilder_rma(losses)

    # RS = avg_gain / avg_loss; constant series -> 0/0 -> NaN (per spec).
    rs = avg_gain / avg_loss
    rsi_values = 100.0 - 100.0 / (1.0 + rs)
    # When avg_loss == 0 and avg_gain > 0, rs is +inf and rsi -> 100 (limit case).
    rsi_values = rsi_values.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    # When both are zero (constant series), result is NaN.
    rsi_values = rsi_values.where(~((avg_loss == 0) & (avg_gain == 0)), np.nan)
    return rsi_values


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line, and histogram.

    Returns a DataFrame with columns ['macd', 'signal', 'histogram'] in that
    order. `macd = ema(close, fast) - ema(close, slow)`,
    `signal = ema(macd, signal_period)`, `histogram = macd - signal`. Reuses
    this module's own `ema` so the SMA-seeding convention is consistent.
    """
    macd_line = ema(close, fast) - ema(close, slow)
    # ema() requires `length` non-NaN priming values; dropna before seeding then
    # reindex so the signal line's own warmup is honored on top of macd's warmup.
    macd_valid = macd_line.dropna()
    signal_on_valid = ema(macd_valid, signal)
    signal_line = signal_on_valid.reindex(close.index)
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": histogram},
        index=close.index,
    )[["macd", "signal", "histogram"]]
