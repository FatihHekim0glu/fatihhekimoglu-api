"""Verify hand-rolled indicators against TA-Lib and closed-form properties.

Two layers:
  1. Closed-form (always runs): degenerate inputs with hand-computable answers
     plus edge-case parametrizations covering all-NaN, short input, and NaN gaps.
  2. Oracle (talib required): assert_allclose vs TA-Lib on the tail of a
     synthetic OHLCV fixture. Gated via pytest.importorskip - auto-skipped
     when TA-Lib is not installed on the host.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src import indicators

# ---------------------------------------------------------------------------
# Closed-form correctness - SMA
# ---------------------------------------------------------------------------


class TestSMA:
    def test_constant_input(self, constant_series):
        result = indicators.sma(constant_series, length=20)
        assert result.iloc[19:].eq(100.0).all()
        assert result.iloc[:19].isna().all()

    def test_warmup_first_length_minus_one_are_nan(self, monotone_up_series):
        length = 14
        result = indicators.sma(monotone_up_series, length=length)
        assert result.iloc[: length - 1].isna().all()
        assert result.iloc[length - 1 :].notna().all()


# ---------------------------------------------------------------------------
# Closed-form correctness - EMA
# ---------------------------------------------------------------------------


class TestEMA:
    def test_constant_input_after_seeding(self, constant_series):
        length = 12
        result = indicators.ema(constant_series, length=length)
        assert result.iloc[: length - 1].isna().all()
        assert result.iloc[length - 1 :].eq(100.0).all()


# ---------------------------------------------------------------------------
# Closed-form correctness - Bollinger Bands
# ---------------------------------------------------------------------------


class TestBollinger:
    def test_columns_are_upper_middle_lower(self, monotone_up_series):
        bands = indicators.bollinger_bands(monotone_up_series, length=20, k=2.0)
        assert list(bands.columns) == ["upper", "middle", "lower"]

    def test_middle_equals_sma(self, monotone_up_series):
        length = 20
        bands = indicators.bollinger_bands(monotone_up_series, length=length, k=2.0)
        expected_middle = indicators.sma(monotone_up_series, length=length)
        pd.testing.assert_series_equal(bands["middle"], expected_middle, check_names=False)

    def test_upper_minus_middle_is_k_times_population_std(self, monotone_up_series):
        length = 20
        k = 2.0
        bands = indicators.bollinger_bands(monotone_up_series, length=length, k=k)
        expected_offset = k * monotone_up_series.rolling(window=length, min_periods=length).std(
            ddof=0
        )
        actual_offset = bands["upper"] - bands["middle"]
        pd.testing.assert_series_equal(actual_offset, expected_offset, check_names=False)

    def test_constant_input_zero_width(self, constant_series):
        length = 20
        bands = indicators.bollinger_bands(constant_series, length=length, k=2.0)
        width = (bands["upper"] - bands["lower"]).iloc[length - 1 :]
        assert width.eq(0.0).all()


# ---------------------------------------------------------------------------
# Closed-form correctness - RSI
# ---------------------------------------------------------------------------


class TestRSI:
    def test_monotone_up_approaches_100(self, monotone_up_series):
        result = indicators.rsi(monotone_up_series, length=14)
        assert result.dropna().iloc[-1] == pytest.approx(100.0, abs=1e-6)

    def test_monotone_down_approaches_0(self, monotone_down_series):
        result = indicators.rsi(monotone_down_series, length=14)
        assert result.dropna().iloc[-1] == pytest.approx(0.0, abs=1e-6)

    def test_constant_input_is_nan(self, constant_series):
        result = indicators.rsi(constant_series, length=14)
        # No movement => 0/0 in Wilder's RS => NaN per the implementation contract.
        assert result.iloc[14:].isna().all()

    def test_wilder_textbook_golden_value(self):
        # 16 closes taken from Wilder's "New Concepts in Technical Trading
        # Systems" (1978). The textbook prints the first RSI value at the end
        # of the 14-bar warmup - that lands at iloc[14] in 0-indexed terms
        # (the seed position with SMA seeding, matching TA-Lib). The point of
        # this assertion is to catch the most common silent bug in RSI: using
        # ewm(span=14) (alpha = 2/15) instead of Wilder's RMA (alpha = 1/14).
        # The two differ by several points on this series.
        closes = pd.Series(
            [
                44.34,
                44.09,
                44.15,
                43.61,
                44.33,
                44.83,
                45.10,
                45.42,
                45.84,
                46.08,
                45.89,
                46.03,
                45.61,
                46.28,
                46.28,
                46.00,
            ]
        )
        result = indicators.rsi(closes, length=14)
        # Textbook rounds to 2 dp; rtol=1e-2 keeps us inside that tolerance.
        assert result.iloc[14] == pytest.approx(70.46, rel=1e-2)


# ---------------------------------------------------------------------------
# Closed-form correctness - MACD
# ---------------------------------------------------------------------------


class TestMACD:
    def test_columns_are_macd_signal_histogram(self, monotone_up_series):
        result = indicators.macd(monotone_up_series, fast=12, slow=26, signal=9)
        assert list(result.columns) == ["macd", "signal", "histogram"]

    def test_histogram_equals_macd_minus_signal(self, monotone_up_series):
        result = indicators.macd(monotone_up_series, fast=12, slow=26, signal=9)
        expected_hist = result["macd"] - result["signal"]
        pd.testing.assert_series_equal(result["histogram"], expected_hist, check_names=False)

    def test_signal_matches_module_ema_of_macd(self, synthetic_ohlcv):
        close = synthetic_ohlcv["close"]
        result = indicators.macd(close, fast=12, slow=26, signal=9)
        macd_valid = result["macd"].dropna()
        expected_signal = indicators.ema(macd_valid, length=9).reindex(close.index)
        pd.testing.assert_series_equal(result["signal"], expected_signal, check_names=False)


# ---------------------------------------------------------------------------
# Edge cases - parametrized across every indicator
# ---------------------------------------------------------------------------


def _series_all_nan(n: int = 60) -> pd.Series:
    return pd.Series([np.nan] * n)


def _series_short(n: int = 5) -> pd.Series:
    return pd.Series(np.arange(n, dtype=float))


def _series_with_nan_gap(n: int = 80, gap_start: int = 30, gap_len: int = 5) -> pd.Series:
    values = np.arange(n, dtype=float)
    values[gap_start : gap_start + gap_len] = np.nan
    return pd.Series(values)


# Each entry is (id, callable that takes a close Series and returns a Series
# or DataFrame, length used). We extract a representative numeric Series from
# the result so the same assertions work for SMA/EMA/RSI (Series) and for
# Bollinger/MACD (DataFrame).
def _to_series(result) -> pd.Series:
    if isinstance(result, pd.DataFrame):
        # Pick the first column as a witness - for Bollinger that is 'upper',
        # for MACD that is 'macd'. Both behave like the rest under NaN inputs.
        return result.iloc[:, 0]
    return result


INDICATOR_CASES = [
    ("sma", lambda s: indicators.sma(s, length=20), 20),
    ("ema", lambda s: indicators.ema(s, length=12), 12),
    ("bollinger", lambda s: indicators.bollinger_bands(s, length=20, k=2.0), 20),
    ("rsi", lambda s: indicators.rsi(s, length=14), 14),
    ("macd", lambda s: indicators.macd(s, fast=12, slow=26, signal=9), 26),
]


@pytest.mark.parametrize("name,fn,length", INDICATOR_CASES)
def test_all_nan_input_returns_all_nan(name, fn, length):
    series = _series_all_nan(n=max(60, length * 3))
    result = _to_series(fn(series))
    assert len(result) == len(series)
    assert result.isna().all()


@pytest.mark.parametrize("name,fn,length", INDICATOR_CASES)
def test_input_shorter_than_period_is_all_nan(name, fn, length):
    # Length strictly less than the longest internal window - for MACD the
    # binding constraint is `slow` (26), not `fast`.
    series = _series_short(n=max(3, length - 1))
    result = _to_series(fn(series))
    assert len(result) == len(series)
    assert result.isna().all()


@pytest.mark.parametrize("name,fn,length", INDICATOR_CASES)
def test_nan_gap_does_not_permanently_corrupt_tail(name, fn, length):
    n = max(120, length * 5)
    gap_start = length * 2
    gap_len = 5
    series = _series_with_nan_gap(n=n, gap_start=gap_start, gap_len=gap_len)
    result = _to_series(fn(series))
    assert len(result) == len(series)
    # Far enough past the gap, at least one bar must be finite again. Without
    # this, a silently-poisoning recursion (e.g. EMA without ignore_na) would
    # leave every subsequent value NaN.
    tail = result.iloc[gap_start + gap_len + length * 2 :]
    assert tail.notna().any(), f"{name}: indicator stayed NaN forever after a {gap_len}-bar NaN gap"


# ---------------------------------------------------------------------------
# Oracle layer - compare against TA-Lib on the synthetic fixture
# ---------------------------------------------------------------------------

try:
    import talib
except ImportError:
    talib = None  # type: ignore[assignment]


@pytest.mark.oracle
@pytest.mark.skipif(talib is None, reason="ta-lib not installed; skip oracle parity tests")
class TestOracle:
    """Reference-implementation parity tests.

    TA-Lib is the de facto reference for these five indicators; matching it to
    1e-6 rules out subtle off-by-one, ddof, and seeding bugs that the
    closed-form tests above cannot reach (they only exercise degenerate
    inputs). We skip the warmup region on each indicator because TA-Lib emits
    a slightly different sentinel there (0.0 vs NaN, depending on version).
    """

    TAIL = 400  # plenty of bars past warmup for every default-length indicator

    def test_sma_matches_talib(self, synthetic_ohlcv):
        close = synthetic_ohlcv["close"].astype(float)
        length = 20
        ours = indicators.sma(close, length=length).iloc[length:].to_numpy()
        ref = talib.SMA(close.to_numpy(), timeperiod=length)[length:]
        ours_tail = ours[-self.TAIL :]
        ref_tail = ref[-self.TAIL :]
        np.testing.assert_allclose(ours_tail, ref_tail, rtol=1e-6, equal_nan=True)

    def test_ema_matches_talib(self, synthetic_ohlcv):
        close = synthetic_ohlcv["close"].astype(float)
        length = 12
        ours = indicators.ema(close, length=length).iloc[length:].to_numpy()
        ref = talib.EMA(close.to_numpy(), timeperiod=length)[length:]
        np.testing.assert_allclose(ours[-self.TAIL :], ref[-self.TAIL :], rtol=1e-6, equal_nan=True)

    def test_bollinger_matches_talib(self, synthetic_ohlcv):
        close = synthetic_ohlcv["close"].astype(float)
        length = 20
        bands = indicators.bollinger_bands(close, length=length, k=2.0)
        ref_upper, ref_middle, ref_lower = talib.BBANDS(
            close.to_numpy(),
            timeperiod=length,
            nbdevup=2.0,
            nbdevdn=2.0,
            matype=0,
        )
        for ours, ref in (
            (bands["upper"].to_numpy(), ref_upper),
            (bands["middle"].to_numpy(), ref_middle),
            (bands["lower"].to_numpy(), ref_lower),
        ):
            np.testing.assert_allclose(
                ours[length:][-self.TAIL :],
                ref[length:][-self.TAIL :],
                rtol=1e-6,
                equal_nan=True,
            )

    def test_rsi_matches_talib(self, synthetic_ohlcv):
        close = synthetic_ohlcv["close"].astype(float)
        length = 14
        ours = indicators.rsi(close, length=length).to_numpy()
        ref = talib.RSI(close.to_numpy(), timeperiod=length)
        # RSI's recursion needs a few extra bars beyond `length` to stabilize
        # to within 1e-6 of TA-Lib; skip the initial 2*length region.
        skip = length * 2
        np.testing.assert_allclose(
            ours[skip:][-self.TAIL :],
            ref[skip:][-self.TAIL :],
            rtol=1e-6,
            equal_nan=True,
        )

    def test_macd_matches_talib(self, synthetic_ohlcv):
        close = synthetic_ohlcv["close"].astype(float)
        fast, slow, signal = 12, 26, 9
        result = indicators.macd(close, fast=fast, slow=slow, signal=signal)
        ref_macd, ref_signal, ref_hist = talib.MACD(
            close.to_numpy(),
            fastperiod=fast,
            slowperiod=slow,
            signalperiod=signal,
        )
        # Skip well past slow + signal so both EMAs and the signal-EMA have
        # converged on both sides.
        skip = slow + signal
        for ours, ref in (
            (result["macd"].to_numpy(), ref_macd),
            (result["signal"].to_numpy(), ref_signal),
            (result["histogram"].to_numpy(), ref_hist),
        ):
            # rtol=1e-5 (not 1e-6 like the other oracle tests) because MACD
            # chains two EMAs then a third signal-line EMA, and accumulated
            # floating-point error is ~2x the single-pass EMA case.
            np.testing.assert_allclose(
                ours[skip:][-self.TAIL :],
                ref[skip:][-self.TAIL :],
                rtol=1e-5,
                equal_nan=True,
            )
