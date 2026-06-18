"""Smoke tests for src.charts.build_figure.

These don't assert pixel-level rendering (that needs a browser); they verify
the figure object's structure: row count, trace count, layout keys, range
selector placement, and graceful handling of missing indicator columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from src import charts


@pytest.fixture
def ohlcv_with_indicators() -> pd.DataFrame:
    """A 60-bar OHLCV frame pre-populated with every indicator column."""
    rng = np.random.default_rng(7)
    n = 60
    idx = pd.bdate_range(start="2024-01-02", periods=n)
    close = pd.Series(100 + np.cumsum(rng.standard_normal(n)), index=idx)
    df = pd.DataFrame(
        {
            "Open": close,
            "High": close + rng.uniform(0.1, 1.0, n),
            "Low": close - rng.uniform(0.1, 1.0, n),
            "Close": close,
            "Volume": rng.integers(1_000_000, 10_000_000, n),
            "sma_20": close.rolling(20).mean(),
            "sma_50": close.rolling(50).mean(),
            "ema_12": close.ewm(span=12, adjust=False).mean(),
            "ema_26": close.ewm(span=26, adjust=False).mean(),
            "bb_upper": close.rolling(20).mean() + 2 * close.rolling(20).std(ddof=0),
            "bb_middle": close.rolling(20).mean(),
            "bb_lower": close.rolling(20).mean() - 2 * close.rolling(20).std(ddof=0),
            "rsi": pd.Series(50.0 + 20 * rng.standard_normal(n), index=idx),
            "macd": pd.Series(rng.standard_normal(n), index=idx),
            "macd_signal": pd.Series(rng.standard_normal(n), index=idx),
            "macd_hist": pd.Series(rng.standard_normal(n), index=idx),
        },
        index=idx,
    )
    return df


class TestSubplotRowCount:
    """Conditional row layout - 2 to 4 rows depending on requested indicators."""

    def test_no_indicators_yields_two_rows(self, ohlcv_with_indicators):
        fig = charts.build_figure(ohlcv_with_indicators, indicators=())
        # Two rows of subplots → two distinct x-axes (xaxis + xaxis2).
        assert "xaxis2" in fig.layout
        assert "xaxis3" not in fig.layout

    def test_rsi_only_yields_three_rows(self, ohlcv_with_indicators):
        fig = charts.build_figure(ohlcv_with_indicators, indicators=("rsi",))
        assert "xaxis3" in fig.layout
        assert "xaxis4" not in fig.layout

    def test_macd_only_yields_three_rows(self, ohlcv_with_indicators):
        fig = charts.build_figure(ohlcv_with_indicators, indicators=("macd",))
        assert "xaxis3" in fig.layout
        assert "xaxis4" not in fig.layout

    def test_rsi_and_macd_yields_four_rows(self, ohlcv_with_indicators):
        fig = charts.build_figure(ohlcv_with_indicators, indicators=("rsi", "macd"))
        assert "xaxis4" in fig.layout


class TestTraces:
    def test_candlestick_default_renders_candle_trace(self, ohlcv_with_indicators):
        fig = charts.build_figure(ohlcv_with_indicators, indicators=())
        types = [type(t).__name__ for t in fig.data]
        assert "Candlestick" in types

    def test_line_view_renders_scatter_close(self, ohlcv_with_indicators):
        fig = charts.build_figure(ohlcv_with_indicators, indicators=(), candlestick=False)
        types = [type(t).__name__ for t in fig.data]
        assert "Scatter" in types and "Candlestick" not in types

    def test_macd_renders_all_three_traces(self, ohlcv_with_indicators):
        fig = charts.build_figure(ohlcv_with_indicators, indicators=("macd",))
        names = {t.name for t in fig.data}
        # macd line, signal line, histogram - three named MACD-related traces.
        assert {"MACD", "Signal", "Histogram"}.issubset(names)

    def test_bollinger_renders_three_band_traces(self, ohlcv_with_indicators):
        fig = charts.build_figure(ohlcv_with_indicators, indicators=("bb",))
        names = {t.name for t in fig.data}
        assert {"BB Upper", "BB Lower", "BB Middle"}.issubset(names)


class TestGracefulDegradation:
    def test_missing_indicator_columns_are_skipped(self, ohlcv_with_indicators):
        # Drop the bb_* columns; request "bb" anyway - should NOT raise and
        # should NOT add band traces.
        df = ohlcv_with_indicators.drop(columns=["bb_upper", "bb_middle", "bb_lower"])
        fig = charts.build_figure(df, indicators=("bb",))
        names = {t.name for t in fig.data}
        assert "BB Upper" not in names

    def test_minimal_ohlcv_only_no_indicators(self, ohlcv_with_indicators):
        cols = ["Open", "High", "Low", "Close", "Volume"]
        df = ohlcv_with_indicators[cols]
        fig = charts.build_figure(df, indicators=())
        # Must produce a valid figure with Candlestick + Volume bars.
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2


class TestLayout:
    def test_rangeslider_is_off(self, ohlcv_with_indicators):
        fig = charts.build_figure(ohlcv_with_indicators, indicators=("rsi", "macd"))
        # Plotly stores rangeslider visibility under xaxis.rangeslider.visible
        # for candlestick traces; the global xaxis_rangeslider_visible=False is
        # reflected here.
        assert fig.layout.xaxis.rangeslider.visible is False

    def test_rangeselector_on_bottom_axis(self, ohlcv_with_indicators):
        fig = charts.build_figure(ohlcv_with_indicators, indicators=("rsi", "macd"))
        # 4 rows → bottom axis is xaxis4.
        assert fig.layout.xaxis4.rangeselector is not None
        assert fig.layout.xaxis4.rangeselector.buttons is not None

    def test_hover_unified(self, ohlcv_with_indicators):
        fig = charts.build_figure(ohlcv_with_indicators, indicators=())
        assert fig.layout.hovermode == "x unified"

    def test_volume_yaxis_si_tickformat(self, ohlcv_with_indicators):
        fig = charts.build_figure(ohlcv_with_indicators, indicators=())
        assert fig.layout.yaxis2.tickformat == "~s"
