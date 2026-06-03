"""Plotly chart builders for the dashboard.

Opinionated defaults from research:

  - 2-4 rows dynamically: price and volume always present; RSI and MACD added
    only when requested (and their columns are present in the DataFrame).
    Row heights are 0.55 for price and 0.15 for each remaining row, then
    normalised so they sum to 1.0.
  - shared_xaxes=True so zoom and pan stay in sync.
  - hovermode='x unified' plus spike lines on the x-axis for synchronised cursor.
  - rangeslider OFF — it is the single biggest perf killer at 2500 bars
    because it renders a second candlestick that reflows on every interaction.
    Offer a rangeselector button row (1M / 6M / YTD / 1Y / 5Y / MAX) instead.
  - rangebreaks=[dict(bounds=["sat", "mon"])] to kill the weekend gap on the x-axis.
  - connectgaps=False (the default) so warmup NaNs at the start of EMA / RSI / MACD
    do NOT draw a line back to zero.
  - Colourblind-safe up/down: #00897B (teal-700) / #C62828 (red-800), Material
    palette. Both pass WCAG AA contrast against white (4.18:1 and 6.36:1
    respectively), so they are legible as both graphics and text. Apply
    identically to candle increasing/decreasing colours and to volume /
    MACD-histogram bar colours.
  - Bollinger band as a filled region: add upper trace, then lower trace with
    fill="tonexty" and a low-alpha fillcolor — do NOT draw three opaque lines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_UP = "#00897B"  # Material teal-700, 4.18:1 vs white — passes WCAG AA
_DOWN = "#C62828"  # Material red-800, 6.36:1 vs white — passes WCAG AA
_BB_FILL = "rgba(0,137,123,0.12)"  # alpha-blended _UP

# NYSE calendar (XNYS). Reused across calls to avoid re-instantiation cost.
_NYSE_CAL = mcal.get_calendar("XNYS")


def _nyse_holiday_dates(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """NYSE holidays in [start, end] as pd.Timestamp list, for Plotly rangebreaks.

    Without this, weekday holidays (MLK Day, Presidents Day, Good Friday,
    Memorial Day, Juneteenth, July 4, Labor Day, Thanksgiving, Christmas)
    render as phantom flat gaps in the candlestick chart — ~9 per year.
    """
    # `.holidays` is a tuple of numpy.datetime64 in current pandas-market-calendars;
    # wrap in np.asarray so boolean masking works (iter-10 regression fix).
    holidays = np.asarray(_NYSE_CAL.holidays().holidays)
    mask = (holidays >= start.to_datetime64()) & (holidays <= end.to_datetime64())
    return [pd.Timestamp(h) for h in holidays[mask]]


def _compute_layout(want_rsi: bool, want_macd: bool) -> dict:
    """Compute subplot layout parameters based on requested optional rows.

    Returns a dict with keys: n_rows, row_heights, subplot_titles, rsi_row,
    macd_row, height.
    """
    extra_titles: list[str] = []
    rsi_row: int | None = None
    macd_row: int | None = None
    if want_rsi:
        extra_titles.append("RSI")
        rsi_row = 2 + len(extra_titles)
    if want_macd:
        extra_titles.append("MACD")
        macd_row = 2 + len(extra_titles)

    n_rows = 2 + len(extra_titles)
    row_heights = [0.55, 0.15] + [0.15] * len(extra_titles)
    # Normalise so heights sum to 1.0
    total = sum(row_heights)
    row_heights = [h / total for h in row_heights]

    # Figure height scales with the number of optional rows, capped at 760px
    # so the entire chart (plus app header) fits within a single mobile
    # viewport on a typical phone (~800px tall). Desktop effective heights:
    # 500 / 680 / 760 for 0 / 1 / 2 extra rows.
    height = min(500 + 180 * len(extra_titles), 760)

    return {
        "n_rows": n_rows,
        "row_heights": row_heights,
        "subplot_titles": ("Price", "Volume", *extra_titles),
        "rsi_row": rsi_row,
        "macd_row": macd_row,
        "height": height,
    }


def _apply_layout(
    fig: go.Figure,
    n_rows: int,
    height: int,
    holidays: list[pd.Timestamp] | None = None,
) -> None:
    """Apply axis and layout styling to the assembled figure."""
    rangebreaks: list[dict] = [dict(bounds=["sat", "mon"])]
    if holidays:
        rangebreaks.append(dict(values=holidays))
    fig.update_xaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="dot",
        hoverformat="%Y-%m-%d",
        rangebreaks=rangebreaks,
    )
    # Volume in SI suffixes (K / M / B) instead of scientific notation.
    fig.update_yaxes(tickformat="~s", row=2, col=1)

    # Range selector buttons attach to the bottom xaxis, which depends on n_rows.
    bottom_xaxis = f"xaxis{n_rows}"
    fig.update_layout(
        **{
            bottom_xaxis: dict(
                rangeselector=dict(
                    buttons=[
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=5, label="5Y", step="year", stepmode="backward"),
                        dict(label="MAX", step="all"),
                    ]
                ),
            ),
            "xaxis_rangeslider_visible": False,
            "hovermode": "x unified",
            "height": height,
            "template": "plotly_white",
            "showlegend": True,
            "legend": dict(orientation="h", y=1.02, x=0),
        }
    )


def build_figure(
    df: pd.DataFrame,
    indicators: tuple[str, ...] = (),
    candlestick: bool = True,
) -> go.Figure:
    """Build the dashboard figure with a dynamic number of subplots (2-4 rows).

    Parameters
    ----------
    df:
        OHLCV DataFrame indexed by date. Should already contain whichever
        indicator columns are requested (computed upstream in src.indicators).
    indicators:
        Tuple of indicator names to draw, e.g. ("sma_20", "sma_50", "rsi", "macd").
        Must be a tuple (not a list) so it's hashable for Streamlit caching.
    candlestick:
        True for candlesticks on the price subplot, False for a single line of Close.
    """
    requested = set(indicators)
    cols = set(df.columns)

    want_rsi = "rsi" in requested and "rsi" in cols
    want_macd = "macd" in requested and "macd" in cols

    # Build row list: price is always row 1, volume always row 2.
    # RSI and MACD are optional, in that order.
    layout = _compute_layout(want_rsi, want_macd)
    rsi_row = layout["rsi_row"]
    macd_row = layout["macd_row"]

    fig = make_subplots(
        rows=layout["n_rows"],
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=layout["row_heights"],
        subplot_titles=layout["subplot_titles"],
    )

    # ---- Row 1: Price ----------------------------------------------------
    if candlestick:
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Price",
                increasing_line_color=_UP,
                decreasing_line_color=_DOWN,
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Close"],
                mode="lines",
                name="Close",
                line=dict(color=_UP),
                connectgaps=False,
            ),
            row=1,
            col=1,
        )

    # SMA overlays
    for name in ("sma_20", "sma_50"):
        if name in requested and name in cols:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[name],
                    mode="lines",
                    name=name.upper().replace("_", " "),
                    connectgaps=False,
                ),
                row=1,
                col=1,
            )

    # EMA overlays
    for name in ("ema_12", "ema_26"):
        if name in requested and name in cols:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[name],
                    mode="lines",
                    name=name.upper().replace("_", " "),
                    connectgaps=False,
                ),
                row=1,
                col=1,
            )

    # Bollinger Bands — only if requested AND all three columns are present
    if "bb" in requested and {"bb_upper", "bb_middle", "bb_lower"}.issubset(cols):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["bb_upper"],
                mode="lines",
                name="BB Upper",
                line=dict(width=0),
                connectgaps=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["bb_lower"],
                mode="lines",
                name="BB Lower",
                fill="tonexty",
                fillcolor=_BB_FILL,
                line=dict(width=0),
                connectgaps=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["bb_middle"],
                mode="lines",
                name="BB Middle",
                line=dict(width=1, dash="dash"),
                connectgaps=False,
            ),
            row=1,
            col=1,
        )

    # ---- Row 2: Volume ---------------------------------------------------
    vol_colors = [_UP if c >= o else _DOWN for c, o in zip(df["Close"], df["Open"], strict=True)]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
            marker_color=vol_colors,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # ---- Optional row: RSI ----------------------------------------------
    if rsi_row is not None:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["rsi"],
                mode="lines",
                name="RSI",
                connectgaps=False,
            ),
            row=rsi_row,
            col=1,
        )
        fig.add_hline(
            y=30,
            row=rsi_row,
            col=1,
            line_dash="dot",
            line_color="gray",
            annotation_text="oversold (30)",
            annotation_position="right",
        )
        fig.add_hline(
            y=70,
            row=rsi_row,
            col=1,
            line_dash="dot",
            line_color="gray",
            annotation_text="overbought (70)",
            annotation_position="right",
        )

    # ---- Optional row: MACD ---------------------------------------------
    if macd_row is not None:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["macd"],
                mode="lines",
                name="MACD",
                connectgaps=False,
            ),
            row=macd_row,
            col=1,
        )
        if "macd_signal" in cols:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["macd_signal"],
                    mode="lines",
                    name="Signal",
                    connectgaps=False,
                ),
                row=macd_row,
                col=1,
            )
        if "macd_hist" in cols:
            hist_colors = [_UP if v >= 0 else _DOWN for v in df["macd_hist"]]
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df["macd_hist"],
                    name="Histogram",
                    marker_color=hist_colors,
                    showlegend=False,
                ),
                row=macd_row,
                col=1,
            )

    # ---- Layout ----------------------------------------------------------
    holidays = _nyse_holiday_dates(df.index.min(), df.index.max())
    _apply_layout(fig, layout["n_rows"], layout["height"], holidays)

    return fig
