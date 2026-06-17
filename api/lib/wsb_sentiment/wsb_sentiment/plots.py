"""Plotly figure builders (LAZY plotly).

Each builder returns a plain ``dict`` shaped ``{"data": [...], "layout": {...}}`` —
the same JSON shape the FastAPI layer serializes and the Next.js ``PlotlyChart``
component renders — so no Plotly object leaks across the API boundary. Plotly is
an OPTIONAL dependency (the ``viz`` extra) imported LAZILY inside each builder;
importing this module has no side effects and does not require Plotly.

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

#: A Plotly figure serialized as a plain mapping with ``data`` and ``layout`` keys.
FigureDict = dict[str, Any]

# quantcore-candidate: mirrors hrp-portfolio:src/hrp/plots.py ({data, layout} shape).


def _jsonify(value: Any) -> Any:
    """Recursively convert numpy/pandas scalars and arrays to native Python types."""
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonify(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Period):
        return str(value)
    return value


def _iso_index(index: pd.Index) -> list[str]:
    """Render a (date) index as a list of ISO-8601 / string x-values."""
    out: list[str] = []
    for value in index:
        if isinstance(value, pd.Timestamp):
            out.append(value.isoformat())
        else:
            out.append(str(value))
    return out


def _cumulative_wealth(returns: pd.Series) -> pd.Series:
    """Cumulative wealth ``prod(1 + r)`` of a per-period return series (NaN -> 0)."""
    clean = returns.astype("float64").fillna(0.0)
    return (1.0 + clean).cumprod()


def oos_equity_figure(
    net_returns: pd.Series,
    buyhold_returns: pd.Series,
    *,
    title: str = "Out-of-sample equity: signal vs buy-and-hold",
) -> FigureDict:
    """Build the OOS equity-curve figure (signal vs buy-and-hold).

    Plots cumulative wealth for the net (after-cost) sentiment signal against the
    equal-weight buy-and-hold baseline over the out-of-sample window.

    Parameters
    ----------
    net_returns:
        The net OOS return series of the signal.
    buyhold_returns:
        The OOS return series of the buy-and-hold baseline.
    title:
        The figure title.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` mapping rendering the two equity curves.
    """
    net = pd.Series(net_returns, dtype="float64")
    buyhold = pd.Series(buyhold_returns, dtype="float64")

    net_wealth = _cumulative_wealth(net)
    buyhold_wealth = _cumulative_wealth(buyhold)

    data = [
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Sentiment signal (net)",
            "x": _iso_index(net_wealth.index),
            "y": _jsonify(net_wealth.to_numpy()),
            "line": {"color": "#2563eb", "width": 2},
        },
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Buy & hold",
            "x": _iso_index(buyhold_wealth.index),
            "y": _jsonify(buyhold_wealth.to_numpy()),
            "line": {"color": "#9ca3af", "width": 2, "dash": "dash"},
        },
    ]
    layout = {
        "title": {"text": title},
        "xaxis": {"title": {"text": "Date"}},
        "yaxis": {"title": {"text": "Cumulative wealth (growth of 1)"}},
        "legend": {"orientation": "h"},
        "template": "plotly_white",
        "hovermode": "x unified",
    }
    return {"data": data, "layout": layout}


def _aggregate_series(panel: pd.DataFrame, ticker: str | None, *, how: str) -> pd.Series:
    """Reduce a wide ``day x ticker`` panel to one daily series.

    When ``ticker`` is given, the single column is returned; otherwise the panel is
    aggregated across tickers — by ``mean`` (sentiment) or ``sum`` (mention count).
    """
    if ticker is not None:
        if ticker not in panel.columns:
            from wsb_sentiment._exceptions import ValidationError

            raise ValidationError(
                f"sentiment_figure: ticker {ticker!r} not in panel columns "
                f"{[str(c) for c in panel.columns]}."
            )
        return panel[ticker].astype("float64")
    if how == "sum":
        return panel.sum(axis=1, skipna=True).astype("float64")
    return panel.mean(axis=1, skipna=True).astype("float64")


def sentiment_figure(
    mean_compound: pd.DataFrame,
    mention_count: pd.DataFrame,
    *,
    ticker: str | None = None,
    title: str = "Daily WSB sentiment and mention count",
) -> FigureDict:
    """Build the daily-sentiment + mention-count figure.

    Plots the daily mean compound sentiment (left axis, line) against the mention
    count (right axis, bars), either aggregated across tickers or for a single
    ``ticker``.

    Parameters
    ----------
    mean_compound:
        Wide ``day x ticker`` mean compound sentiment.
    mention_count:
        Wide ``day x ticker`` mention count.
    ticker:
        Optional single ticker to plot; when ``None`` an aggregate is shown
        (mean sentiment, total mentions).
    title:
        The figure title.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` mapping rendering sentiment and mention count.
    """
    sentiment = pd.DataFrame(mean_compound).astype("float64")
    mentions = pd.DataFrame(mention_count).astype("float64")

    sent_series = _aggregate_series(sentiment, ticker, how="mean")
    mention_series = _aggregate_series(mentions, ticker, how="sum")

    data = [
        {
            "type": "bar",
            "name": "Mentions",
            "x": _iso_index(mention_series.index),
            "y": _jsonify(mention_series.to_numpy()),
            "yaxis": "y2",
            "marker": {"color": "#e5e7eb"},
            "opacity": 0.7,
        },
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Mean compound sentiment",
            "x": _iso_index(sent_series.index),
            "y": _jsonify(sent_series.to_numpy()),
            "line": {"color": "#16a34a", "width": 2},
        },
    ]
    suffix = f" — {ticker}" if ticker is not None else " — aggregate"
    layout = {
        "title": {"text": title + suffix},
        "xaxis": {"title": {"text": "Date"}},
        "yaxis": {"title": {"text": "Mean compound sentiment"}, "range": [-1.0, 1.0]},
        "yaxis2": {
            "title": {"text": "Mention count"},
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
            "rangemode": "tozero",
        },
        "legend": {"orientation": "h"},
        "template": "plotly_white",
        "bargap": 0.0,
        "hovermode": "x unified",
    }
    return {"data": data, "layout": layout}
