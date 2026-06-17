"""Plotly figure builders for the anomaly-detector tool.

Each builder returns a plain ``dict`` shaped ``{"data": [...], "layout": {...}}``
— the same JSON shape the FastAPI layer serializes and the Next.js
``PlotlyChart`` component renders — so no Plotly object leaks across the API
boundary. Plotly is an OPTIONAL dependency (the ``viz`` extra) imported LAZILY
inside each builder; importing this module has no side effects and does not
require Plotly.

Two figures back the tool:

- :func:`price_anomaly_figure` — the price (and/or return) series with markers on
  the flagged anomalous days.
- :func:`score_threshold_figure` — the per-day anomaly-score series with the
  train-derived threshold drawn as a horizontal line.

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import pandas as pd

#: A Plotly figure serialized as a plain mapping with ``data`` and ``layout`` keys.
FigureDict = dict[str, Any]


def _jsonify(value: Any) -> Any:
    """Recursively convert numpy/pandas scalars and arrays to native Python types.

    Non-finite floats (NaN/Inf) are mapped to ``None`` so the emitted figure is
    strictly JSON-safe (the API contract forbids non-finite floats), matching the
    ``_safe_float`` discipline used elsewhere in the package.
    """
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonify(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonify(value.item())
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return value.isoformat() if hasattr(value, "isoformat") else str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _as_plain_dict(obj: Any) -> dict[str, Any]:
    """Coerce a Plotly graph-object (trace/layout) to a plain, JSON-safe ``dict``.

    Plotly graph-objects expose ``.to_plotly_json()``; the result is a nested
    mapping of plain Python / numpy types. We round-trip numpy scalars/arrays to
    native types so the figure crosses the API boundary with no Plotly object
    leaking through and no non-finite floats surviving.
    """
    raw = obj.to_plotly_json() if hasattr(obj, "to_plotly_json") else dict(obj)
    jsonified = _jsonify(raw)
    # ``raw`` is a mapping (a graph-object's JSON), so ``_jsonify`` returns a dict.
    return dict(jsonified)


def _x_axis(index: pd.Index) -> list[Any]:
    """Render a (possibly datetime) index as a JSON-safe x-axis list."""
    return [v.isoformat() if hasattr(v, "isoformat") else str(v) for v in index]


def price_anomaly_figure(
    prices: pd.Series,
    flags: pd.Series,
    *,
    title: str = "Price with flagged anomalies",
) -> FigureDict:
    """Build a price line chart with markers on flagged anomalous days.

    Renders the price series as a line and overlays a scatter of markers at the
    dates where ``flags`` is ``True``, so the reader sees where each anomaly
    landed relative to the price path. The flag series is aligned to the price
    index (inner join) before the markers are placed, so a flag set indexed by
    the OOS slice only marks its own dates.

    Parameters
    ----------
    prices:
        The price (or return) level series, indexed by date.
    flags:
        Boolean per-day flag series aligned to ``prices``.
    title:
        The figure title.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` mapping.
    """
    # LAZY import: keep Plotly off this pure module's import path.
    import plotly.graph_objects as go

    price_s = prices if isinstance(prices, pd.Series) else pd.Series(prices)
    price_s = price_s.astype("float64")

    flag_s = flags if isinstance(flags, pd.Series) else pd.Series(flags)
    # Align flags onto the price index so markers only land on real price points;
    # missing/NaN flags become False (no marker).
    flag_aligned = flag_s.reindex(price_s.index).fillna(False).astype(bool)
    flagged_idx = price_s.index[flag_aligned.to_numpy()]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=_x_axis(price_s.index),
            y=[_jsonify(v) for v in price_s.to_numpy(dtype="float64")],
            mode="lines",
            name="price",
            line={"color": "#1f77b4", "width": 1.5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=_x_axis(flagged_idx),
            y=[_jsonify(v) for v in price_s.reindex(flagged_idx).to_numpy(dtype="float64")],
            mode="markers",
            name="anomaly",
            marker={"color": "#d62728", "size": 9, "symbol": "x"},
        )
    )

    layout = _as_plain_dict(fig.layout)
    layout["title"] = {"text": title}
    layout["xaxis"] = {"title": {"text": "date"}}
    layout["yaxis"] = {"title": {"text": "price"}}
    layout["legend"] = {"orientation": "h"}
    return {"data": [_as_plain_dict(trace) for trace in fig.data], "layout": layout}


def score_threshold_figure(
    scores: pd.Series,
    threshold: float,
    *,
    title: str = "Anomaly score with threshold",
) -> FigureDict:
    """Build the per-day anomaly-score series with a horizontal threshold line.

    Renders ``scores`` as a line and draws the train-derived ``threshold`` as a
    horizontal reference line, so the reader sees which days breach it. Days whose
    score exceeds the threshold are additionally marked, so a breach is visible
    even at a glance.

    Parameters
    ----------
    scores:
        The per-day anomaly score series over the OOS slice, indexed by date.
    threshold:
        The train-derived flag threshold to draw as a horizontal line.
    title:
        The figure title.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` mapping.
    """
    # LAZY import: keep Plotly off this pure module's import path.
    import plotly.graph_objects as go

    score_s = scores if isinstance(scores, pd.Series) else pd.Series(scores)
    score_s = score_s.astype("float64")
    thr = float(threshold)

    breach = score_s.to_numpy(dtype="float64") > thr
    breach_idx = score_s.index[breach]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=_x_axis(score_s.index),
            y=[_jsonify(v) for v in score_s.to_numpy(dtype="float64")],
            mode="lines",
            name="anomaly score",
            line={"color": "#2ca02c", "width": 1.5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=_x_axis(breach_idx),
            y=[_jsonify(v) for v in score_s.reindex(breach_idx).to_numpy(dtype="float64")],
            mode="markers",
            name="breach",
            marker={"color": "#d62728", "size": 8, "symbol": "circle"},
        )
    )

    layout = _as_plain_dict(fig.layout)
    layout["title"] = {"text": title}
    layout["xaxis"] = {"title": {"text": "date"}}
    layout["yaxis"] = {"title": {"text": "anomaly score"}}
    layout["legend"] = {"orientation": "h"}
    # Horizontal threshold line spanning the full x-range (drawn in paper-x so it
    # never depends on the date axis bounds).
    layout["shapes"] = [
        {
            "type": "line",
            "xref": "paper",
            "yref": "y",
            "x0": 0.0,
            "x1": 1.0,
            "y0": thr,
            "y1": thr,
            "line": {"color": "firebrick", "dash": "dash", "width": 2},
        }
    ]
    layout["annotations"] = [
        {
            "xref": "paper",
            "yref": "y",
            "x": 1.0,
            "y": thr,
            "xanchor": "right",
            "yanchor": "bottom",
            "text": f"threshold = {thr:.4g}",
            "showarrow": False,
            "font": {"color": "firebrick"},
        }
    ]
    return {"data": [_as_plain_dict(trace) for trace in fig.data], "layout": layout}
