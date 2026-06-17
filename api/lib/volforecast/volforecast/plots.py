"""Plotly figure builders (lazy plotly).

Each builder returns a plain ``dict`` shaped ``{"data": [...], "layout": {...}}``
— the same JSON shape the FastAPI layer serializes and the Next.js
``PlotlyChart`` component renders — so figures cross the API boundary with no
Plotly object leaking through. Plotly is an OPTIONAL dependency (the ``viz``
extra) imported lazily inside each builder; importing this module has no side
effects and does not require Plotly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from volforecast._exceptions import ValidationError
from volforecast._validation import ensure_dataframe, ensure_series

# quantcore-candidate: mirrors hrp / markowitz / pairs-trading plots.py
# ({data, layout} figure shape).

#: A Plotly figure serialized as a plain mapping with ``data`` and ``layout`` keys.
FigureDict = dict[str, Any]

#: Colour used to highlight the ``best_model`` bar in the QLIKE chart.
_HIGHLIGHT_COLOR = "#2563eb"
#: Muted colour for the non-best bars (the honest, de-emphasised challengers).
_MUTED_COLOR = "#94a3b8"


def _jsonify(value: Any) -> Any:
    """Recursively convert numpy/pandas scalars and arrays to native Python types."""
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonify(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp | pd.Period):
        return value.isoformat() if hasattr(value, "isoformat") else str(value)
    if isinstance(value, float) and not np.isfinite(value):
        # JSON has no NaN/Inf literal; map non-finite floats to ``None`` so the
        # figure stays JSON-serializable across the API boundary.
        return None
    return value


def _x_axis(index: pd.Index) -> list[str]:
    """Render a (possibly datetime) index as ISO strings so no Timestamp leaks."""
    return [v.isoformat() if hasattr(v, "isoformat") else str(v) for v in index]


def _y_values(series: pd.Series) -> list[Any]:
    """Render a float series to a JSON-safe list (NaN -> ``None``)."""
    return [_jsonify(float(v)) for v in series.to_numpy(dtype="float64")]


def rv_forecast_figure(
    realized_vol: pd.Series,
    forecasts: pd.DataFrame,
    *,
    title: str = "Realized volatility: actual vs model forecasts",
) -> FigureDict:
    """Build the RV-actual-vs-forecasts line figure.

    Plots the realized forward volatility as a reference line plus one line per
    model column in ``forecasts``, all on a shared date axis.

    LAZY IMPORT: ``plotly`` is imported inside this function.

    Parameters
    ----------
    realized_vol:
        The realized forward-volatility series (the truth) indexed by date.
    forecasts:
        A ``(T, M)`` frame of per-model forecasts aligned to ``realized_vol``.
    title:
        The figure title.

    Returns
    -------
    FigureDict
        A ``{"data": [...], "layout": {...}}`` mapping.

    Raises
    ------
    ValidationError
        If ``realized_vol`` and ``forecasts`` cannot be aligned.
    """
    # Coerce/validate; NaN is permitted so partially-covered folds still plot.
    actual = ensure_series(realized_vol, name="realized_vol", allow_nan=True)
    frame = ensure_dataframe(forecasts, name="forecasts", allow_nan=True)

    # Align every model forecast to the realized-vol index (inner-join is the
    # no-lookahead-safe way to line up panels with differing coverage).
    common = actual.index.intersection(frame.index)
    if len(common) == 0:
        raise ValidationError(
            "rv_forecast_figure: realized_vol and forecasts share no common index."
        )
    common = common.sort_values()
    actual = actual.reindex(common)
    frame = frame.reindex(common)

    x_axis = _x_axis(common)

    # The realized (truth) line is drawn first and styled distinctly so it reads
    # as the reference the forecasts are chasing.
    data: list[dict[str, Any]] = [
        {
            "type": "scatter",
            "mode": "lines",
            "name": "realized vol",
            "x": x_axis,
            "y": _y_values(actual),
            "line": {"color": "#111827", "width": 2},
        }
    ]
    # One line per model column, in the caller's column order.
    for column in frame.columns:
        data.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": str(column),
                "x": x_axis,
                "y": _y_values(frame[column]),
            }
        )

    layout = {
        "title": {"text": title},
        "xaxis": {"title": {"text": "date"}},
        "yaxis": {"title": {"text": "realized volatility"}},
        "legend": {"orientation": "h"},
    }
    return {"data": data, "layout": layout}


def qlike_bar_figure(
    qlike_by_model: dict[str, float],
    *,
    best_model: str | None = None,
    title: str = "Out-of-sample QLIKE by model (lower is better)",
) -> FigureDict:
    """Build the QLIKE-by-model bar figure (the headline error chart).

    One bar per model, sorted ascending (best first); the ``best_model`` bar is
    highlighted so the honest ranking is obvious at a glance.

    LAZY IMPORT: ``plotly`` is imported inside this function.

    Parameters
    ----------
    qlike_by_model:
        Mapping ``{model_label: mean_OOS_QLIKE}``.
    best_model:
        The label to highlight (defaults to the argmin of ``qlike_by_model``).
    title:
        The figure title.

    Returns
    -------
    FigureDict
        A ``{"data": [...], "layout": {...}}`` mapping.

    Raises
    ------
    ValidationError
        If ``qlike_by_model`` is empty.
    """
    if not qlike_by_model:
        raise ValidationError("qlike_bar_figure: qlike_by_model must be non-empty.")

    # Sort ascending so the best (lowest QLIKE) model is leftmost; non-finite
    # QLIKE values sort last (treated as +inf) without crashing the sort.
    def _sort_key(item: tuple[str, float]) -> float:
        value = float(item[1])
        return value if np.isfinite(value) else float("inf")

    ordered = sorted(qlike_by_model.items(), key=_sort_key)
    labels = [str(label) for label, _ in ordered]
    values = [_jsonify(float(value)) for _, value in ordered]

    # The highlighted model defaults to the argmin (the first after sorting).
    highlight = str(best_model) if best_model is not None else labels[0]
    colors = [_HIGHLIGHT_COLOR if label == highlight else _MUTED_COLOR for label in labels]

    data = [
        {
            "type": "bar",
            "x": labels,
            "y": values,
            "marker": {"color": colors},
            "name": "OOS QLIKE",
        }
    ]
    layout = {
        "title": {"text": title},
        "xaxis": {"title": {"text": "model"}},
        "yaxis": {"title": {"text": "mean QLIKE"}},
    }
    return {"data": data, "layout": layout}
