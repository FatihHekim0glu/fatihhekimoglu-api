"""Plotly figure builders (no Plotly dependency; the ``viz`` extra is for render).

Each builder returns a plain ``dict`` shaped ``{"data": [...], "layout": {...}}``
— the same Plotly-schema JSON the FastAPI layer serializes and the Next.js
``PlotlyChart`` component renders — built directly from native Python types, so no
Plotly object ever crosses the API boundary and the builders do not import Plotly
at all. Plotly/kaleido (the OPTIONAL ``viz`` extra) are only needed downstream to
*render* these dicts to an image; importing this module has no side effects and
does not require Plotly.

The two figures back the honest story: predicted-vs-actual next-day RETURNS (not
price levels) and the model-vs-persistence error bar chart (equal bars on a
random walk — the visual of the documented NULL).

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from lstmforecast._typing import FloatArray

# quantcore-candidate: mirrors hrp / pairs-trading plots.py ({data, layout} shape).

#: A Plotly figure serialized as a plain mapping with ``data`` and ``layout`` keys.
FigureDict = dict[str, Any]


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
    return value


def _coerce_1d(value: Any, *, name: str) -> FloatArray:
    """Coerce ``value`` to a finite, non-empty 1-D float64 array.

    Raises :class:`ValidationError` (not a bare ``ValueError``) so the FastAPI
    layer can map a malformed figure request to a 422, consistent with the rest
    of the library's boundary discipline.
    """
    from lstmforecast._exceptions import ValidationError

    arr = np.asarray(value, dtype="float64").reshape(-1)
    if arr.size == 0:
        raise ValidationError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(arr)):
        raise ValidationError(f"{name} must contain only finite values.")
    return arr


def forecast_vs_actual_figure(
    dates: pd.Index,
    y_true: FloatArray,
    y_pred_model: FloatArray,
) -> FigureDict:
    """Build the predicted-vs-actual next-day RETURN series figure.

    Two line traces over ``dates``: realized next-day returns and the model's
    forecast returns. NEVER plots price levels (the whole point is return-space
    honesty). The ``{data, layout}`` dict is built from native types — Plotly is
    not imported.

    Parameters
    ----------
    dates:
        The out-of-sample target dates.
    y_true:
        Realized next-day returns.
    y_pred_model:
        The model's forecast returns (same length).

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` mapping rendering the two return series.

    Raises
    ------
    ValidationError
        If the inputs are empty or length-mismatched.
    """
    # We build the {data, layout} dicts directly (no Plotly graph-objects needed),
    # so the ``viz`` extra stays entirely off this pure module's import path.
    from lstmforecast._exceptions import ValidationError

    actual = _coerce_1d(y_true, name="y_true")
    predicted = _coerce_1d(y_pred_model, name="y_pred_model")
    if actual.size != predicted.size:
        raise ValidationError(
            f"y_true and y_pred_model must be the same length, got "
            f"{actual.size} and {predicted.size}."
        )

    index = pd.Index(dates)
    if len(index) != actual.size:
        raise ValidationError(
            f"dates must align with the return series length {actual.size}, got {len(index)}."
        )
    # ``_jsonify`` maps pandas Timestamp/Period to ISO strings (and any other
    # value to a native type), so no Timestamp leaks across the API boundary.
    x_values = [_jsonify(v) for v in index]

    data = [
        {
            "type": "scatter",
            "mode": "lines",
            "name": "actual",
            "x": x_values,
            "y": _jsonify(actual),
            "line": {"color": "#2563eb"},
        },
        {
            "type": "scatter",
            "mode": "lines",
            "name": "model forecast",
            "x": x_values,
            "y": _jsonify(predicted),
            "line": {"color": "#f97316"},
        },
    ]
    layout = {
        "title": {"text": "Predicted vs. actual next-day return"},
        "xaxis": {"title": {"text": "date"}},
        "yaxis": {"title": {"text": "log-return"}, "tickformat": ".2%", "zeroline": True},
        "legend": {"orientation": "h"},
    }
    return {"data": data, "layout": layout}


def error_vs_baseline_figure(
    y_true: FloatArray,
    y_pred_model: FloatArray,
    y_pred_naive: FloatArray | None = None,
) -> FigureDict:
    """Build the model-vs-persistence out-of-sample error bar chart.

    A two-bar figure comparing the model's return-space error (RMSE and MAE) to
    the persistence baseline's — the visual of the honest NULL (the bars are
    essentially equal on random-walk data). The ``{data, layout}`` dict is built
    from native types — Plotly is not imported.

    Parameters
    ----------
    y_true:
        Realized next-day returns.
    y_pred_model:
        The model's forecasts.
    y_pred_naive:
        The persistence forecasts; defaults to all zeros (``r_hat = 0``).

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` grouped bar-chart mapping.

    Raises
    ------
    ValidationError
        If the inputs are empty or length-mismatched.
    """
    from lstmforecast._exceptions import ValidationError

    actual = _coerce_1d(y_true, name="y_true")
    model = _coerce_1d(y_pred_model, name="y_pred_model")
    if actual.size != model.size:
        raise ValidationError(
            f"y_true and y_pred_model must be the same length, got {actual.size} and {model.size}."
        )

    if y_pred_naive is None:
        # Persistence forecast: r_hat = 0 (the random-walk price forecast).
        naive = np.zeros(actual.size, dtype="float64")
    else:
        naive = _coerce_1d(y_pred_naive, name="y_pred_naive")
        if naive.size != actual.size:
            raise ValidationError(
                f"y_pred_naive must align with y_true length {actual.size}, got {naive.size}."
            )

    def _rmse(pred: FloatArray) -> float:
        return float(np.sqrt(np.mean((actual - pred) ** 2)))

    def _mae(pred: FloatArray) -> float:
        return float(np.mean(np.abs(actual - pred)))

    metric_labels = ["RMSE", "MAE"]
    model_errors = [_rmse(model), _mae(model)]
    naive_errors = [_rmse(naive), _mae(naive)]

    data = [
        {
            "type": "bar",
            "name": "LSTM model",
            "x": metric_labels,
            "y": _jsonify(model_errors),
            "marker": {"color": "#f97316"},
        },
        {
            "type": "bar",
            "name": "persistence (naive)",
            "x": metric_labels,
            "y": _jsonify(naive_errors),
            "marker": {"color": "#2563eb"},
        },
    ]
    layout = {
        "title": {"text": "Return-space error: model vs. persistence baseline"},
        "barmode": "group",
        "xaxis": {"title": {"text": "metric"}},
        "yaxis": {"title": {"text": "error (return units)"}},
        "legend": {"orientation": "h"},
    }
    return {"data": data, "layout": layout}
