"""Plotly figure builders (LAZY plotly, [viz]) — serialize to ``{data, layout}``.

Two figures back the frontend tool:

- :func:`forecast_figure` — the realized target return vs. each model's forecast
  over the OOS window (so the eye can see that the deep forecasts hug zero, like
  naive);
- :func:`error_figure` — a grouped bar of return-space RMSE and directional
  accuracy by model (so the lack of a deep edge is legible at a glance).

``plotly`` is imported LAZILY inside the builders (the ``[viz]`` extra), and each
figure is returned as a plain ``{data, layout}`` dict via
``json.loads(pio.to_json(fig, validate=False))`` so the API response carries no
plotly objects. Importing this module has no side effects.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mvtsforecast._exceptions import ValidationError
from mvtsforecast._typing import FloatArray

# quantcore-candidate: mirrors lstm-forecast / hrp plots.py ({data, layout} shape).

#: Stable, colour-blind-aware palette keyed by trace index (cycles if exceeded).
_PALETTE: tuple[str, ...] = (
    "#111827",  # near-black — reserved for the realized/actual series
    "#2563eb",  # blue — naive
    "#f97316",  # orange
    "#16a34a",  # green
    "#9333ea",  # purple
    "#dc2626",  # red
    "#0891b2",  # cyan
)


def _coerce_1d(value: Any, *, name: str) -> FloatArray:
    """Coerce ``value`` to a finite, non-empty 1-D float64 array.

    Raises :class:`ValidationError` (not a bare ``ValueError``) so the FastAPI
    layer can map a malformed figure request to a 422, consistent with the rest
    of the library's boundary discipline.
    """
    arr = np.asarray(value, dtype="float64").reshape(-1)
    if arr.size == 0:
        raise ValidationError(f"{name} must be non-empty.")
    if not bool(np.isfinite(arr).all()):
        raise ValidationError(f"{name} must contain only finite values.")
    return arr


def forecast_figure(
    y_true: FloatArray,
    forecasts_by_model: Mapping[str, FloatArray],
    *,
    title: str = "Actual vs. forecast (next-step returns)",
) -> dict[str, Any]:
    """Build the actual-vs-forecast Plotly figure as a ``{data, layout}`` dict.

    LAZY IMPORT: ``plotly`` is imported inside this function (the ``[viz]`` extra).
    Plots the realized target return as one trace and each model's forecast as an
    overlaid trace, then serializes to a plain dict.

    Parameters
    ----------
    y_true:
        The realized next-step target returns over the OOS window.
    forecasts_by_model:
        Map of model name -> its ``(n_obs,)`` forecast (same length as
        ``y_true``).
    title:
        Figure title.

    Returns
    -------
    dict[str, Any]
        A Plotly ``{data, layout}`` dict (no plotly objects).

    Raises
    ------
    ImportError
        If the ``[viz]`` extra (plotly) is not installed.
    ValidationError
        If a forecast length does not match ``y_true``.
    """
    # LAZY import: keep plotly off the import path of this pure module.
    import plotly.graph_objects as go

    actual = _coerce_1d(y_true, name="y_true")
    n_obs = int(actual.size)
    x_values = list(range(n_obs))

    fig = go.Figure()
    # The realized series is drawn first (near-black) so the forecasts overlay it.
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=actual.tolist(),
            mode="lines",
            name="actual",
            line={"color": _PALETTE[0], "width": 2},
        )
    )
    for offset, model in enumerate(_ordered_models(forecasts_by_model), start=1):
        forecast = _coerce_1d(forecasts_by_model[model], name=f"forecast[{model}]")
        if forecast.size != n_obs:
            raise ValidationError(
                f"forecast for '{model}' must align with y_true length {n_obs}, "
                f"got {forecast.size}."
            )
        colour = _PALETTE[offset % len(_PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=forecast.tolist(),
                mode="lines",
                name=model,
                line={"color": colour, "width": 1},
                opacity=0.85,
            )
        )

    fig.update_layout(
        title={"text": title},
        xaxis={"title": {"text": "out-of-sample step"}},
        yaxis={"title": {"text": "next-step return"}, "tickformat": ".2%", "zeroline": True},
        legend={"orientation": "h"},
        template="plotly_white",
    )
    return _figure_to_dict(fig)


def error_figure(
    rmse_by_model: Mapping[str, float],
    directional_acc_by_model: Mapping[str, float],
    *,
    title: str = "RMSE and directional accuracy by model",
) -> dict[str, Any]:
    """Build the per-model error bar Plotly figure as a ``{data, layout}`` dict.

    LAZY IMPORT: ``plotly`` is imported inside this function (the ``[viz]`` extra).
    Draws a grouped bar chart of return-space RMSE and directional accuracy keyed
    by model, then serializes to a plain dict.

    Parameters
    ----------
    rmse_by_model:
        Map of model name -> return-space OOS RMSE.
    directional_acc_by_model:
        Map of model name -> directional accuracy.
    title:
        Figure title.

    Returns
    -------
    dict[str, Any]
        A Plotly ``{data, layout}`` dict (no plotly objects).

    Raises
    ------
    ImportError
        If the ``[viz]`` extra (plotly) is not installed.
    ValidationError
        If the two maps do not share the same model keys.
    """
    # LAZY import: keep plotly off the import path of this pure module.
    import plotly.graph_objects as go

    if not rmse_by_model:
        raise ValidationError("error_figure: rmse_by_model must be non-empty.")
    if set(rmse_by_model) != set(directional_acc_by_model):
        raise ValidationError(
            "error_figure: rmse_by_model and directional_acc_by_model must share "
            f"the same model keys (got {sorted(rmse_by_model)} vs "
            f"{sorted(directional_acc_by_model)})."
        )

    models = list(_ordered_models(rmse_by_model, directional_acc_by_model))
    rmse_values = [_safe_float(rmse_by_model[m], name=f"rmse[{m}]") for m in models]
    acc_values = [_safe_float(directional_acc_by_model[m], name=f"dir_acc[{m}]") for m in models]

    fig = go.Figure()
    # RMSE on the primary axis (lower is better); directional accuracy on a
    # secondary axis (0.5 is the coin-flip line — the honest-null reference).
    fig.add_trace(
        go.Bar(
            x=models,
            y=rmse_values,
            name="RMSE",
            marker={"color": _PALETTE[1]},
            yaxis="y",
        )
    )
    fig.add_trace(
        go.Bar(
            x=models,
            y=acc_values,
            name="directional accuracy",
            marker={"color": _PALETTE[2]},
            yaxis="y2",
        )
    )
    fig.update_layout(
        title={"text": title},
        barmode="group",
        xaxis={"title": {"text": "model"}},
        yaxis={"title": {"text": "RMSE (return units)"}, "rangemode": "tozero"},
        yaxis2={
            "title": {"text": "directional accuracy"},
            "overlaying": "y",
            "side": "right",
            "range": [0.0, 1.0],
        },
        legend={"orientation": "h"},
        template="plotly_white",
    )
    return _figure_to_dict(fig)


def _figure_to_dict(figure: Any) -> dict[str, Any]:
    """Serialize a Plotly ``Figure`` to a plain ``{data, layout}`` dict.

    Uses ``json.loads(pio.to_json(fig, validate=False))`` so the result is a plain
    JSON-safe dict carrying no plotly objects — exactly the form the API response
    embeds. ``plotly.io`` is imported lazily inside this helper.

    Parameters
    ----------
    figure:
        A ``plotly.graph_objects.Figure``.

    Returns
    -------
    dict[str, Any]
        The ``{data, layout}`` dict.
    """
    # LAZY import: keep plotly off the import path of this pure module.
    import plotly.io as pio

    payload: dict[str, Any] = json.loads(pio.to_json(figure, validate=False))
    return payload


def _safe_float(value: Any, *, name: str) -> float:
    """Coerce a scalar to a finite ``float`` (NaN/inf rejected)."""
    out = float(value)
    if not np.isfinite(out):
        raise ValidationError(f"{name} must be finite, got {value!r}.")
    return out


def _ordered_models(*maps: Mapping[str, Any]) -> Sequence[str]:
    """Return a stable, de-duplicated model ordering across one or more maps."""
    seen: list[str] = []
    for mapping in maps:
        for key in mapping:
            if key not in seen:
                seen.append(key)
    return seen
