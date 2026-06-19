"""Plotly figure builders.

Each builder returns a plain ``dict`` shaped ``{"data": [...], "layout": {...}}``
— the same JSON shape the FastAPI layer serializes and the Next.js ``PlotlyChart``
component renders — so the figures cross the API boundary with no Plotly object
leaking through. The builders construct a Plotly graph-object ``Figure`` and
serialize it with ``json.loads(pio.to_json(fig, validate=False))``, which round-
trips every numpy scalar/array to native JSON types.

Plotly is an OPTIONAL dependency (the ``viz`` extra) and is imported lazily inside
each builder; importing this module has no side effects and does not require
Plotly.

Importing this module has no side effects.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from fedcausal._exceptions import ValidationError

if TYPE_CHECKING:
    from numpy.typing import NDArray

# quantcore-candidate: mirrors hrp-portfolio:src/hrp/plots.py ({data, layout} shape).

#: A Plotly figure serialized as a plain mapping with ``data`` and ``layout`` keys.
FigureDict = dict[str, Any]


def _figure_to_dict(fig: Any) -> FigureDict:
    """Serialize a Plotly ``Figure`` to a plain ``{"data", "layout"}`` mapping.

    Uses ``plotly.io.to_json(fig, validate=False)`` and parses it back with
    :func:`json.loads`, exactly as the brief and the FastAPI layer require, so the
    result is a nested mapping of native JSON types (no numpy scalars, no Plotly
    objects leaking across the API boundary).
    """
    import json

    import plotly.io as pio

    payload: dict[str, Any] = json.loads(pio.to_json(fig, validate=False))
    # Guarantee the two top-level keys the frontend ``PlotlyChart`` expects exist,
    # even for an empty figure.
    payload.setdefault("data", [])
    payload.setdefault("layout", {})
    return payload


def _finite_1d(values: object, *, name: str) -> NDArray[np.float64]:
    """Coerce ``values`` to a finite, 1-D float64 array (raise on bad input).

    Raises
    ------
    ValidationError
        If ``values`` is not 1-dimensional, is empty, or contains any non-finite
        (NaN/inf) entry.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValidationError(f"{name} must be 1-dimensional, got ndim={arr.ndim}.")
    if arr.size == 0:
        raise ValidationError(f"{name} must be non-empty.")
    if not bool(np.isfinite(arr).all()):
        raise ValidationError(f"{name} must be finite (no NaN/inf).")
    return arr


def _finite_scalar(value: object, *, name: str) -> float:
    """Coerce ``value`` to a finite float (raise on NaN/inf or non-numeric)."""
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be a finite number, got {value!r}.") from exc
    if not math.isfinite(out):
        raise ValidationError(f"{name} must be finite, got {value!r}.")
    return out


def car_path_figure(
    car_path: NDArray[np.float64],
    ci_lower: NDArray[np.float64],
    ci_upper: NDArray[np.float64],
    *,
    event_half_width: int = 1,
    title: str = "Cumulative abnormal return around FOMC announcements",
) -> FigureDict:
    """Build the CAR-path figure (mean CAR with a confidence band).

    Plots the cross-sectional mean cumulative abnormal return at each event-
    relative day with a shaded confidence band, centred on the announcement day
    (event-relative day 0). The band is drawn as the upper bound plus the lower
    bound filled to it, and a reference line at zero shows whether the path
    departs from no abnormal return.

    Parameters
    ----------
    car_path:
        The mean cumulative abnormal return at each event-relative day (length
        ``2k + 1``).
    ci_lower, ci_upper:
        The lower/upper confidence-band bounds at each event-relative day.
    event_half_width:
        The half-width ``k`` (so the x-axis runs ``-k .. +k``).
    title:
        Figure title.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` mapping rendering the CAR path with its band.

    Raises
    ------
    ValidationError
        If the three series are non-finite, empty, or mismatched in length, if
        ``event_half_width`` is not a positive int consistent with the series
        length, or if any lower bound exceeds its upper bound.
    """
    path = _finite_1d(car_path, name="car_path")
    lower = _finite_1d(ci_lower, name="ci_lower")
    upper = _finite_1d(ci_upper, name="ci_upper")
    if not (path.size == lower.size == upper.size):
        raise ValidationError(
            "car_path, ci_lower and ci_upper must have the same length, got "
            f"{path.size}, {lower.size}, {upper.size}."
        )
    if int(event_half_width) < 1:
        raise ValidationError(f"event_half_width must be >= 1, got {event_half_width}.")
    if path.size != 2 * int(event_half_width) + 1:
        raise ValidationError(
            f"car_path length ({path.size}) must equal 2*event_half_width + 1 "
            f"({2 * int(event_half_width) + 1})."
        )
    if bool((lower > upper).any()):
        raise ValidationError("ci_lower must not exceed ci_upper at any event-relative day.")

    k = int(event_half_width)
    rel_days = list(range(-k, k + 1))

    import plotly.graph_objects as go

    fig = go.Figure()
    # Upper band edge (invisible line; anchors the fill).
    fig.add_trace(
        go.Scatter(
            x=rel_days,
            y=upper.tolist(),
            mode="lines",
            line={"width": 0},
            name="upper",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    # Lower band edge filled up to the upper edge -> the shaded confidence band.
    fig.add_trace(
        go.Scatter(
            x=rel_days,
            y=lower.tolist(),
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(31, 119, 180, 0.2)",
            name="confidence band",
            showlegend=True,
            hoverinfo="skip",
        )
    )
    # The mean CAR path itself.
    fig.add_trace(
        go.Scatter(
            x=rel_days,
            y=path.tolist(),
            mode="lines+markers",
            line={"color": "rgb(31, 119, 180)", "width": 2},
            name="mean CAR",
        )
    )
    fig.update_layout(
        title={"text": title},
        xaxis={"title": {"text": "event-relative day"}, "zeroline": True, "dtick": 1},
        yaxis={"title": {"text": "cumulative abnormal return"}, "tickformat": ".2%"},
        legend={"orientation": "h"},
        # Reference line at zero abnormal return.
        shapes=[
            {
                "type": "line",
                "xref": "paper",
                "yref": "y",
                "x0": 0.0,
                "x1": 1.0,
                "y0": 0.0,
                "y1": 0.0,
                "line": {"color": "black", "dash": "dot", "width": 1},
            }
        ],
    )
    return _figure_to_dict(fig)


def placebo_histogram_figure(
    placebo_cars: NDArray[np.float64],
    observed_car: float,
    *,
    percentile: float | None = None,
    title: str = "Placebo-date null distribution vs. observed CAR",
) -> FigureDict:
    """Build the placebo null-distribution histogram with the observed CAR marked.

    Plots the histogram of placebo-date cross-sectional mean CARs (the honest
    null) and draws a vertical marker at the observed CAR, so the reader can SEE
    that the observed effect sits inside (not in the tail of) the placebo null.

    Parameters
    ----------
    placebo_cars:
        The placebo-date null distribution of cross-sectional mean CARs.
    observed_car:
        The observed cross-sectional mean CAR (drawn as a vertical marker).
    percentile:
        Optional observed-CAR percentile (0-100) to annotate.
    title:
        Figure title.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` mapping rendering the placebo histogram.

    Raises
    ------
    ValidationError
        If ``placebo_cars`` is non-finite/empty, ``observed_car`` is non-finite,
        or ``percentile`` (when given) is outside ``[0, 100]``.
    """
    draws = _finite_1d(placebo_cars, name="placebo_cars")
    observed = _finite_scalar(observed_car, name="observed_car")
    pct: float | None = None
    if percentile is not None:
        pct = _finite_scalar(percentile, name="percentile")
        if not 0.0 <= pct <= 100.0:
            raise ValidationError(f"percentile must lie in [0, 100], got {percentile}.")

    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=draws.tolist(),
            name="placebo null",
            opacity=0.75,
            marker={"color": "rgb(31, 119, 180)"},
        )
    )
    annotation = "observed CAR"
    if pct is not None:
        annotation = f"observed CAR (pctile {pct:.1f})"
    fig.update_layout(
        title={"text": title},
        xaxis={"title": {"text": "cross-sectional mean CAR"}, "tickformat": ".2%"},
        yaxis={"title": {"text": "count"}},
        bargap=0.02,
        legend={"orientation": "h"},
        # Vertical marker at the observed CAR so the reader sees where it falls
        # within the placebo null (mid-mass for the honest-null deliverable).
        shapes=[
            {
                "type": "line",
                "xref": "x",
                "yref": "paper",
                "x0": observed,
                "x1": observed,
                "y0": 0.0,
                "y1": 1.0,
                "line": {"color": "firebrick", "dash": "dash", "width": 2},
            }
        ],
        annotations=[
            {
                "x": observed,
                "y": 1.0,
                "xref": "x",
                "yref": "paper",
                "text": annotation,
                "showarrow": True,
                "arrowhead": 2,
                "ax": 0,
                "ay": -20,
                "font": {"color": "firebrick"},
            }
        ],
    )
    return _figure_to_dict(fig)


def did_coefficient_figure(
    coef: float,
    ci_lower: float,
    ci_upper: float,
    *,
    title: str = "Difference-in-differences coefficient (treated minus control)",
) -> FigureDict:
    """Build the DiD coefficient figure (point estimate with a clustered-SE CI).

    Plots the ``treated x post`` interaction coefficient as a point with its
    clustered confidence interval (rendered as an error bar), with a reference
    line at zero so the reader can see whether the heterogeneity is
    distinguishable from zero.

    Parameters
    ----------
    coef:
        The DiD interaction coefficient.
    ci_lower, ci_upper:
        The clustered confidence-interval bounds on ``coef``.
    title:
        Figure title.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` mapping rendering the DiD coefficient.

    Raises
    ------
    ValidationError
        If any value is non-finite or ``ci_lower > ci_upper``.
    """
    point = _finite_scalar(coef, name="coef")
    lower = _finite_scalar(ci_lower, name="ci_lower")
    upper = _finite_scalar(ci_upper, name="ci_upper")
    if lower > upper:
        raise ValidationError(f"ci_lower ({lower}) must not exceed ci_upper ({upper}).")

    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=["treated x post"],
            y=[point],
            mode="markers",
            marker={"size": 12, "color": "rgb(31, 119, 180)"},
            error_y={
                "type": "data",
                "symmetric": False,
                "array": [upper - point],
                "arrayminus": [point - lower],
                "color": "rgb(31, 119, 180)",
                "thickness": 2,
                "width": 8,
            },
            name="DiD coefficient",
        )
    )
    fig.update_layout(
        title={"text": title},
        xaxis={"title": {"text": ""}},
        yaxis={"title": {"text": "treated - control differential"}, "tickformat": ".2%"},
        showlegend=False,
        # Reference line at zero: heterogeneity is "distinguishable" only if the
        # whole CI sits on one side of it.
        shapes=[
            {
                "type": "line",
                "xref": "paper",
                "yref": "y",
                "x0": 0.0,
                "x1": 1.0,
                "y0": 0.0,
                "y1": 0.0,
                "line": {"color": "black", "dash": "dot", "width": 1},
            }
        ],
    )
    return _figure_to_dict(fig)
