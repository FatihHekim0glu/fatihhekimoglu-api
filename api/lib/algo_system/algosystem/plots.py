"""Plotly figure builders (LAZY plotly): backtest-vs-live equity overlay + drawdown.

Each builder returns a plain ``dict`` shaped ``{"data": [...], "layout": {...}}`` —
the same JSON shape the FastAPI layer serializes and the Next.js ``PlotlyChart``
component renders — so the figures cross the API boundary with no Plotly object
leaking through. Plotly is an OPTIONAL dependency (the ``viz`` extra) and is
imported lazily inside each builder; importing this module has no side effects and
does not require Plotly.

The serialization always routes through
``json.loads(plotly.io.to_json(fig, validate=False))`` so the emitted mapping is a
plain, JSON-safe ``dict`` (no numpy scalars, no Plotly classes) regardless of the
input container the caller passed. Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from algosystem._exceptions import ValidationError
from algosystem._typing import FloatArray

if TYPE_CHECKING:
    import plotly.graph_objects as go

#: A Plotly figure serialized as a plain mapping with ``data`` and ``layout`` keys.
FigureDict = dict[str, Any]


def _finite_1d(values: object, *, name: str) -> FloatArray:
    """Coerce ``values`` to a non-empty, finite 1-D float64 array (or raise).

    The single input boundary every figure builder funnels its curves through:
    flatten to 1-D, require non-emptiness, and reject any NaN/Inf so a malformed
    series never silently produces a broken chart.

    Parameters
    ----------
    values:
        A sequence / ndarray of floats (an equity curve, a drawdown series).
    name:
        Human-readable label used in the error message.

    Returns
    -------
    FloatArray
        The coerced 1-D float64 array.

    Raises
    ------
    ValidationError
        If ``values`` is empty or contains any non-finite value.
    """
    arr = np.asarray(values, dtype="float64").ravel()
    if arr.size == 0:
        raise ValidationError(f"{name} must be non-empty.")
    if not np.isfinite(arr).all():
        raise ValidationError(f"{name} contains non-finite values.")
    return arr


def _serialize(fig: go.Figure) -> FigureDict:
    """Serialize a Plotly figure to a plain ``{data, layout}`` mapping.

    Routes through ``plotly.io.to_json(fig, validate=False)`` (then
    :func:`json.loads`) so the result is a JSON-safe ``dict`` with no numpy scalars
    or Plotly objects — exactly what the FastAPI layer returns and the frontend
    ``PlotlyChart`` renders. ``validate=False`` skips Plotly's schema validation
    (the figures are constructed in-house from trusted traces).

    Parameters
    ----------
    fig:
        The constructed Plotly figure.

    Returns
    -------
    FigureDict
        A plain ``{"data": [...], "layout": {...}}`` mapping.
    """
    import json

    import plotly.io as pio

    # ``to_json`` walks the figure to a JSON string with no numpy scalars or Plotly
    # classes left in it; ``json.loads`` then yields a plain ``dict``. ``validate=False``
    # skips Plotly's schema validation (the traces are built in-house from trusted,
    # already-finite arrays). The result is exactly the ``{data, layout}`` shape the
    # FastAPI layer returns and the frontend ``PlotlyChart`` renders.
    payload: FigureDict = json.loads(pio.to_json(fig, validate=False))
    return payload


def equity_overlay_figure(
    backtest_equity: FloatArray,
    live_equity: FloatArray,
    buyhold_equity: FloatArray,
    *,
    title: str = "Backtest vs. live (paper-broker) equity",
) -> FigureDict:
    """Build the equity-overlay figure: backtest + paper-broker live + buy-hold.

    Overlays the vectorized backtest equity curve, the simulated paper-broker
    "live" equity curve, and the buy-and-hold curve. The backtest and live curves
    should COINCIDE to the eye (the parity oracle), visually proving the
    backtest<->live agreement; buy-and-hold is the bar the strategy must clear.

    Parameters
    ----------
    backtest_equity:
        The vectorized backtest cumulative-wealth curve.
    live_equity:
        The simulated paper-broker cumulative-wealth curve (should coincide with
        ``backtest_equity``).
    buyhold_equity:
        The buy-and-hold cumulative-wealth curve.
    title:
        The figure title.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` line-chart mapping.

    Raises
    ------
    ValidationError
        If the curves are empty or length-mismatched.
    """
    bt = _finite_1d(backtest_equity, name="backtest_equity")
    live = _finite_1d(live_equity, name="live_equity")
    bh = _finite_1d(buyhold_equity, name="buyhold_equity")
    if not (bt.size == live.size == bh.size):
        raise ValidationError(
            f"equity_overlay_figure: backtest ({bt.size}), live ({live.size}), and "
            f"buy-hold ({bh.size}) equity curves must have the same length."
        )

    # LAZY plotly: imported here (the ``viz`` extra) so importing this module pulls
    # in nothing heavy and has no side effects.
    import plotly.graph_objects as go

    x = list(range(bt.size))
    fig = go.Figure()
    # Backtest first (solid), then the paper-broker "live" curve as a dashed overlay
    # on top — they should COINCIDE to the eye (the parity oracle), so the dashed
    # line tracks exactly over the solid one. Buy-and-hold is the bar to clear.
    fig.add_trace(
        go.Scatter(
            x=x,
            y=bt.tolist(),
            mode="lines",
            name="Backtest",
            line={"color": "#2563eb", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=live.tolist(),
            mode="lines",
            name="Live (paper broker)",
            line={"color": "#f59e0b", "width": 2, "dash": "dash"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=bh.tolist(),
            mode="lines",
            name="Buy & hold",
            line={"color": "#94a3b8", "width": 1.5},
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Bar",
        yaxis_title="Equity (wealth index)",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
        margin={"l": 60, "r": 20, "t": 60, "b": 50},
    )
    return _serialize(fig)


def drawdown_figure(
    net_returns: FloatArray,
    *,
    title: str = "Strategy drawdown",
) -> FigureDict:
    """Build the drawdown figure from a per-bar net-return series.

    Renders the running peak-to-trough drawdown ``W_t / max_{s<=t} W_s - 1`` of the
    cumulative-wealth curve as a filled area below zero — the depth-of-pain view of
    the strategy.

    Parameters
    ----------
    net_returns:
        The per-bar net-return series.
    title:
        The figure title.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` area-chart mapping.

    Raises
    ------
    ValidationError
        If ``net_returns`` is empty or non-finite.
    """
    arr = _finite_1d(net_returns, name="net_returns")

    # LAZY plotly: imported here (the ``viz`` extra) so importing this module pulls
    # in nothing heavy and has no side effects.
    import plotly.graph_objects as go

    # Cumulative wealth W_t = prod_{s<=t}(1 + r_s), its running peak, and the
    # peak-to-trough drawdown W_t / max_{s<=t} W_s - 1 (<= 0). Mirrors the same
    # accounting as ``evaluation.metrics.max_drawdown`` so the chart and the scalar
    # agree.
    wealth = np.cumprod(1.0 + arr)
    running_peak = np.maximum.accumulate(wealth)
    drawdown = wealth / running_peak - 1.0

    x = list(range(drawdown.size))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=drawdown.tolist(),
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
            line={"color": "#dc2626", "width": 1.5},
            fillcolor="rgba(220, 38, 38, 0.2)",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Bar",
        yaxis_title="Drawdown",
        template="plotly_white",
        showlegend=False,
        margin={"l": 60, "r": 20, "t": 60, "b": 50},
    )
    return _serialize(fig)
