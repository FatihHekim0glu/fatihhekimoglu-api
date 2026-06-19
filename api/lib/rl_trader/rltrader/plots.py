"""Plotly figure builders (LAZY plotly): equity curves + the seed-lottery dispersion.

Each builder returns a plain ``dict`` shaped ``{"data": [...], "layout": {...}}``
— the same JSON shape the FastAPI layer serializes and the Next.js ``PlotlyChart``
component renders — so the figures cross the API boundary with no Plotly object
leaking through. Plotly is an OPTIONAL dependency (the ``viz`` extra) and is
imported lazily inside each builder; importing this module has no side effects and
does not require Plotly.

The serialization always routes through
``json.loads(plotly.io.to_json(fig, validate=False))`` so the emitted mapping is a
plain, JSON-safe ``dict`` (no numpy scalars, no Plotly classes) regardless of the
input container the caller passed.

Importing this module has no side effects.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

from rltrader._exceptions import ValidationError
from rltrader._typing import FloatArray

if TYPE_CHECKING:
    import plotly.graph_objects as go

#: A Plotly figure serialized as a plain mapping with ``data`` and ``layout`` keys.
FigureDict = dict[str, Any]


def _finite_1d(values: object, *, name: str) -> FloatArray:
    """Coerce ``values`` to a non-empty, finite 1-D float64 array (or raise).

    The single input boundary every figure builder funnels its curves / Sharpe
    vectors through: flatten to 1-D, require non-emptiness, and reject any
    NaN/Inf so a malformed series never silently produces a broken chart.

    Parameters
    ----------
    values:
        A sequence / ndarray of floats (an equity curve, a band edge, the
        per-seed Sharpes).
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
    :func:`json.loads`) so the result is a JSON-safe ``dict`` with no numpy
    scalars or Plotly objects — exactly what the FastAPI layer returns and the
    frontend ``PlotlyChart`` renders. ``validate=False`` skips Plotly's schema
    validation (the figures are constructed in-house from trusted traces).

    Parameters
    ----------
    fig:
        The constructed Plotly figure.

    Returns
    -------
    FigureDict
        A plain ``{"data": [...], "layout": {...}}`` mapping.
    """
    import plotly.io as pio

    payload: FigureDict = json.loads(pio.to_json(fig, validate=False))
    return payload


def equity_curve_figure(
    rl_median_equity: FloatArray,
    buyhold_equity: FloatArray,
    *,
    seed_band_lo: FloatArray | None = None,
    seed_band_hi: FloatArray | None = None,
    title: str = "Out-of-sample equity curves",
) -> FigureDict:
    """Build the OOS equity-curve figure: RL median + buy-hold + the across-seed band.

    Overlays the median-seed RL equity curve and the buy-and-hold equity curve, with
    an optional shaded band spanning the per-seed equity dispersion (``seed_band_lo``
    .. ``seed_band_hi``) so the reader sees that the median curve is one draw from a
    wide seed lottery, not a singular result.

    The band is rendered as two traces: an invisible lower edge plus an upper edge
    that fills down to it (``fill="tonexty"``), so the shaded region honestly shows
    the per-seed spread behind the median line.

    Parameters
    ----------
    rl_median_equity:
        The median-seed RL cumulative-wealth curve.
    buyhold_equity:
        The buy-and-hold cumulative-wealth curve.
    seed_band_lo, seed_band_hi:
        Optional per-bar lower/upper envelope of the per-seed RL equity curves.
        Both must be provided together and match the RL curve length.
    title:
        The figure title.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` line-chart mapping.

    Raises
    ------
    ValidationError
        If the curves are empty, length-mismatched, or only one band edge is given.
    """
    rl = _finite_1d(rl_median_equity, name="rl_median_equity")
    bh = _finite_1d(buyhold_equity, name="buyhold_equity")
    if rl.size != bh.size:
        raise ValidationError(
            f"rl_median_equity (len {rl.size}) and buyhold_equity (len {bh.size}) "
            "must have the same length."
        )

    if (seed_band_lo is None) != (seed_band_hi is None):
        raise ValidationError(
            "seed_band_lo and seed_band_hi must be provided together (or both omitted)."
        )

    band: tuple[FloatArray, FloatArray] | None = None
    if seed_band_lo is not None and seed_band_hi is not None:
        lo = _finite_1d(seed_band_lo, name="seed_band_lo")
        hi = _finite_1d(seed_band_hi, name="seed_band_hi")
        if lo.size != rl.size or hi.size != rl.size:
            raise ValidationError(
                f"seed_band_lo (len {lo.size}) and seed_band_hi (len {hi.size}) must "
                f"match the RL equity curve length ({rl.size})."
            )
        band = (lo, hi)

    import plotly.graph_objects as go

    x = list(range(rl.size))
    fig = go.Figure()

    if band is not None:
        lo, hi = band
        # The band is two traces: an invisible lower edge, then an upper edge that
        # fills DOWN to the previous (lower) trace, shading the per-seed spread.
        fig.add_trace(
            go.Scatter(
                x=x,
                y=lo.tolist(),
                mode="lines",
                line={"width": 0.0, "color": "rgba(99,110,250,0.0)"},
                name="seed band (lo)",
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=hi.tolist(),
                mode="lines",
                line={"width": 0.0, "color": "rgba(99,110,250,0.0)"},
                fill="tonexty",
                fillcolor="rgba(99,110,250,0.18)",
                name="across-seed band",
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=rl.tolist(),
            mode="lines",
            line={"color": "#636efa", "width": 2.0},
            name="RL (median seed)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=bh.tolist(),
            mode="lines",
            line={"color": "#ef553b", "width": 2.0, "dash": "dash"},
            name="Buy & hold",
        )
    )

    fig.update_layout(
        title={"text": title},
        xaxis={"title": {"text": "OOS bar"}},
        yaxis={"title": {"text": "Cumulative wealth"}},
        legend={"orientation": "h"},
        template="plotly_white",
        margin={"l": 60, "r": 20, "t": 50, "b": 50},
    )
    return _serialize(fig)


def seed_lottery_figure(
    seed_sharpes: FloatArray,
    *,
    buyhold_sharpe: float,
    title: str = "Seed-lottery OOS-Sharpe dispersion",
) -> FigureDict:
    """Build the seed-lottery dispersion figure: the OOS-Sharpe distribution across seeds.

    Renders the distribution of per-seed OOS net Sharpes (a histogram) with a
    vertical marker at the buy-and-hold Sharpe and at zero, so the reader sees
    whether the seed dispersion straddles zero (the honest NULL) — the apparent
    skill is a training-path lottery when it does.

    Parameters
    ----------
    seed_sharpes:
        The per-seed OOS net Sharpe values.
    buyhold_sharpe:
        The buy-and-hold OOS Sharpe to mark on the dispersion.
    title:
        The figure title.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` histogram mapping with zero / buy-hold markers.

    Raises
    ------
    ValidationError
        If ``seed_sharpes`` is empty or non-finite, or ``buyhold_sharpe`` is
        non-finite.
    """
    sharpes = _finite_1d(seed_sharpes, name="seed_sharpes")
    bh = float(buyhold_sharpe)
    if not np.isfinite(bh):
        raise ValidationError(f"buyhold_sharpe must be finite, got {buyhold_sharpe!r}.")

    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=sharpes.tolist(),
            marker={"color": "rgba(99,110,250,0.65)"},
            name="per-seed OOS Sharpe",
        )
    )

    # Zero marker (the honest-NULL reference): a Sharpe of zero is no edge.
    fig.add_vline(
        x=0.0,
        line={"color": "#2a3f5f", "width": 2.0, "dash": "dot"},
        annotation={"text": "0"},
    )
    # Buy-and-hold marker: the bar the RL agent must clear net of costs.
    fig.add_vline(
        x=bh,
        line={"color": "#ef553b", "width": 2.0, "dash": "dash"},
        annotation={"text": "buy & hold"},
    )

    fig.update_layout(
        title={"text": title},
        xaxis={"title": {"text": "OOS net Sharpe"}},
        yaxis={"title": {"text": "Seed count"}},
        template="plotly_white",
        bargap=0.05,
        margin={"l": 60, "r": 20, "t": 50, "b": 50},
        showlegend=False,
    )
    return _serialize(fig)
