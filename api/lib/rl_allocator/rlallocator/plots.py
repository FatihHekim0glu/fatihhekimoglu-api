"""Plotly figure builders (LAZY plotly): equity curves, the weight-allocation area chart, seed dispersion.

Each builder returns a plain ``dict`` shaped ``{"data": [...], "layout": {...}}`` —
the same JSON shape the FastAPI layer serializes and the Next.js ``PlotlyChart``
component renders — so the figures cross the API boundary with no Plotly object
leaking through. Plotly is an OPTIONAL dependency (the ``viz`` extra) and is imported
lazily inside each builder; importing this module has no side effects and does not
require Plotly.

The serialization always routes through
``json.loads(plotly.io.to_json(fig, validate=False))`` so the emitted mapping is a
plain, JSON-safe ``dict`` (no numpy scalars, no Plotly classes) regardless of the
input container the caller passed.

Importing this module has no side effects.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from rlallocator._exceptions import ValidationError
from rlallocator._typing import FloatArray

if TYPE_CHECKING:
    import plotly.graph_objects as go

#: A Plotly figure serialized as a plain mapping with ``data`` and ``layout`` keys.
FigureDict = dict[str, Any]

#: A small, distinct colour cycle for the baseline / asset traces.
_PALETTE: tuple[str, ...] = (
    "#636efa",
    "#ef553b",
    "#00cc96",
    "#ab63fa",
    "#ffa15a",
    "#19d3f3",
    "#ff6692",
    "#b6e880",
    "#ff97ff",
    "#fecb52",
)


def _finite_1d(values: object, *, name: str) -> FloatArray:
    """Coerce ``values`` to a non-empty, finite 1-D float64 array (or raise).

    The single input boundary every figure builder funnels its curves / Sharpe
    vectors through: flatten to 1-D, require non-emptiness, and reject any NaN/Inf so
    a malformed series never silently produces a broken chart.

    Parameters
    ----------
    values:
        A sequence / ndarray of floats (an equity curve, a band edge, per-seed Sharpes).
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

    Routes through ``plotly.io.to_json(fig, validate=False)`` (then :func:`json.loads`)
    so the result is a JSON-safe ``dict`` with no numpy scalars or Plotly objects —
    exactly what the FastAPI layer returns and the frontend ``PlotlyChart`` renders.
    ``validate=False`` skips Plotly's schema validation (the figures are constructed
    in-house from trusted traces).

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
    *,
    rl_median_equity: FloatArray,
    baseline_equities: Mapping[str, Sequence[float]],
    seed_band_lo: Sequence[float] | None = None,
    seed_band_hi: Sequence[float] | None = None,
    title: str = "Out-of-sample equity curves (purged walk-forward)",
) -> FigureDict:
    """Build the OOS equity-curve figure: RL median + the baselines + the across-seed band.

    Overlays the median-seed RL equity curve and each baseline's equity curve, with an
    optional shaded band spanning the per-seed equity dispersion (``seed_band_lo`` ..
    ``seed_band_hi``) so the reader sees that the median curve is one draw from a wide
    seed lottery, not a singular result. All curves are trimmed to a common length.

    Parameters
    ----------
    rl_median_equity:
        The median-seed RL cumulative-wealth curve.
    baseline_equities:
        Mapping of baseline name -> its cumulative-wealth curve (1/N, Markowitz,
        risk-parity).
    seed_band_lo, seed_band_hi:
        Optional per-bar lower/upper envelope of the per-seed RL equity curves. Both
        must be provided together.
    title:
        The figure title.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` line-chart mapping.

    Raises
    ------
    ValidationError
        If the curves are empty or only one band edge is given.
    """
    rl = _finite_1d(rl_median_equity, name="rl_median_equity")
    if (seed_band_lo is None) != (seed_band_hi is None):
        raise ValidationError(
            "seed_band_lo and seed_band_hi must be provided together (or both omitted)."
        )

    curves: dict[str, FloatArray] = {"RL (median seed)": rl}
    for base_name, base_curve in baseline_equities.items():
        curves[base_name] = _finite_1d(base_curve, name=f"baseline[{base_name}]")
    lengths = [c.size for c in curves.values()]
    band: tuple[FloatArray, FloatArray] | None = None
    if seed_band_lo is not None and seed_band_hi is not None:
        lo = _finite_1d(seed_band_lo, name="seed_band_lo")
        hi = _finite_1d(seed_band_hi, name="seed_band_hi")
        band = (lo, hi)
        lengths.extend([lo.size, hi.size])
    n = min(lengths)

    import plotly.graph_objects as go

    x = list(range(n))
    fig = go.Figure()

    if band is not None:
        lo, hi = band
        fig.add_trace(
            go.Scatter(
                x=x,
                y=lo[:n].tolist(),
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
                y=hi[:n].tolist(),
                mode="lines",
                line={"width": 0.0, "color": "rgba(99,110,250,0.0)"},
                fill="tonexty",
                fillcolor="rgba(99,110,250,0.18)",
                name="across-seed band",
                hoverinfo="skip",
            )
        )

    for i, (trace_name, trace_curve) in enumerate(curves.items()):
        is_rl = trace_name.startswith("RL")
        fig.add_trace(
            go.Scatter(
                x=x,
                y=trace_curve[:n].tolist(),
                mode="lines",
                line={
                    "color": _PALETTE[i % len(_PALETTE)],
                    "width": 2.5 if is_rl else 2.0,
                    "dash": "solid" if is_rl else "dash",
                },
                name=trace_name,
            )
        )

    fig.update_layout(
        title={"text": title},
        xaxis={"title": {"text": "OOS bar (concatenated purged folds)"}},
        yaxis={"title": {"text": "Cumulative wealth"}},
        legend={"orientation": "h"},
        template="plotly_white",
        margin={"l": 60, "r": 20, "t": 50, "b": 50},
    )
    return _serialize(fig)


def weights_area_figure(
    weight_path: FloatArray,
    *,
    asset_names: Sequence[str] | None = None,
    title: str = "RL allocation over time",
) -> FigureDict:
    """Build the RL allocation-over-time STACKED-AREA figure.

    Renders the per-bar portfolio weight path as a stacked area chart (one band per
    asset, stacked to the unit budget), so the reader sees how the allocator tilts the
    simplex over the OOS window. Each row of ``weight_path`` should be a valid simplex.

    Parameters
    ----------
    weight_path:
        A ``(n_bars, n_assets)`` weight path.
    asset_names:
        Optional asset labels; ``None`` => ``asset_0 .. asset_{N-1}``.
    title:
        The figure title.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` stacked-area mapping.

    Raises
    ------
    ValidationError
        If ``weight_path`` is not a non-empty 2-D matrix or is non-finite.
    """
    w = np.asarray(weight_path, dtype="float64")
    if w.ndim == 1:
        w = w.reshape(-1, 1)
    if w.ndim != 2 or w.shape[0] == 0:
        raise ValidationError("weights_area_figure: weight_path must be a non-empty 2-D matrix.")
    if not bool(np.isfinite(w).all()):
        raise ValidationError("weights_area_figure: weight_path contains non-finite values.")

    n_bars, n_assets = w.shape
    names = (
        list(asset_names) if asset_names is not None else [f"asset_{i}" for i in range(n_assets)]
    )
    if len(names) != n_assets:
        raise ValidationError(
            f"weights_area_figure: asset_names has {len(names)} labels but weight_path has "
            f"{n_assets} assets."
        )

    import plotly.graph_objects as go

    x = list(range(n_bars))
    fig = go.Figure()
    for i in range(n_assets):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=w[:, i].tolist(),
                mode="lines",
                line={"width": 0.5, "color": _PALETTE[i % len(_PALETTE)]},
                stackgroup="weights",
                name=names[i],
            )
        )
    fig.update_layout(
        title={"text": title},
        xaxis={"title": {"text": "OOS bar"}},
        yaxis={"title": {"text": "Portfolio weight"}, "range": [0.0, 1.0]},
        legend={"orientation": "h"},
        template="plotly_white",
        margin={"l": 60, "r": 20, "t": 50, "b": 50},
    )
    return _serialize(fig)


def seed_lottery_figure(
    seed_sharpes: FloatArray,
    *,
    best_baseline_sharpe: float,
    title: str = "Seed-lottery OOS-Sharpe dispersion",
) -> FigureDict:
    """Build the seed-lottery dispersion figure: the OOS-Sharpe distribution across seeds.

    Renders the distribution of per-seed OOS net Sharpes (a histogram) with a vertical
    marker at the best-baseline Sharpe and at zero, so the reader sees whether the seed
    dispersion straddles zero / the baseline (the honest NULL) — the apparent skill is
    a training-path lottery when it does.

    Parameters
    ----------
    seed_sharpes:
        The per-seed OOS net Sharpe values.
    best_baseline_sharpe:
        The best baseline's OOS Sharpe to mark on the dispersion.
    title:
        The figure title.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` histogram mapping with zero / baseline markers.

    Raises
    ------
    ValidationError
        If ``seed_sharpes`` is empty / non-finite, or ``best_baseline_sharpe`` is
        non-finite.
    """
    sharpes = _finite_1d(seed_sharpes, name="seed_sharpes")
    base = float(best_baseline_sharpe)
    if not np.isfinite(base):
        raise ValidationError(f"best_baseline_sharpe must be finite, got {best_baseline_sharpe!r}.")

    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=sharpes.tolist(),
            marker={"color": "rgba(99,110,250,0.65)"},
            name="per-seed OOS Sharpe",
        )
    )
    fig.add_vline(
        x=0.0,
        line={"color": "#2a3f5f", "width": 2.0, "dash": "dot"},
        annotation={"text": "0"},
    )
    fig.add_vline(
        x=base,
        line={"color": "#ef553b", "width": 2.0, "dash": "dash"},
        annotation={"text": "best baseline"},
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
