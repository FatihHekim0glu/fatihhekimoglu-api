"""Plotly figure builders (lazy).

Each builder returns a plain ``dict`` shaped ``{"data": [...], "layout": {...}}`` —
the same JSON shape the FastAPI layer serializes and the Next.js ``PlotlyChart``
component renders — so the figures cross the API boundary with no Plotly object
leaking through. Plotly is an OPTIONAL dependency (the ``viz`` extra) imported
LAZILY inside each builder; importing this module has no side effects and does not
require Plotly.

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from regimehmm._typing import FloatArray

#: A Plotly figure serialized as a plain mapping with ``data`` and ``layout`` keys.
FigureDict = dict[str, Any]

# quantcore-candidate: mirrors hrp-portfolio:src/hrp/plots.py ({data, layout} shape).

#: A small categorical palette for the regime bands (calm -> turbulent). Keyed by
#: canonical state index; colours cycle if more states than entries are supplied.
_REGIME_COLORS: tuple[str, ...] = (
    "#2ca02c",  # 0 = lowest mean / calm  (green)
    "#ff7f0e",  # 1 = mid                 (orange)
    "#d62728",  # 2 = highest vol / crisis(red)
    "#9467bd",  # 3                       (purple)
    "#8c564b",  # 4                       (brown)
)


def _regime_color(state: int) -> str:
    """Return the band colour for a canonical regime index (cycles if needed)."""
    return _REGIME_COLORS[state % len(_REGIME_COLORS)]


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
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return value.isoformat() if hasattr(value, "isoformat") else str(value)
    return value


def regime_shaded_figure(
    series: pd.Series,
    states: FloatArray,
    *,
    n_states: int,
    title: str = "Regime-shaded series",
) -> FigureDict:
    r"""Build a price/return line with a FILTERED-regime background ribbon.

    Plots ``series`` (price or cumulative return) as a line, with the plot
    background shaded by the (canonical) decoded regime at each step — one band
    colour per regime — so the persistent high/low-vol regimes are visible at a
    glance. A small legend maps each band colour to its regime label.

    SIGNAL DISCIPLINE: for an HONEST out-of-sample picture, ``states`` should be the
    ONLINE-FILTER labels (:func:`regimehmm.hmm.filter.filtered_states`). Smoothed /
    Viterbi labels may be shaded for in-sample EDA but must be captioned as
    non-tradable.

    Parameters
    ----------
    series:
        The per-period series to draw (price level or cumulative return), indexed
        by date.
    states:
        The ``(n_obs,)`` canonical decoded regime labels aligned to ``series``.
    n_states:
        The number of regimes (drives the colour palette and legend).
    title:
        The figure title.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` mapping with the line trace and per-regime
        background shapes.

    Raises
    ------
    ValidationError
        If ``series`` and ``states`` are misaligned or a label is out of range.
    """
    from regimehmm._exceptions import ValidationError

    if not isinstance(series, pd.Series):
        raise ValidationError("regime_shaded_figure: series must be a pandas Series.")
    if int(n_states) < 1:
        raise ValidationError(f"regime_shaded_figure: n_states must be >= 1, got {n_states}.")

    values = series.to_numpy(dtype="float64")
    labels_arr = np.asarray(states)
    if labels_arr.ndim != 1:
        raise ValidationError(
            f"regime_shaded_figure: states must be 1-D, got ndim={labels_arr.ndim}."
        )
    if labels_arr.shape[0] != values.shape[0]:
        raise ValidationError(
            f"regime_shaded_figure: series ({values.shape[0]}) and states "
            f"({labels_arr.shape[0]}) must be the same length."
        )

    labels_int = np.round(labels_arr).astype(int)
    if not np.allclose(labels_int, labels_arr):
        raise ValidationError("regime_shaded_figure: states must be integer-valued labels.")
    if labels_int.size and (int(labels_int.min()) < 0 or int(labels_int.max()) >= int(n_states)):
        raise ValidationError(
            f"regime_shaded_figure: states contains a label outside 0..{int(n_states) - 1}."
        )

    # The x-axis is the (date) index, rendered as ISO strings so it crosses the
    # API boundary as plain JSON regardless of index dtype.
    x_axis = [v.isoformat() if hasattr(v, "isoformat") else str(v) for v in series.index.tolist()]
    n_obs = values.shape[0]

    # The price/return line.
    data: list[dict[str, Any]] = [
        {
            "type": "scatter",
            "mode": "lines",
            "x": list(x_axis),
            "y": [_jsonify(v) for v in values],
            "name": str(series.name) if series.name is not None else "series",
            "line": {"color": "#222222", "width": 1.4},
            "hovertemplate": "%{x}<br>%{y:.4f}<extra></extra>",
        }
    ]

    # Background ribbon: one rectangle per maximal run of a single regime label,
    # spanning the full vertical extent (``yref="paper"``). Walking contiguous runs
    # keeps the shape count small and the bands visually clean.
    shapes: list[dict[str, Any]] = []
    run_start = 0
    for t in range(1, n_obs + 1):
        if t == n_obs or labels_int[t] != labels_int[run_start]:
            state = int(labels_int[run_start])
            # Half-step padding at the run edges so adjacent bands abut cleanly.
            x0 = x_axis[run_start]
            x1 = x_axis[min(t, n_obs - 1)]
            shapes.append(
                {
                    "type": "rect",
                    "xref": "x",
                    "yref": "paper",
                    "x0": x0,
                    "x1": x1,
                    "y0": 0.0,
                    "y1": 1.0,
                    "fillcolor": _regime_color(state),
                    "opacity": 0.16,
                    "line": {"width": 0},
                    "layer": "below",
                }
            )
            run_start = t

    # Invisible marker traces purely to drive a per-regime legend (one entry per
    # canonical state, in order).
    for state in range(int(n_states)):
        data.append(
            {
                "type": "scatter",
                "mode": "markers",
                "x": [x_axis[0]] if x_axis else [],
                "y": [_jsonify(values[0])] if n_obs else [],
                "name": f"regime {state}",
                "marker": {"size": 10, "color": _regime_color(state), "symbol": "square"},
                "opacity": 0.0,
                "hoverinfo": "skip",
                "showlegend": True,
            }
        )

    layout: dict[str, Any] = {
        "title": {"text": str(title)},
        "xaxis": {"title": {"text": "date"}},
        "yaxis": {"title": {"text": "level"}},
        "shapes": shapes,
        "legend": {"title": {"text": "filtered regime"}},
        "hovermode": "x unified",
    }
    return {"data": data, "layout": layout}


def regime_stats_figure(
    means: FloatArray,
    vols: FloatArray,
    *,
    title: str = "Per-regime risk/return",
) -> FigureDict:
    """Build a per-regime mean-vs-volatility bar/scatter summary.

    Renders each regime's annualized mean return against its annualized volatility
    so the reader sees the characterization headline: low-vol regimes cluster at
    modest positive mean, high-vol regimes at lower/negative mean and much higher
    risk.

    Parameters
    ----------
    means:
        The ``(n_states,)`` per-regime annualized mean returns (canonical order).
    vols:
        The ``(n_states,)`` per-regime annualized volatilities (canonical order).
    title:
        The figure title.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` mapping.

    Raises
    ------
    ValidationError
        If ``means`` and ``vols`` have different lengths.
    """
    from regimehmm._exceptions import ValidationError

    means_arr = np.asarray(means, dtype="float64").ravel()
    vols_arr = np.asarray(vols, dtype="float64").ravel()
    if means_arr.shape[0] != vols_arr.shape[0]:
        raise ValidationError(
            f"regime_stats_figure: means ({means_arr.shape[0]}) and vols "
            f"({vols_arr.shape[0]}) must have the same length."
        )

    n_states = means_arr.shape[0]
    labels = [f"regime {k}" for k in range(n_states)]
    colors = [_regime_color(k) for k in range(n_states)]

    # A risk/return scatter: annualized volatility (x) vs annualized mean (y), one
    # marker per regime, coloured by canonical index. This shows the headline at a
    # glance — low-vol regimes cluster at modest positive mean, high-vol regimes at
    # lower mean and far higher risk.
    data: list[dict[str, Any]] = [
        {
            "type": "scatter",
            "mode": "markers+text",
            "x": [_jsonify(v) for v in vols_arr],
            "y": [_jsonify(m) for m in means_arr],
            "text": labels,
            "textposition": "top center",
            "marker": {
                "size": 16,
                "color": colors,
                "line": {"width": 1, "color": "#333333"},
            },
            "name": "regimes",
            "hovertemplate": ("%{text}<br>vol=%{x:.3f}<br>mean=%{y:.3f}<extra></extra>"),
        }
    ]

    layout: dict[str, Any] = {
        "title": {"text": str(title)},
        "xaxis": {"title": {"text": "annualized volatility"}, "tickformat": ".1%"},
        "yaxis": {"title": {"text": "annualized mean return"}, "tickformat": ".1%"},
        # A zero-mean reference line so negative-mean (risk-off) regimes are obvious.
        "shapes": [
            {
                "type": "line",
                "xref": "paper",
                "yref": "y",
                "x0": 0.0,
                "x1": 1.0,
                "y0": 0.0,
                "y1": 0.0,
                "line": {"color": "#999999", "width": 1, "dash": "dot"},
                "layer": "below",
            }
        ],
        "showlegend": False,
    }
    return {"data": data, "layout": layout}


def oos_equity_figure(
    overlay_returns: pd.Series,
    buyhold_returns: pd.Series,
    *,
    title: str = "OOS equity: regime overlay vs buy-and-hold",
) -> FigureDict:
    r"""Build the out-of-sample equity curve: overlay vs buy-and-hold.

    Compounds each net OOS return series into a wealth curve and overlays the two
    so the reader sees the honest headline directly: the regime-timing overlay does
    NOT reliably separate from buy-and-hold after costs. Both curves start at 1.0.

    Parameters
    ----------
    overlay_returns:
        The net (after-cost) OOS overlay return series.
    buyhold_returns:
        The buy-and-hold benchmark return series over the same OOS window.
    title:
        The figure title.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` mapping with the two equity traces.

    Raises
    ------
    ValidationError
        If the two series cannot be aligned.
    """
    from regimehmm._exceptions import ValidationError

    if not isinstance(overlay_returns, pd.Series):
        raise ValidationError("oos_equity_figure: overlay_returns must be a pandas Series.")
    if not isinstance(buyhold_returns, pd.Series):
        raise ValidationError("oos_equity_figure: buyhold_returns must be a pandas Series.")

    # Align on the intersection of the two indexes (no-lookahead-safe) and require a
    # non-empty overlap so the two wealth curves are directly comparable.
    common = overlay_returns.index.intersection(buyhold_returns.index)
    if len(common) == 0:
        raise ValidationError(
            "oos_equity_figure: overlay and buy-and-hold series share no common index."
        )
    common = common.sort_values()
    overlay = overlay_returns.reindex(common).astype("float64")
    buyhold = buyhold_returns.reindex(common).astype("float64")

    def _equity(returns: pd.Series) -> FloatArray:
        # Compound net returns into a wealth curve starting at 1.0. NaNs are
        # treated as zero return (a flat step) so a single gap never voids the
        # whole curve.
        rets = returns.to_numpy(dtype="float64")
        rets = np.where(np.isfinite(rets), rets, 0.0)
        return np.asarray(np.cumprod(1.0 + rets), dtype="float64")

    overlay_curve = _equity(overlay)
    buyhold_curve = _equity(buyhold)

    x_axis = [v.isoformat() if hasattr(v, "isoformat") else str(v) for v in common.tolist()]

    data: list[dict[str, Any]] = [
        {
            "type": "scatter",
            "mode": "lines",
            "x": list(x_axis),
            "y": [_jsonify(v) for v in overlay_curve],
            "name": "regime overlay",
            "line": {"color": "#1f77b4", "width": 1.6},
            "hovertemplate": "%{x}<br>overlay=%{y:.4f}<extra></extra>",
        },
        {
            "type": "scatter",
            "mode": "lines",
            "x": list(x_axis),
            "y": [_jsonify(v) for v in buyhold_curve],
            "name": "buy & hold",
            "line": {"color": "#7f7f7f", "width": 1.6, "dash": "dash"},
            "hovertemplate": "%{x}<br>buy&hold=%{y:.4f}<extra></extra>",
        },
    ]

    layout: dict[str, Any] = {
        "title": {"text": str(title)},
        "xaxis": {"title": {"text": "date"}},
        "yaxis": {"title": {"text": "growth of 1.0"}},
        "legend": {"title": {"text": "strategy"}},
        "hovermode": "x unified",
    }
    return {"data": data, "layout": layout}
