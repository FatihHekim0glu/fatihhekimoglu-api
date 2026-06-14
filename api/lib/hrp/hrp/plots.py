"""Plotly figure builders.

Each builder returns a plain ``dict`` shaped ``{"data": [...], "layout": {...}}``
— the same JSON shape the FastAPI layer serializes and the Next.js ``PlotlyChart``
component renders — so the figures cross the API boundary with no Plotly object
leaking through. Plotly is an OPTIONAL dependency (the ``viz`` extra) and is
imported lazily inside each builder; importing this module has no side effects and
does not require Plotly.

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from hrp._typing import MatrixLike, ReturnsLike

# quantcore-candidate: mirrors markowitz-optimizer / pairs-trading plots.py
# ({data, layout} figure shape).

#: A Plotly figure serialized as a plain mapping with ``data`` and ``layout`` keys.
FigureDict = dict[str, Any]


def _as_plain_dict(obj: Any) -> dict[str, Any]:
    """Coerce a Plotly graph-object (trace/layout) to a plain, JSON-safe ``dict``.

    Plotly graph-objects expose ``.to_plotly_json()``; the result is a nested
    mapping of plain Python / numpy types. We round-trip numpy scalars/arrays to
    native types so the figure crosses the API boundary with no Plotly object
    leaking through.
    """
    raw = obj.to_plotly_json() if hasattr(obj, "to_plotly_json") else dict(obj)
    return _jsonify(raw)


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


def dendrogram_figure(link: np.ndarray, labels: list[str]) -> FigureDict:
    """Build a dendrogram figure from a linkage matrix.

    Parameters
    ----------
    link:
        The ``(N - 1) x 4`` SciPy linkage matrix.
    labels:
        The ``N`` asset labels, in original input order.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` mapping rendering the clustering dendrogram.
    """
    # Lazy imports: keep Plotly/SciPy off this pure module's import path.
    from plotly.figure_factory import create_dendrogram

    link_arr = np.asarray(link, dtype="float64")
    labels = [str(label) for label in labels]

    # figure_factory.create_dendrogram re-clusters from points by default; feed it
    # our precomputed linkage via a custom linkagefun so the tree matches HRP's.
    fig = create_dendrogram(
        np.arange(len(labels)).reshape(-1, 1),
        labels=labels,
        linkagefun=lambda _x: link_arr,
    )
    layout = _as_plain_dict(fig.layout)
    layout.setdefault("title", {"text": "HRP dendrogram"})
    return {"data": [_as_plain_dict(trace) for trace in fig.data], "layout": layout}


def quasidiag_heatmap_figure(
    corr: MatrixLike,
    order: list[int],
    *,
    title: str = "Quasi-diagonalized correlation",
) -> FigureDict:
    """Build a correlation heatmap reordered by the quasi-diagonal leaf order.

    Used to show the before/after of quasi-diagonalization: passing the identity
    order renders the raw correlation; passing the leaf ``order`` renders the
    quasi-diagonal matrix with large correlations along the diagonal.

    Parameters
    ----------
    corr:
        An ``N x N`` correlation matrix.
    order:
        The leaf order (permutation of ``0 .. N-1``) to reorder rows/columns by.
    title:
        The figure title.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` heatmap mapping.
    """
    if isinstance(corr, pd.DataFrame):
        labels = [str(c) for c in corr.columns]
        mat = corr.to_numpy(dtype="float64")
    else:
        mat = np.asarray(corr, dtype="float64")
        labels = [str(i) for i in range(mat.shape[0])]

    perm = [int(i) for i in order]
    reordered = mat[np.ix_(perm, perm)]
    reordered_labels = [labels[i] for i in perm]

    data = [
        {
            "type": "heatmap",
            "z": _jsonify(reordered),
            "x": reordered_labels,
            "y": reordered_labels,
            "zmin": -1.0,
            "zmax": 1.0,
            "colorscale": "RdBu",
            "reversescale": True,
            "colorbar": {"title": "corr"},
        }
    ]
    layout = {
        "title": {"text": title},
        # Mirror the y-axis so the diagonal runs top-left to bottom-right.
        "yaxis": {"autorange": "reversed", "scaleanchor": "x"},
        "xaxis": {"side": "top"},
    }
    return {"data": data, "layout": layout}


def weights_figure(weights: pd.Series, *, title: str = "Portfolio weights") -> FigureDict:
    """Build a horizontal bar chart of portfolio weights.

    Parameters
    ----------
    weights:
        Portfolio weights labelled by asset.
    title:
        The figure title.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` bar-chart mapping.
    """
    series = weights if isinstance(weights, pd.Series) else pd.Series(weights)
    assets = [str(idx) for idx in series.index]
    values = [_jsonify(v) for v in np.asarray(series.to_numpy(dtype="float64"))]

    data = [
        {
            "type": "bar",
            "orientation": "h",
            "x": values,
            "y": assets,
        }
    ]
    layout = {
        "title": {"text": title},
        "xaxis": {"title": {"text": "weight"}, "tickformat": ".1%"},
        "yaxis": {"title": {"text": "asset"}, "autorange": "reversed"},
    }
    return {"data": data, "layout": layout}


def oos_equity_figure(
    equity_curves: pd.DataFrame,
    *,
    title: str = "Out-of-sample equity curves",
) -> FigureDict:
    """Build overlaid out-of-sample equity curves for each allocator.

    Parameters
    ----------
    equity_curves:
        A DataFrame of cumulative-wealth curves (rows = time, columns =
        allocator name).
    title:
        The figure title.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` line-chart mapping.
    """
    frame = equity_curves if isinstance(equity_curves, pd.DataFrame) else pd.DataFrame(equity_curves)
    x_axis = [
        v.isoformat() if hasattr(v, "isoformat") else str(v)
        for v in frame.index
    ]
    data = [
        {
            "type": "scatter",
            "mode": "lines",
            "name": str(column),
            "x": x_axis,
            "y": [_jsonify(v) for v in np.asarray(frame[column].to_numpy(dtype="float64"))],
        }
        for column in frame.columns
    ]
    layout = {
        "title": {"text": title},
        "xaxis": {"title": {"text": "date"}},
        "yaxis": {"title": {"text": "cumulative wealth"}},
        "legend": {"orientation": "h"},
    }
    return {"data": data, "layout": layout}


def sharpe_gap_bootstrap_figure(
    gap_samples: ReturnsLike,
    *,
    ci_low: float,
    ci_high: float,
    title: str = "Bootstrap Sharpe-gap distribution",
) -> FigureDict:
    """Build a histogram of bootstrap Sharpe-gap samples with CI markers.

    Renders the bootstrap distribution of the HRP-vs-1/N Sharpe gap with vertical
    markers at the confidence-interval bounds and at zero, so the reader sees
    whether the CI straddles zero.

    Parameters
    ----------
    gap_samples:
        The bootstrap Sharpe-gap samples.
    ci_low, ci_high:
        The confidence-interval bounds to mark.
    title:
        The figure title.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` histogram mapping.
    """
    if isinstance(gap_samples, pd.Series):
        samples = gap_samples.to_numpy(dtype="float64")
    elif isinstance(gap_samples, pd.DataFrame):
        samples = gap_samples.to_numpy(dtype="float64").ravel()
    else:
        samples = np.asarray(gap_samples, dtype="float64").ravel()

    data = [
        {
            "type": "histogram",
            "x": [_jsonify(v) for v in samples],
            "name": "bootstrap gap",
            "opacity": 0.75,
        }
    ]

    # Vertical markers at the CI bounds and at zero so the reader sees whether the
    # CI straddles zero (the honest null).
    def _vline(x: float, color: str, dash: str) -> dict[str, Any]:
        return {
            "type": "line",
            "xref": "x",
            "yref": "paper",
            "x0": float(x),
            "x1": float(x),
            "y0": 0.0,
            "y1": 1.0,
            "line": {"color": color, "dash": dash, "width": 2},
        }

    layout = {
        "title": {"text": title},
        "xaxis": {"title": {"text": "Sharpe gap (HRP - 1/N)"}},
        "yaxis": {"title": {"text": "count"}},
        "bargap": 0.02,
        "shapes": [
            _vline(0.0, "black", "solid"),
            _vline(ci_low, "firebrick", "dash"),
            _vline(ci_high, "firebrick", "dash"),
        ],
    }
    return {"data": data, "layout": layout}
