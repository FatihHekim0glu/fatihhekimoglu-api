"""Plotly figure builders for the clustering tool.

Each builder returns a plain ``dict`` shaped ``{"data": [...], "layout": {...}}``
— the same JSON shape the FastAPI layer serializes and the Next.js ``PlotlyChart``
component renders — so the figures cross the API boundary with no Plotly object
leaking through. Plotly is an OPTIONAL dependency (the ``viz`` extra) and is
imported LAZILY inside each builder; importing this module has no side effects and
does not require Plotly.

Importing this module has no side effects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from stockclusters._typing import MatrixLike, ReturnsLike

#: A Plotly figure serialized as a plain mapping with ``data`` and ``layout`` keys.
FigureDict = dict[str, Any]

#: A qualitative palette for colouring clusters (cycled if there are more clusters).
_CLUSTER_PALETTE: tuple[str, ...] = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)

__all__ = [
    "FigureDict",
    "cluster_heatmap_figure",
    "dendrogram_figure",
    "embedding_scatter_figure",
    "mst_network_figure",
    "oos_equity_figure",
    "stability_figure",
]


def _jsonify(value: Any) -> Any:
    """Recursively convert numpy/pandas scalars and arrays to native Python types.

    Guarantees the returned structure contains only JSON-safe leaves (no numpy
    scalars/arrays, no pandas timestamps), and maps non-finite floats to ``None``
    so the API never emits ``NaN``/``Infinity`` (invalid JSON).
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
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _as_plain_dict(obj: Any) -> dict[str, Any]:
    """Coerce a Plotly graph-object (trace/layout) to a plain, JSON-safe ``dict``."""
    raw = obj.to_plotly_json() if hasattr(obj, "to_plotly_json") else dict(obj)
    plain: dict[str, Any] = _jsonify(raw)
    return plain


def _cluster_color(cluster_id: int) -> str:
    """Stable colour for a cluster id (cycled through the palette)."""
    return _CLUSTER_PALETTE[int(cluster_id) % len(_CLUSTER_PALETTE)]


def _aligned_label_array(labels: pd.Series, order: Sequence[str]) -> np.ndarray:
    """Return the integer cluster labels in ``order`` (by ticker), as an array."""
    return np.asarray([int(labels.loc[a]) for a in order], dtype=int)


def cluster_heatmap_figure(
    corr: MatrixLike,
    ordered_assets: Sequence[str],
    labels: pd.Series,
) -> FigureDict:
    """Cluster-ordered correlation heatmap.

    Reorders the correlation matrix by dendrogram leaf order so blocks of
    high-correlation assets appear on the diagonal, annotated by cluster.

    Parameters
    ----------
    corr:
        The ``N x N`` correlation matrix, labelled by asset.
    ordered_assets:
        Asset tickers in dendrogram-leaf order.
    labels:
        Integer cluster labels indexed by asset.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` mapping (lazy Plotly import).
    """
    order = [str(a) for a in ordered_assets]
    if isinstance(corr, pd.DataFrame):
        frame = corr.copy()
        frame.index = pd.Index([str(i) for i in frame.index])
        frame.columns = pd.Index([str(c) for c in frame.columns])
        reordered = frame.reindex(index=order, columns=order)
        z = reordered.to_numpy(dtype="float64")
    else:
        mat = np.asarray(corr, dtype="float64")
        base = [str(i) for i in range(mat.shape[0])]
        pos = {a: i for i, a in enumerate(base)}
        perm = [pos[a] for a in order]
        z = mat[np.ix_(perm, perm)]

    cluster_seq = _aligned_label_array(labels, order)
    data = [
        {
            "type": "heatmap",
            "z": _jsonify(z),
            "x": order,
            "y": order,
            "zmin": -1.0,
            "zmax": 1.0,
            "colorscale": "RdBu",
            "reversescale": True,
            "colorbar": {"title": {"text": "corr"}},
        }
    ]
    # Draw cluster-block boundaries as rectangles along the diagonal so the reader
    # sees where one cluster ends and the next begins.
    shapes: list[dict[str, Any]] = []
    start = 0
    n = len(order)
    for i in range(1, n + 1):
        if i == n or cluster_seq[i] != cluster_seq[start]:
            shapes.append(
                {
                    "type": "rect",
                    "xref": "x",
                    "yref": "y",
                    "x0": start - 0.5,
                    "x1": i - 0.5,
                    "y0": start - 0.5,
                    "y1": i - 0.5,
                    "line": {"color": "black", "width": 1.5},
                    "fillcolor": "rgba(0,0,0,0)",
                }
            )
            start = i
    layout = {
        "title": {"text": "Cluster-ordered correlation heatmap"},
        "yaxis": {"autorange": "reversed", "scaleanchor": "x"},
        "xaxis": {"side": "top"},
        "shapes": shapes,
    }
    return {"data": data, "layout": layout}


def dendrogram_figure(linkage: np.ndarray, labels: Sequence[str]) -> FigureDict:
    """Dendrogram of the agglomerative clustering.

    Parameters
    ----------
    linkage:
        The ``(N - 1) x 4`` SciPy linkage matrix.
    labels:
        Asset tickers in the original (pre-linkage) order.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` mapping (lazy Plotly import).
    """
    # Lazy imports: keep Plotly/SciPy off this pure module's import path.
    from plotly.figure_factory import create_dendrogram

    link_arr = np.asarray(linkage, dtype="float64")
    leaf_labels = [str(label) for label in labels]

    # create_dendrogram re-clusters from points by default; feed it our precomputed
    # linkage via a custom linkagefun so the tree matches the clustering exactly.
    fig = create_dendrogram(
        np.arange(len(leaf_labels)).reshape(-1, 1),
        labels=leaf_labels,
        linkagefun=lambda _x: link_arr,
    )
    layout = _as_plain_dict(fig.layout)
    layout.setdefault("title", {"text": "Cluster dendrogram"})
    return {"data": [_as_plain_dict(trace) for trace in fig.data], "layout": layout}


def mst_network_figure(
    edges: pd.DataFrame,
    labels: pd.Series,
) -> FigureDict:
    """Minimum-spanning-tree network of the correlation graph.

    Parameters
    ----------
    edges:
        The MST edge list (columns ``["source", "target", "weight"]``).
    labels:
        Integer cluster labels indexed by asset (colours the nodes).

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` mapping (lazy Plotly import).
    """
    # Lazy import: NetworkX-free spring layout via a small Fruchterman-Reingold
    # implementation would be heavy; instead use a deterministic circular layout
    # grouped by cluster so the figure is reproducible and dependency-light.
    edge_frame = edges if isinstance(edges, pd.DataFrame) else pd.DataFrame(edges)
    nodes = [str(n) for n in labels.index]
    label_map = {str(k): int(v) for k, v in labels.items()}

    # Order nodes by cluster, then place them on a circle so same-cluster nodes are
    # adjacent (deterministic, no random seed).
    ordered = sorted(nodes, key=lambda a: (label_map.get(a, -1), a))
    n = max(len(ordered), 1)
    angles = np.linspace(0.0, 2.0 * np.pi, num=n, endpoint=False)
    pos = {a: (float(np.cos(t)), float(np.sin(t))) for a, t in zip(ordered, angles, strict=False)}

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for _, row in edge_frame.iterrows():
        src, tgt = str(row["source"]), str(row["target"])
        if src in pos and tgt in pos:
            x0, y0 = pos[src]
            x1, y1 = pos[tgt]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    edge_trace = {
        "type": "scatter",
        "mode": "lines",
        "x": _jsonify(edge_x),
        "y": _jsonify(edge_y),
        "line": {"color": "#888", "width": 1},
        "hoverinfo": "none",
        "name": "MST edges",
        "showlegend": False,
    }
    node_trace = {
        "type": "scatter",
        "mode": "markers+text",
        "x": [pos[a][0] for a in ordered],
        "y": [pos[a][1] for a in ordered],
        "text": ordered,
        "textposition": "top center",
        "marker": {
            "size": 12,
            "color": [_cluster_color(label_map.get(a, 0)) for a in ordered],
            "line": {"color": "#222", "width": 1},
        },
        "hovertext": [f"{a} (cluster {label_map.get(a, 0)})" for a in ordered],
        "hoverinfo": "text",
        "name": "assets",
        "showlegend": False,
    }
    layout = {
        "title": {"text": "Correlation MST (Mantegna distance)"},
        "xaxis": {"showgrid": False, "zeroline": False, "showticklabels": False},
        "yaxis": {
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
            "scaleanchor": "x",
        },
    }
    return {"data": [edge_trace, node_trace], "layout": layout}


def embedding_scatter_figure(
    embedding: MatrixLike,
    labels: pd.Series,
) -> FigureDict:
    """2-D scatter of the RMT-signal embedding, coloured by cluster.

    Parameters
    ----------
    embedding:
        An ``N x d`` embedding (the first two components are plotted).
    labels:
        Integer cluster labels indexed by asset.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` mapping (lazy Plotly import).
    """
    if isinstance(embedding, pd.DataFrame):
        assets = [str(i) for i in embedding.index]
        emb = embedding.to_numpy(dtype="float64")
    else:
        emb = np.asarray(embedding, dtype="float64")
        assets = [str(i) for i in range(emb.shape[0])]

    if emb.ndim != 2 or emb.shape[1] == 0:
        emb = emb.reshape(emb.shape[0], -1) if emb.size else emb.reshape(0, 0)

    # First component on x; second on y (or a zero column when only one component).
    comp1 = emb[:, 0] if emb.shape[1] >= 1 else np.zeros(emb.shape[0])
    comp2 = emb[:, 1] if emb.shape[1] >= 2 else np.zeros(emb.shape[0])

    label_map = {str(k): int(v) for k, v in labels.items()}
    cluster_ids = [label_map.get(a, 0) for a in assets]

    # One trace per cluster so the legend names each community.
    data: list[dict[str, Any]] = []
    for cid in sorted(set(cluster_ids)):
        mask = [c == cid for c in cluster_ids]
        data.append(
            {
                "type": "scatter",
                "mode": "markers+text",
                "name": f"cluster {cid}",
                "x": [float(comp1[i]) for i, m in enumerate(mask) if m],
                "y": [float(comp2[i]) for i, m in enumerate(mask) if m],
                "text": [assets[i] for i, m in enumerate(mask) if m],
                "textposition": "top center",
                "marker": {"size": 10, "color": _cluster_color(cid)},
            }
        )
    layout = {
        "title": {"text": "RMT-signal embedding (first two components)"},
        "xaxis": {"title": {"text": "component 1"}},
        "yaxis": {"title": {"text": "component 2"}},
        "legend": {"orientation": "h"},
    }
    return {"data": data, "layout": layout}


def oos_equity_figure(
    equity_curves: Mapping[str, pd.Series | ReturnsLike],
) -> FigureDict:
    """Out-of-sample equity curves for each allocation strategy.

    Parameters
    ----------
    equity_curves:
        A mapping ``{strategy_name: oos_return_series}`` (1/N, cluster-EW,
        stripped-HRP). Each value is a per-strategy per-period OOS return series
        (or 1-D array); the cumulative-wealth curve is formed inside the builder.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` mapping (lazy Plotly import).
    """
    data: list[dict[str, Any]] = []
    for name, series in equity_curves.items():
        if isinstance(series, pd.Series):
            returns = series.astype("float64")
            x_index = returns.index
        else:
            arr = np.asarray(series, dtype="float64").ravel()
            returns = pd.Series(arr)
            x_index = returns.index
        # Cumulative wealth from per-period returns: prod(1 + r).
        wealth = (1.0 + returns.fillna(0.0)).cumprod()
        x_axis = [v.isoformat() if hasattr(v, "isoformat") else str(v) for v in x_index]
        data.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": str(name),
                "x": x_axis,
                "y": [_jsonify(v) for v in wealth.to_numpy(dtype="float64")],
            }
        )
    layout = {
        "title": {"text": "Out-of-sample equity curves"},
        "xaxis": {"title": {"text": "date"}},
        "yaxis": {"title": {"text": "cumulative wealth"}},
        "legend": {"orientation": "h"},
    }
    return {"data": data, "layout": layout}


def stability_figure(
    ari_series: Sequence[float],
    window_dates: Sequence[str],
) -> FigureDict:
    """Adjacent-window ARI over time (cluster-stability chart).

    Parameters
    ----------
    ari_series:
        The per-adjacent-pair ARI values.
    window_dates:
        The (end) date of each window pair, parallel to ``ari_series``.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` mapping (lazy Plotly import).
    """
    y = [_jsonify(float(v)) for v in ari_series]
    x = [str(d) for d in window_dates]
    data = [
        {
            "type": "scatter",
            "mode": "lines+markers",
            "name": "adjacent-window ARI",
            "x": x,
            "y": y,
        }
    ]
    layout = {
        "title": {"text": "Cluster stability (adjacent-window ARI)"},
        "xaxis": {"title": {"text": "window end date"}},
        "yaxis": {"title": {"text": "ARI"}, "range": [-0.1, 1.05]},
        "legend": {"orientation": "h"},
    }
    return {"data": data, "layout": layout}
