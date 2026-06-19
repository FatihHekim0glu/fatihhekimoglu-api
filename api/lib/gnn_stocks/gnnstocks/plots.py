"""Plotly figure builders (LAZY plotly, [viz]) — serialize to ``{data, layout}``.

Two figures back the frontend tool:

- :func:`network_figure` — the learned stock-relationship graph (the per-fold
  correlation-kNN / sector adjacency) laid out as a network, with nodes COLORED BY
  SECTOR so the eye can see the message-passing structure the GNN recovers
  (DESCRIPTIVE) even though it adds no OOS ranking skill;
- :func:`ic_figure` — a grouped bar of OOS rank-IC and long-short Sharpe by model,
  so the lack of a GNN edge over the baselines is legible at a glance.

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

from gnnstocks._exceptions import ValidationError
from gnnstocks._typing import AdjacencyMatrix

# quantcore-candidate: mirrors mvts-forecast:plots.py ({data, layout} shape).

#: Stable, colour-blind-aware palette keyed by sector/trace index (cycles if exceeded).
_PALETTE: tuple[str, ...] = (
    "#2563eb",  # blue
    "#f97316",  # orange
    "#16a34a",  # green
    "#9333ea",  # purple
    "#dc2626",  # red
    "#0891b2",  # cyan
    "#ca8a04",  # amber
    "#111827",  # near-black
)


def network_figure(
    adjacency: AdjacencyMatrix,
    sector_labels: Sequence[int],
    *,
    seed: int = 7,
    title: str = "Learned stock-relationship graph (nodes colored by sector)",
) -> dict[str, Any]:
    """Build the network-graph Plotly figure as a ``{data, layout}`` dict.

    LAZY IMPORT: ``plotly`` is imported inside this function (the ``[viz]`` extra).
    Lays out the nodes (a seeded spring/force layout computed in pure numpy so no
    networkx dependency is needed), draws the adjacency edges as line segments, and
    overlays the nodes colored by sector block. Serializes to a plain dict.

    Parameters
    ----------
    adjacency:
        The ``(n_nodes, n_nodes)`` symmetric adjacency to draw.
    sector_labels:
        Per-node integer sector labels (drive the node colours).
    seed:
        Seed for the deterministic layout.
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
        If ``adjacency`` is not square or labels misalign.
    """
    # LAZY import: keep plotly off the import path of this pure module.
    import plotly.graph_objects as go

    matrix = np.asarray(adjacency, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValidationError(
            f"network_figure: adjacency must be a square 2-D matrix, got shape {matrix.shape}."
        )
    n_nodes = int(matrix.shape[0])
    if n_nodes == 0:
        raise ValidationError("network_figure: adjacency must have at least one node.")
    if not np.isfinite(matrix).all():
        raise ValidationError("network_figure: adjacency contains non-finite values.")

    labels = [int(x) for x in sector_labels]
    if len(labels) != n_nodes:
        raise ValidationError(
            f"network_figure: sector_labels has length {len(labels)} but adjacency has "
            f"{n_nodes} node(s)."
        )

    positions = _spring_layout(matrix, seed=seed)
    edge_traces = _edge_traces(go, matrix, positions)
    node_traces = _node_traces(go, positions, labels)

    fig = go.Figure(data=[*edge_traces, *node_traces])
    fig.update_layout(
        title={"text": title},
        showlegend=True,
        legend={"orientation": "h", "title": {"text": "sector"}},
        xaxis={"visible": False, "showgrid": False, "zeroline": False},
        yaxis={"visible": False, "showgrid": False, "zeroline": False, "scaleanchor": "x"},
        template="plotly_white",
        hovermode="closest",
    )
    return _figure_to_dict(fig)


def ic_figure(
    ic_by_model: Mapping[str, float],
    ls_sharpe_by_model: Mapping[str, float],
    *,
    title: str = "Rank-IC and long-short Sharpe by model",
) -> dict[str, Any]:
    """Build the per-model rank-IC / long-short-Sharpe bar Plotly figure.

    LAZY IMPORT: ``plotly`` is imported inside this function (the ``[viz]`` extra).
    Draws a grouped bar chart of OOS rank-IC and long-short decile-spread Sharpe
    keyed by model (with a zero-IC reference line — the honest-null floor), then
    serializes to a plain dict.

    Parameters
    ----------
    ic_by_model:
        Map of model name -> OOS mean rank-IC.
    ls_sharpe_by_model:
        Map of model name -> long-short decile-spread Sharpe.
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

    if set(ic_by_model) != set(ls_sharpe_by_model):
        raise ValidationError(
            "ic_figure: ic_by_model and ls_sharpe_by_model must share the same model "
            f"keys (got {sorted(ic_by_model)} vs {sorted(ls_sharpe_by_model)})."
        )

    models = list(_ordered_models(ic_by_model, ls_sharpe_by_model))
    if not models:
        raise ValidationError("ic_figure: at least one model is required.")
    ic_values = [_safe_float(ic_by_model[m], name=f"ic[{m}]") for m in models]
    sharpe_values = [_safe_float(ls_sharpe_by_model[m], name=f"ls_sharpe[{m}]") for m in models]

    fig = go.Figure()
    # Rank-IC on the primary axis (the honest-null floor is 0 — drawn as a zero
    # reference line below); the long-short decile-spread Sharpe on a secondary
    # axis so the two scales coexist without one swamping the other.
    fig.add_trace(
        go.Bar(
            x=models,
            y=ic_values,
            name="rank-IC",
            marker={"color": _PALETTE[0]},
            yaxis="y",
        )
    )
    fig.add_trace(
        go.Bar(
            x=models,
            y=sharpe_values,
            name="long-short Sharpe",
            marker={"color": _PALETTE[1]},
            yaxis="y2",
        )
    )
    # The 0-IC reference: on the leakage-free synthetic null every model hugs it,
    # so the lack of a GNN edge is legible at a glance.
    fig.add_hline(y=0.0, line={"color": _PALETTE[7], "width": 1, "dash": "dash"})
    fig.update_layout(
        title={"text": title},
        barmode="group",
        xaxis={"title": {"text": "model"}},
        yaxis={"title": {"text": "OOS mean rank-IC"}, "zeroline": True},
        yaxis2={
            "title": {"text": "long-short Sharpe"},
            "overlaying": "y",
            "side": "right",
            "zeroline": True,
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


def _spring_layout(
    adjacency: np.ndarray,
    *,
    seed: int = 7,
    iterations: int = 80,
) -> np.ndarray:
    """Compute a deterministic Fruchterman-Reingold force-directed layout (pure numpy).

    A seeded spring layout so the network figure is reproducible without a
    ``networkx`` dependency: every pair of nodes repels (inverse-distance) while
    connected pairs attract (distance-squared), iterated with a cooling step. The
    coordinates are recentred and scaled to a unit-ish box for stable rendering.

    Parameters
    ----------
    adjacency:
        The ``(n_nodes, n_nodes)`` symmetric adjacency (edge weights drive attraction).
    seed:
        Seed for the initial node placement.
    iterations:
        Number of force-relaxation iterations.

    Returns
    -------
    numpy.ndarray
        An ``(n_nodes, 2)`` array of node coordinates.
    """
    n = int(adjacency.shape[0])
    rng = np.random.default_rng(seed)
    pos: np.ndarray = rng.standard_normal((n, 2))
    if n == 1:
        return np.zeros((1, 2), dtype=np.float64)  # a single node sits at the origin.

    # Symmetrize the weights so attraction is undirected; the 0/1 (or weighted)
    # adjacency drives the spring strength between connected nodes.
    weights = 0.5 * (adjacency + adjacency.T)
    np.fill_diagonal(weights, 0.0)

    # The optimal pairwise distance (FR's ``k``) scales with the available area
    # per node; ``temperature`` cools linearly so late iterations make fine moves.
    k = 1.0 / np.sqrt(n)
    eps = 1e-9
    for step in range(iterations):
        # Pairwise displacement deltas and distances.
        delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.sqrt((delta * delta).sum(axis=-1))
        dist = np.where(dist < eps, eps, dist)
        direction = delta / dist[:, :, np.newaxis]

        # Repulsion (all pairs) pushes nodes apart; attraction (edges) pulls
        # connected nodes together. Zero the self terms.
        repulsion = (k * k) / dist
        attraction = weights * (dist / k)
        magnitude = repulsion - attraction
        np.fill_diagonal(magnitude, 0.0)

        net = (direction * magnitude[:, :, np.newaxis]).sum(axis=1)
        net_norm = np.sqrt((net * net).sum(axis=1))
        net_norm = np.where(net_norm < eps, eps, net_norm)

        temperature = 0.1 * (1.0 - step / max(1, iterations))
        capped = np.minimum(net_norm, temperature)
        pos = pos + (net / net_norm[:, np.newaxis]) * capped[:, np.newaxis]

    # Recentre and scale to a stable box so the rendered figure is layout-agnostic.
    pos = pos - pos.mean(axis=0, keepdims=True)
    span = float(np.max(np.abs(pos))) if pos.size else 1.0
    if span > eps:
        pos = pos / span
    result: np.ndarray = np.asarray(pos, dtype=np.float64)
    return result


def _edge_traces(go: Any, adjacency: np.ndarray, positions: np.ndarray) -> list[Any]:
    """Return a single Plotly line-segment trace for all upper-triangular edges."""
    rows, cols = np.where(np.triu(adjacency, k=1) != 0.0)
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for i, j in zip(rows.tolist(), cols.tolist(), strict=True):
        edge_x.extend([float(positions[i, 0]), float(positions[j, 0]), None])
        edge_y.extend([float(positions[i, 1]), float(positions[j, 1]), None])
    if not edge_x:
        return []
    return [
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"color": "rgba(120,120,120,0.4)", "width": 1},
            hoverinfo="skip",
            showlegend=False,
            name="edges",
        )
    ]


def _node_traces(go: Any, positions: np.ndarray, labels: Sequence[int]) -> list[Any]:
    """Return one Plotly marker trace per sector (so the legend is sector-keyed)."""
    label_array = np.asarray(labels, dtype=np.int64)
    traces: list[Any] = []
    for offset, sector in enumerate(sorted(set(int(x) for x in label_array))):
        mask = label_array == sector
        idx = np.where(mask)[0]
        colour = _PALETTE[offset % len(_PALETTE)]
        traces.append(
            go.Scatter(
                x=[float(positions[i, 0]) for i in idx.tolist()],
                y=[float(positions[i, 1]) for i in idx.tolist()],
                mode="markers",
                marker={"color": colour, "size": 12, "line": {"color": "white", "width": 1}},
                name=f"sector {sector}",
                text=[f"node {int(i)}" for i in idx.tolist()],
                hoverinfo="text",
            )
        )
    return traces
