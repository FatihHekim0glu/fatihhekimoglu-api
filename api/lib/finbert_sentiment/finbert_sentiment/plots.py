"""Plotly figure builders (no Plotly dependency at import; ``viz`` extra renders).

Each builder returns a plain ``dict`` shaped ``{"data": [...], "layout": {...}}``
— the same Plotly-schema JSON the FastAPI layer serializes and the Next.js
``PlotlyChart`` component renders — built directly from native Python types, so no
Plotly object ever crosses the API boundary and the builders do not import Plotly
at all. Plotly/kaleido (the OPTIONAL ``viz`` extra) are only needed downstream to
*render* these dicts to an image; importing this module has no side effects.

The two figures back the honest story: the test-set confusion matrix (where the
neutral-heavy errors live) and the per-class F1 bar (which class the macro-F1
average is being dragged down by).

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from finbert_sentiment._constants import LABELS, N_CLASSES
from finbert_sentiment._exceptions import ValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

# quantcore-candidate: mirrors hrp / lstm-forecast plots.py ({data, layout} shape).

#: A Plotly figure serialized as a plain mapping with ``data`` and ``layout`` keys.
FigureDict = dict[str, Any]


def _to_native(value: Any) -> Any:
    """Recursively coerce numpy scalars/arrays to native Python types.

    Keeps the returned figure JSON-safe — no numpy object leaks across the API
    boundary — without importing Plotly (the builders never construct a Plotly
    graph-object; they emit the schema dict directly).
    """
    if isinstance(value, np.ndarray):
        return [_to_native(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_native(v) for v in value]
    return value


def confusion_matrix_figure(
    confusion: Sequence[Sequence[int]] | NDArray[np.int64],
    *,
    normalize: bool = False,
    title: str = "Confusion matrix (test set)",
) -> FigureDict:
    """Build a confusion-matrix heatmap figure over the 3-way label space.

    The heatmap axes are the canonical
    :data:`finbert_sentiment._constants.LABELS` (rows=true, cols=pred). When
    ``normalize`` is set, each row is divided by its sum (row-normalized recall
    view); otherwise raw counts are shown.

    Parameters
    ----------
    confusion:
        A ``N_CLASSES x N_CLASSES`` matrix of counts (rows=true, cols=pred).
    normalize:
        Whether to row-normalize the displayed values.
    title:
        Figure title.

    Returns
    -------
    FigureDict
        A ``{data, layout}`` mapping (a single ``heatmap`` trace).

    Raises
    ------
    ValidationError
        If ``confusion`` is not a square ``N_CLASSES``-sized matrix.
    """
    mat = np.asarray(confusion, dtype=np.float64)
    if mat.ndim != 2 or mat.shape != (N_CLASSES, N_CLASSES):
        raise ValidationError(
            f"confusion must be a {N_CLASSES}x{N_CLASSES} matrix, got shape {mat.shape}."
        )
    if bool(np.isnan(mat).any()):
        raise ValidationError("confusion contains NaN values.")
    if bool((mat < 0.0).any()):
        raise ValidationError("confusion contains negative entries.")

    if normalize:
        # Row-normalize to the recall view; an all-zero (never-true) row stays 0.
        row_sums = mat.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            display = np.where(row_sums > 0.0, mat / row_sums, 0.0)
        text_fmt = ".2f"
        colorbar_title = "recall"
    else:
        # Integer counts render cleanest as ints.
        display = mat
        text_fmt = "d"
        colorbar_title = "count"

    labels = list(LABELS)
    # Per-cell annotation text (counts or normalized recall) so the matrix is
    # readable without hovering; built as plain strings, no Plotly object.
    annotations: list[dict[str, Any]] = []
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            raw = int(mat[i, j]) if not normalize else float(display[i, j])
            cell = f"{raw:{text_fmt}}"
            annotations.append(
                {
                    "x": labels[j],
                    "y": labels[i],
                    "text": cell,
                    "showarrow": False,
                    "font": {"color": "black"},
                }
            )

    data = [
        {
            "type": "heatmap",
            "z": _to_native(display),
            "x": labels,
            "y": labels,
            "zmin": 0.0,
            "colorscale": "Blues",
            "colorbar": {"title": {"text": colorbar_title}},
        }
    ]
    layout = {
        "title": {"text": title},
        "xaxis": {"title": {"text": "predicted"}, "side": "bottom"},
        # Mirror the y-axis so the true=pred diagonal runs top-left to bottom-right.
        "yaxis": {"title": {"text": "true"}, "autorange": "reversed"},
        "annotations": annotations,
    }
    return {"data": data, "layout": layout}


def per_class_f1_figure(
    per_class_f1: Sequence[float] | NDArray[np.float64],
    *,
    title: str = "Per-class F1 (test set)",
) -> FigureDict:
    """Build a per-class F1 bar-chart figure over the 3-way label space.

    One bar per class (negative/neutral/positive); the chart makes visible which
    class the macro-F1 average is being held down by.

    Parameters
    ----------
    per_class_f1:
        Per-class F1 values, length ``N_CLASSES``, in
        :data:`finbert_sentiment._constants.LABELS` order.
    title:
        Figure title.

    Returns
    -------
    FigureDict
        A ``{data, layout}`` mapping (a single ``bar`` trace).

    Raises
    ------
    ValidationError
        If ``per_class_f1`` does not have length ``N_CLASSES`` or contains
        out-of-range values.
    """
    vals = np.asarray(per_class_f1, dtype=np.float64)
    if vals.ndim != 1 or vals.shape[0] != N_CLASSES:
        raise ValidationError(f"per_class_f1 must have length {N_CLASSES}, got shape {vals.shape}.")
    if bool(np.isnan(vals).any()):
        raise ValidationError("per_class_f1 contains NaN values.")
    if bool(((vals < 0.0) | (vals > 1.0)).any()):
        raise ValidationError("per_class_f1 entries must lie in [0, 1].")

    labels = list(LABELS)
    values = [float(v) for v in vals]
    data = [
        {
            "type": "bar",
            "x": labels,
            "y": values,
            # Per-bar value labels so the figure reads without hover.
            "text": [f"{v:.3f}" for v in values],
            "textposition": "auto",
            "marker": {"color": values, "colorscale": "Blues", "cmin": 0.0, "cmax": 1.0},
        }
    ]
    layout = {
        "title": {"text": title},
        "xaxis": {"title": {"text": "class"}},
        "yaxis": {"title": {"text": "F1"}, "range": [0.0, 1.0]},
    }
    return {"data": data, "layout": layout}
