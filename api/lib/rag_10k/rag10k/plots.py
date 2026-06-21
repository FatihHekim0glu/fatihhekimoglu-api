"""Plotly figure builders (LAZY plotly, [viz]) — serialize to ``{data, layout}``.

Two figures back the frontend tool:

- :func:`retrieval_score_figure` — a horizontal bar of the top-k retrieved chunks'
  cosine scores (annotated with the abstention threshold), so the eye can see how
  confidently the filing answers the question (and when it abstains);
- :func:`eval_summary_figure` — a small bar of the frozen-harness metrics
  (recall@k, citation-soundness, abstention rate), so the honest evaluation is
  legible at a glance.

``plotly`` is imported LAZILY inside the builders (the ``[viz]`` extra), and each
figure is returned as a plain ``{data, layout}`` dict via
``json.loads(pio.to_json(fig, validate=False))`` so the API response carries no
plotly objects. Importing this module has no side effects.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from rag10k._constants import ABSTENTION_THRESHOLD
from rag10k._exceptions import ValidationError

# quantcore-candidate: mirrors mvts-forecast / hrp plots.py ({data, layout} shape).

#: Stable, colour-blind-aware palette.
_PALETTE: tuple[str, ...] = (
    "#2563eb",  # blue — retrieved chunk scores
    "#16a34a",  # green — metric bars
    "#dc2626",  # red — abstention threshold marker
    "#f97316",  # orange
    "#9333ea",  # purple
)


def retrieval_score_figure(
    chunk_ids: Sequence[str],
    scores: Sequence[float],
    *,
    threshold: float = ABSTENTION_THRESHOLD,
    title: str = "Top-k retrieval scores (cosine)",
) -> dict[str, Any]:
    """Build the top-k retrieval-score bar as a ``{data, layout}`` dict.

    LAZY IMPORT: ``plotly`` is imported inside this function (the ``[viz]`` extra).
    Plots one bar per retrieved chunk (score, labelled by chunk id) with the
    abstention threshold drawn as a reference line, then serializes to a plain
    dict.

    Parameters
    ----------
    chunk_ids:
        The retrieved chunk ids (bar labels), in rank order.
    scores:
        Their cosine scores (same length as ``chunk_ids``).
    threshold:
        The abstention threshold to annotate.
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
        If ``chunk_ids`` and ``scores`` disagree in length, or a score is
        non-finite.
    """
    ids = list(chunk_ids)
    values = [_safe_score(score, name="score") for score in scores]
    if len(ids) != len(values):
        raise ValidationError(
            "retrieval_score_figure: chunk_ids and scores must have the same length "
            f"(got {len(ids)} ids and {len(values)} scores)."
        )
    if not _is_finite(threshold):
        raise ValidationError(
            f"retrieval_score_figure: threshold must be finite, got {threshold!r}."
        )

    # Lazy plotly import: the figure builders are the only consumers of the
    # ``[viz]`` extra, so importing this module never pulls plotly in.
    import plotly.graph_objects as go
    import plotly.io as pio

    # Bars are drawn top-to-bottom in rank order: reverse so rank-0 (the most
    # similar chunk) sits at the TOP of the horizontal bar chart.
    ids_top_down = list(reversed(ids))
    values_top_down = list(reversed(values))

    fig = go.Figure(
        go.Bar(
            x=values_top_down,
            y=ids_top_down,
            orientation="h",
            marker={"color": _PALETTE[0]},
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            name="cosine score",
        )
    )
    # Draw the abstention threshold as a reference line so the eye can read when a
    # retrieval would have abstained (top bar below the line).
    fig.add_vline(
        x=threshold,
        line={"color": _PALETTE[2], "dash": "dash", "width": 2},
        annotation_text=f"abstain < {threshold:.2f}",
        annotation_position="top",
    )
    fig.update_layout(
        title=title,
        xaxis_title="cosine similarity",
        yaxis_title="chunk id",
        template="plotly_white",
        showlegend=False,
        margin={"l": 80, "r": 30, "t": 60, "b": 50},
    )

    # Serialize to a plain ``{data, layout}`` dict so the API response carries no
    # plotly objects (``validate=False`` skips the round-trip schema validation).
    serialized: dict[str, Any] = json.loads(pio.to_json(fig, validate=False))
    return serialized


def eval_summary_figure(
    metrics: Mapping[str, float],
    *,
    title: str = "Frozen 150-Q eval (recall@k / citation-soundness / abstention)",
) -> dict[str, Any]:
    """Build the frozen-harness metric bar as a ``{data, layout}`` dict.

    LAZY IMPORT: ``plotly`` is imported inside this function (the ``[viz]`` extra).
    Plots one bar per metric (recall@k, citation-soundness, abstention rate) on a
    ``[0, 1]`` axis, then serializes to a plain dict.

    Parameters
    ----------
    metrics:
        Map of metric name -> value in ``[0, 1]``.
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
        If ``metrics`` is empty or contains a non-finite value.
    """
    items = list(metrics.items())
    if not items:
        raise ValidationError("eval_summary_figure: metrics must be non-empty.")

    names = [str(name) for name, _ in items]
    values = [_safe_score(value, name=str(name)) for name, value in items]

    # Lazy plotly import (the ``[viz]`` extra).
    import plotly.graph_objects as go
    import plotly.io as pio

    fig = go.Figure(
        go.Bar(
            x=names,
            y=values,
            marker={"color": _PALETTE[1]},
            text=[f"{value:.3f}" for value in values],
            textposition="outside",
            hovertemplate="%{x}: %{y:.4f}<extra></extra>",
            name="metric",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="metric",
        yaxis_title="value",
        # The honest IR metrics all live in ``[0, 1]``; pin the axis so they are
        # legible on a common scale (with headroom for the outside labels).
        yaxis={"range": [0.0, 1.05]},
        template="plotly_white",
        showlegend=False,
        margin={"l": 60, "r": 30, "t": 60, "b": 60},
    )

    serialized: dict[str, Any] = json.loads(pio.to_json(fig, validate=False))
    return serialized


def _is_finite(value: object) -> bool:
    """Return ``True`` iff ``value`` is a real, finite number (no NaN / inf)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _safe_score(value: object, *, name: str) -> float:
    """Coerce ``value`` to a finite ``float`` or raise :class:`ValidationError`.

    Keeps every figure input numeric and finite so the serialized ``{data, layout}``
    dict is always JSON-clean (no NaN / inf leaks into the API response).
    """
    if not _is_finite(value):
        raise ValidationError(f"figure {name} must be a finite number, got {value!r}.")
    return float(value)  # type: ignore[arg-type]  # _is_finite narrows to a real number
