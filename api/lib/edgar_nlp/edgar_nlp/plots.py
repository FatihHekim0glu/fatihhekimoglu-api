"""Plotly figure builders (lazy ``viz`` extra).

Each builder returns a plain ``dict`` shaped ``{"data": [...], "layout": {...}}``
— the same JSON shape the FastAPI layer serializes and the Next.js
``PlotlyChart`` component renders — so the figures cross the API boundary with no
Plotly object leaking through. Plotly is an OPTIONAL dependency (the ``viz``
extra) imported lazily inside each builder; importing this module has no side
effects and does not require Plotly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from edgar_nlp._constants import SECTION_LABELS

if TYPE_CHECKING:
    from edgar_nlp.readability.metrics import ReadabilityScore
    from edgar_nlp.sentiment.lm import LMScore

# quantcore-candidate: mirrors hrp-portfolio:src/hrp/plots.py ({data, layout}
# figure shape, lazy plotly, numpy/pandas jsonification).

#: A Plotly figure serialized as a plain mapping with ``data`` and ``layout`` keys.
FigureDict = dict[str, Any]


def _section_label(slug: str) -> str:
    """Human-facing label for a section slug (falls back to the slug itself)."""
    return SECTION_LABELS.get(slug, slug)


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


def _as_plain_dict(obj: Any) -> dict[str, Any]:
    """Coerce a Plotly graph-object (trace/layout/figure) to a plain JSON-safe dict."""
    raw = obj.to_plotly_json() if hasattr(obj, "to_plotly_json") else dict(obj)
    result: dict[str, Any] = _jsonify(raw)
    return result


def tone_by_section_figure(scores_by_section: dict[str, LMScore]) -> FigureDict:
    """Build a grouped bar figure of LM tone by section.

    One group per section (Risk Factors, MD&A) with bars for net tone and the
    negative/positive/uncertainty fractions, so the reader can see WHICH section
    drives the document tone.

    LAZY IMPORT: ``plotly`` is imported inside this function (the ``viz`` extra).

    Parameters
    ----------
    scores_by_section:
        Mapping of section slug to its :class:`~edgar_nlp.sentiment.lm.LMScore`.

    Returns
    -------
    FigureDict
        A ``{"data": [...], "layout": {...}}`` figure dict.
    """
    # LAZY import: keep Plotly off this pure module's import path.
    import plotly.graph_objects as go

    slugs = list(scores_by_section)
    labels = [_section_label(slug) for slug in slugs]

    # One bar trace per LM metric, grouped by section, so the reader can see which
    # section drives the net tone and how the underlying fractions decompose it.
    metrics: tuple[tuple[str, str], ...] = (
        ("tone", "Net tone (neg - pos)"),
        ("neg_pct", "Negative fraction"),
        ("pos_pct", "Positive fraction"),
        ("uncertainty_pct", "Uncertainty fraction"),
    )

    fig = go.Figure()
    for attr, name in metrics:
        values = [_jsonify(getattr(scores_by_section[slug], attr)) for slug in slugs]
        fig.add_trace(go.Bar(name=name, x=labels, y=values))

    fig.update_layout(
        barmode="group",
        title={"text": "Loughran-McDonald tone by section"},
        xaxis={"title": {"text": "section"}},
        yaxis={"title": {"text": "fraction / net tone"}, "tickformat": ".1%"},
        legend={"orientation": "h"},
    )

    layout = _as_plain_dict(fig.layout)
    return {"data": [_as_plain_dict(trace) for trace in fig.data], "layout": layout}


def readability_figure(scores_by_section: dict[str, ReadabilityScore]) -> FigureDict:
    """Build a grouped bar figure of readability indices by section.

    One group per section with bars for Gunning-Fog, Flesch-Kincaid grade, and
    SMOG, so the reader can compare how dense each section reads.

    LAZY IMPORT: ``plotly`` is imported inside this function (the ``viz`` extra).

    Parameters
    ----------
    scores_by_section:
        Mapping of section slug to its
        :class:`~edgar_nlp.readability.metrics.ReadabilityScore`.

    Returns
    -------
    FigureDict
        A ``{"data": [...], "layout": {...}}`` figure dict.
    """
    # LAZY import: keep Plotly off this pure module's import path.
    import plotly.graph_objects as go

    slugs = list(scores_by_section)
    labels = [_section_label(slug) for slug in slugs]

    # One bar trace per readability index, grouped by section, so the reader can
    # compare how densely each section reads (grade-level units).
    metrics: tuple[tuple[str, str], ...] = (
        ("gunning_fog", "Gunning-Fog"),
        ("flesch_kincaid", "Flesch-Kincaid grade"),
        ("smog", "SMOG"),
    )

    fig = go.Figure()
    for attr, name in metrics:
        values = [_jsonify(getattr(scores_by_section[slug], attr)) for slug in slugs]
        fig.add_trace(go.Bar(name=name, x=labels, y=values))

    fig.update_layout(
        barmode="group",
        title={"text": "Readability indices by section"},
        xaxis={"title": {"text": "section"}},
        yaxis={"title": {"text": "grade level"}},
        legend={"orientation": "h"},
    )

    layout = _as_plain_dict(fig.layout)
    return {"data": [_as_plain_dict(trace) for trace in fig.data], "layout": layout}
