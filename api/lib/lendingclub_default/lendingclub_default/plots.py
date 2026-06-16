"""Plotly figure builders (lazy plotly import).

Each builder returns a ``plotly.graph_objects.Figure`` that serializes to a
``{data, layout}`` JSON blob the API and frontend render. Plotly is imported
LAZILY inside each function so importing this module stays pure (no plotly on the
import path of the compute library).

Figures:

- :func:`roc_figure` — ROC curve (with the 0.5 diagonal and the AUC in the title).
- :func:`pr_figure` — precision-recall curve (with the base-rate floor line).
- :func:`calibration_figure` — reliability curve vs. the perfect-calibration
  diagonal.
- :func:`score_distribution_figure` — predicted-PD histograms split by realised
  outcome (the separation the KS statistic summarizes).

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from lendingclub_default._exceptions import ValidationError
from lendingclub_default._validation import ensure_series

if TYPE_CHECKING:
    import pandas as pd
    import plotly.graph_objects as go

    from lendingclub_default.evaluation.calibration import ReliabilityCurve

# quantcore-candidate: new code (Plotly builders); lazy plotly import.


def _coerce_scores(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    *,
    score_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce ``(y_true, y_score)`` to aligned, finite, binary-labelled float arrays.

    ``y_true`` is validated as binary ``{0, 1}`` labels; ``y_score`` is any finite
    score/probability vector of the same length. ``ensure_series`` already rejects
    empty / non-finite inputs, so the only extra guards here are length alignment,
    binary labels, and both-classes-present.

    Raises
    ------
    ValidationError
        If the inputs are misaligned, non-finite, empty, or ``y_true`` is not
        binary or single-class.
    """
    true = ensure_series(y_true, name="y_true").to_numpy(dtype="float64")
    score = ensure_series(y_score, name=score_name).to_numpy(dtype="float64")
    if true.shape[0] != score.shape[0]:
        raise ValidationError(
            f"y_true and {score_name} must have equal length, "
            f"got {true.shape[0]} and {score.shape[0]}."
        )
    uniq = np.unique(true)
    if not np.all(np.isin(uniq, (0.0, 1.0))):
        raise ValidationError(f"y_true must be binary {{0, 1}}, got values {uniq.tolist()}.")
    if uniq.shape[0] < 2:
        raise ValidationError("y_true must contain both classes for this figure.")
    return true, score


def _roc_points(true: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(fpr, tpr)`` step points of the ROC curve (sklearn-compatible).

    Thresholds descend through the distinct scores; ties at a common threshold are
    collapsed so the curve steps exactly where ``sklearn.metrics.roc_curve`` does.
    The ``(0, 0)`` origin is prepended.
    """
    order = np.argsort(-score, kind="mergesort")
    s_sorted = score[order]
    y_sorted = true[order]

    distinct = np.r_[np.diff(s_sorted) != 0, True]
    tps = np.cumsum(y_sorted)[distinct]
    fps = np.cumsum(1.0 - y_sorted)[distinct]

    n_pos = tps[-1]
    n_neg = fps[-1]
    tpr = np.r_[0.0, tps / n_pos]
    fpr = np.r_[0.0, fps / n_neg]
    return fpr, tpr


def _pr_points(true: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(recall, precision)`` step points of the precision-recall curve.

    Mirrors :func:`sklearn.metrics.precision_recall_curve`: precision/recall step at
    distinct thresholds, with the terminal ``(recall=0, precision=1)`` point appended.
    """
    order = np.argsort(-score, kind="mergesort")
    s_sorted = score[order]
    y_sorted = true[order]

    distinct = np.r_[np.diff(s_sorted) != 0, True]
    tps = np.cumsum(y_sorted)[distinct]
    fps = np.cumsum(1.0 - y_sorted)[distinct]

    n_pos = tps[-1]
    precision = tps / (tps + fps)
    recall = tps / n_pos
    # Append the canonical (recall=0, precision=1) endpoint (sklearn convention).
    recall = np.r_[recall, 0.0]
    precision = np.r_[precision, 1.0]
    return recall, precision


def roc_figure(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    *,
    title: str = "ROC curve",
) -> go.Figure:
    """Build a ROC-curve figure with the no-skill diagonal and AUC annotation.

    Parameters
    ----------
    y_true:
        Binary labels (1 = default).
    y_score:
        Predicted scores.
    title:
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
        The ROC figure.

    Raises
    ------
    ValidationError
        If the inputs are misaligned, non-finite, or single-class.
    """
    import plotly.graph_objects as go

    from lendingclub_default.evaluation.metrics import roc_auc

    true, score = _coerce_scores(y_true, y_score, score_name="y_score")
    fpr, tpr = _roc_points(true, score)
    auc = roc_auc(true, score)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[float(v) for v in fpr],
            y=[float(v) for v in tpr],
            mode="lines",
            name=f"ROC (AUC = {auc:.3f})",
            line={"color": "#2563eb", "width": 2},
            fill="tozeroy",
            fillcolor="rgba(37, 99, 235, 0.10)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            mode="lines",
            name="No skill (AUC = 0.500)",
            line={"color": "#9ca3af", "width": 1, "dash": "dash"},
        )
    )
    fig.update_layout(
        title={"text": f"{title} — AUC = {auc:.3f}"},
        xaxis={"title": {"text": "False positive rate"}, "range": [0.0, 1.0]},
        yaxis={"title": {"text": "True positive rate"}, "range": [0.0, 1.0]},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.25},
        template="plotly_white",
    )
    return fig


def pr_figure(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    *,
    title: str = "Precision-Recall curve",
) -> go.Figure:
    """Build a precision-recall figure with the base-rate (no-skill) floor.

    Parameters
    ----------
    y_true:
        Binary labels (1 = default).
    y_score:
        Predicted scores.
    title:
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
        The PR figure.

    Raises
    ------
    ValidationError
        If the inputs are misaligned, non-finite, or single-class.
    """
    import plotly.graph_objects as go

    from lendingclub_default.evaluation.metrics import pr_auc

    true, score = _coerce_scores(y_true, y_score, score_name="y_score")
    recall, precision = _pr_points(true, score)
    ap = pr_auc(true, score)
    base_rate = float(np.mean(true))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[float(v) for v in recall],
            y=[float(v) for v in precision],
            mode="lines",
            name=f"PR (AP = {ap:.3f})",
            line={"color": "#7c3aed", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[base_rate, base_rate],
            mode="lines",
            name=f"Base rate ({base_rate:.3f})",
            line={"color": "#9ca3af", "width": 1, "dash": "dash"},
        )
    )
    fig.update_layout(
        title={"text": f"{title} — AP = {ap:.3f}, base rate = {base_rate:.3f}"},
        xaxis={"title": {"text": "Recall"}, "range": [0.0, 1.0]},
        yaxis={"title": {"text": "Precision"}, "range": [0.0, 1.0]},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.25},
        template="plotly_white",
    )
    return fig


def calibration_figure(
    curve: ReliabilityCurve,
    *,
    title: str = "Calibration (reliability) curve",
) -> go.Figure:
    """Build a reliability-curve figure vs. the perfect-calibration diagonal.

    Parameters
    ----------
    curve:
        A computed :class:`lendingclub_default.evaluation.calibration.ReliabilityCurve`.
    title:
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
        The calibration figure.

    Raises
    ------
    ValidationError
        If the curve carries no (non-empty) bins.
    """
    import plotly.graph_objects as go

    mean_predicted = [float(v) for v in curve.mean_predicted]
    observed = [float(v) for v in curve.observed_frequency]
    if len(mean_predicted) == 0:
        raise ValidationError("calibration_figure: the reliability curve has no bins.")
    if not all(np.isfinite([*mean_predicted, *observed])):
        raise ValidationError("calibration_figure: the reliability curve has non-finite points.")

    ece = float(curve.expected_calibration_error)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            mode="lines",
            name="Perfect calibration",
            line={"color": "#9ca3af", "width": 1, "dash": "dash"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=mean_predicted,
            y=observed,
            mode="lines+markers",
            name=f"Model (ECE = {ece:.3f})",
            line={"color": "#059669", "width": 2},
            marker={"size": 8},
        )
    )
    fig.update_layout(
        title={"text": f"{title} — ECE = {ece:.3f}"},
        xaxis={"title": {"text": "Mean predicted PD"}, "range": [0.0, 1.0]},
        yaxis={"title": {"text": "Observed default frequency"}, "range": [0.0, 1.0]},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.25},
        template="plotly_white",
    )
    return fig


def score_distribution_figure(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    *,
    title: str = "Predicted PD by outcome",
    n_bins: int = 40,
) -> go.Figure:
    """Build overlaid predicted-PD histograms split by realised outcome.

    Parameters
    ----------
    y_true:
        Binary labels (1 = default).
    y_prob:
        Calibrated probabilities in ``[0, 1]``.
    title:
        Figure title.
    n_bins:
        Histogram bin count.

    Returns
    -------
    plotly.graph_objects.Figure
        The score-distribution figure.

    Raises
    ------
    ValidationError
        If the inputs are misaligned, single-class, ``n_bins`` is not positive, or
        ``y_prob`` falls outside ``[0, 1]``.
    """
    import plotly.graph_objects as go

    if n_bins < 1:
        raise ValidationError(f"n_bins must be >= 1, got {n_bins}.")

    true, prob = _coerce_scores(y_true, y_prob, score_name="y_prob")
    if bool(np.any(prob < 0.0)) or bool(np.any(prob > 1.0)):
        raise ValidationError("y_prob must lie within [0, 1] for the score distribution.")

    bins = {"start": 0.0, "end": 1.0, "size": 1.0 / float(n_bins)}
    defaulted = prob[true == 1.0]
    repaid = prob[true == 0.0]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=[float(v) for v in repaid],
            name="Fully paid",
            xbins=bins,
            marker={"color": "#2563eb"},
            opacity=0.6,
            histnorm="probability density",
        )
    )
    fig.add_trace(
        go.Histogram(
            x=[float(v) for v in defaulted],
            name="Charged off / default",
            xbins=bins,
            marker={"color": "#dc2626"},
            opacity=0.6,
            histnorm="probability density",
        )
    )
    fig.update_layout(
        title={"text": title},
        barmode="overlay",
        xaxis={"title": {"text": "Predicted probability of default"}, "range": [0.0, 1.0]},
        yaxis={"title": {"text": "Density"}},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.25},
        template="plotly_white",
    )
    return fig


def figure_to_dict(figure: go.Figure) -> dict[str, Any]:
    """Serialize a Plotly figure to a ``{data, layout}`` JSON-safe ``dict``.

    Mirrors the API contract ``json.loads(pio.to_json(fig, validate=False))`` so
    the same payload shape is produced in the library and the router.

    Parameters
    ----------
    figure:
        The Plotly figure to serialize.

    Returns
    -------
    dict[str, Any]
        A ``{data, layout}`` mapping.
    """
    import json

    import plotly.io as pio

    payload: dict[str, Any] = json.loads(pio.to_json(figure, validate=False))
    return payload
