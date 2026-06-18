"""Public service entrypoint the FastAPI backend calls: :func:`run_sentiment`.

This is the single function the hosted tool and the backend router invoke. It
threads the serve + evaluation layers into one honest response WITHOUT recomputing
the offline evaluation per request:

* **Live predictions** come from :func:`finbert_sentiment.inference.load_predictor`,
  which serves the committed ONNX model via onnxruntime when present and otherwise
  falls back to the torch-free lexicon. The serve path imports ONLY
  onnxruntime + tokenizers (or nothing, for the lexicon) — never torch/transformers.
* **Evaluation metrics** (``eval_macro_f1``, ``lexicon_macro_f1``,
  ``class_prior_macro_f1``, ``beats_lexicon``, the confusion matrix, per-class F1)
  are loaded VERBATIM from the committed ``artifacts/metrics.json`` produced by the
  offline evaluation. They are NEVER recomputed on the request's input texts (which
  are unlabeled), keeping the reported macro-F1 the honest locked-test-set number.

The honest contract is preserved end-to-end: the summary reports macro-F1 (never
accuracy alone), names which model actually served, and surfaces ``beats_lexicon``
as ``None`` whenever no transformer was trained in this build (the lexicon-only
fallback). Sentiment is a TEXT LABEL, not a tradable signal — no alpha is claimed.

Importing this module has no side effects (json + path arithmetic only; the
predictor and any figure builders are imported lazily inside the function bodies).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from finbert_sentiment._constants import LABELS
from finbert_sentiment._exceptions import ArtifactError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from finbert_sentiment.inference.predictor import RequestedModel

#: Filename of the committed offline-measured evaluation bundle.
METRICS_ARTIFACT_NAME: str = "metrics.json"

#: Directory holding the committed serve artifacts (shared with the ONNX session).
_ARTIFACTS_DIR: Path = Path(__file__).resolve().parent / "artifacts"


def _safe_float(value: Any) -> float | None:
    """Coerce a value to a finite ``float`` or ``None`` (mirrors the API helper).

    Anything that is not a finite real number (``None``, ``NaN``, ``inf``, or a
    non-numeric value) becomes ``None`` so the JSON response never carries a
    non-serializable or misleading scalar.
    """
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):  # NaN / +-inf
        return None
    return out


def default_metrics_path(artifact_dir: str | Path | None = None) -> Path:
    """Return the path to the committed ``metrics.json`` (pure path arithmetic)."""
    directory = Path(artifact_dir) if artifact_dir else _ARTIFACTS_DIR
    return directory / METRICS_ARTIFACT_NAME


def load_committed_metrics(artifact_dir: str | Path | None = None) -> dict[str, Any]:
    """Load the offline-measured evaluation bundle from ``artifacts/metrics.json``.

    Pure stdlib (``json``) — imports nothing heavy and recomputes nothing. The
    returned mapping is the bundle the offline evaluation committed; the service
    reads the eval scalars from it verbatim.

    Parameters
    ----------
    artifact_dir:
        Override for the artifact directory (defaults to the package's
        ``artifacts/``).

    Returns
    -------
    dict
        The parsed metrics mapping.

    Raises
    ------
    ArtifactError
        If the metrics file is missing or is not valid JSON.
    """
    path = default_metrics_path(artifact_dir)
    if not path.is_file():
        raise ArtifactError(
            f"committed metrics file not found at {path}; the offline evaluation "
            "must write metrics.json (run `finbert-sentiment evaluate`)."
        )
    try:
        with path.open(encoding="utf-8") as handle:
            data: dict[str, Any] = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"failed to read committed metrics from {path}: {exc}") from exc
    return data


@dataclass(frozen=True, slots=True)
class SentimentSummary:
    """The honest summary block of a :func:`run_sentiment` response.

    Attributes
    ----------
    served_model:
        Which backend actually served the live predictions
        (``"distilbert-onnx"`` or ``"lexicon"``).
    eval_macro_f1:
        The served model's OFFLINE-measured macro-F1 on the locked test set
        (loaded from the committed metrics, never recomputed per request).
    eval_macro_f1_ci:
        The ``(low, high)`` bootstrap CI for ``eval_macro_f1`` (or ``None``).
    eval_accuracy:
        The offline accuracy (reported alongside, never instead of, macro-F1).
    lexicon_macro_f1:
        The lexicon baseline's measured macro-F1 (the honest floor).
    class_prior_macro_f1:
        The class-prior baseline's measured macro-F1 (the trivial floor).
    beats_lexicon:
        ``True``/``False`` only when a transformer was trained and compared;
        ``None`` on the lexicon-only fallback build.
    n_texts:
        The number of input sentences classified in THIS request.
    data_source:
        The evaluation data source (the Financial PhraseBank mirror + config).
    transformer_measured:
        Whether the transformer macro-F1 was actually measured in this build (if
        ``False``, the figure cited is the published/expected one).
    notes:
        The honest caption (sentiment is a text label, not a tradable signal).
    """

    served_model: str
    eval_macro_f1: float | None
    eval_macro_f1_ci: tuple[float, float] | None
    eval_accuracy: float | None
    lexicon_macro_f1: float | None
    class_prior_macro_f1: float | None
    beats_lexicon: bool | None
    n_texts: int
    data_source: str
    transformer_measured: bool
    notes: str

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the summary."""
        ci = self.eval_macro_f1_ci
        return {
            "served_model": self.served_model,
            "eval_macro_f1": self.eval_macro_f1,
            "eval_macro_f1_ci": [ci[0], ci[1]] if ci is not None else None,
            "eval_accuracy": self.eval_accuracy,
            "lexicon_macro_f1": self.lexicon_macro_f1,
            "class_prior_macro_f1": self.class_prior_macro_f1,
            "beats_lexicon": self.beats_lexicon,
            "n_texts": self.n_texts,
            "data_source": self.data_source,
            "transformer_measured": self.transformer_measured,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class SentimentResult:
    """The full :func:`run_sentiment` result: summary + predictions + figures.

    Attributes
    ----------
    summary:
        The honest :class:`SentimentSummary`.
    predictions:
        One ``{"text", "label", "scores"}`` record per input sentence.
    confusion_figure:
        A Plotly ``{data, layout}`` dict of the offline test-set confusion matrix.
    per_class_f1_figure:
        A Plotly ``{data, layout}`` dict of the offline per-class F1 bar.
    """

    summary: SentimentSummary
    predictions: list[dict[str, Any]] = field(default_factory=list)
    confusion_figure: dict[str, Any] = field(default_factory=dict)
    per_class_f1_figure: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the whole response."""
        return {
            "summary": self.summary.to_dict(),
            "predictions": self.predictions,
            "confusion_figure": self.confusion_figure,
            "per_class_f1_figure": self.per_class_f1_figure,
        }


def _ci_tuple(raw: Any) -> tuple[float, float] | None:
    """Coerce a committed CI pair to a finite ``(low, high)`` tuple, or ``None``."""
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    lo = _safe_float(raw[0])
    hi = _safe_float(raw[1])
    if lo is None or hi is None:
        return None
    return (lo, hi)


def build_evaluation_figures(metrics: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Assemble the confusion + per-class-F1 figures from committed metrics.

    Reads the committed ``confusion`` matrix and ``per_class_f1`` vector and
    builds the two honest figures (the test-set confusion heatmap and the
    per-class F1 bar) via the pure :mod:`finbert_sentiment.plots` builders. No
    figure is recomputed from request input — these depict the LOCKED test set.

    Parameters
    ----------
    metrics:
        The committed metrics mapping (see :func:`load_committed_metrics`).

    Returns
    -------
    tuple
        ``(confusion_figure, per_class_f1_figure)`` as ``{data, layout}`` dicts.

    Raises
    ------
    ArtifactError
        If the committed metrics lack a usable confusion matrix / F1 vector.
    """
    from finbert_sentiment.plots import confusion_matrix_figure, per_class_f1_figure

    confusion = metrics.get("confusion")
    per_class_f1 = metrics.get("per_class_f1")
    if confusion is None or per_class_f1 is None:
        raise ArtifactError(
            "committed metrics are missing 'confusion' and/or 'per_class_f1'; "
            "regenerate metrics.json via the offline evaluation."
        )
    try:
        confusion_fig = confusion_matrix_figure(confusion)
        f1_fig = per_class_f1_figure([float(v) for v in per_class_f1])
    except Exception as exc:  # normalize plot/validation failures
        raise ArtifactError(f"failed to build evaluation figures from metrics: {exc}") from exc
    return confusion_fig, f1_fig


def run_sentiment(
    texts: Sequence[str],
    *,
    model_pref: RequestedModel = "distilbert",
    seed: int = 0,
    artifact_dir: str | Path | None = None,
) -> SentimentResult:
    """Classify ``texts`` and assemble the honest response (the backend entrypoint).

    The single function the FastAPI router calls. It:

    1. Loads the committed offline evaluation bundle (``metrics.json``) — the
       reported ``eval_macro_f1`` / ``lexicon_macro_f1`` / ``class_prior_macro_f1``
       / ``beats_lexicon`` come from here VERBATIM, never recomputed per request.
    2. Serves live per-sentence predictions via
       :func:`finbert_sentiment.inference.load_predictor` (ONNX when present, else
       the torch-free lexicon). The serve path imports no torch/transformers.
    3. Builds the test-set confusion + per-class-F1 figures from the committed
       metrics.

    Parameters
    ----------
    texts:
        The 1..64 input sentences to classify.
    model_pref:
        The model the caller prefers (``"distilbert"`` or ``"lexicon"``); the
        predictor falls back to the lexicon when no ONNX artifact is present.
    seed:
        Accepted for interface symmetry / provenance; inference is deterministic
        so it does not change the result (no per-request randomness).
    artifact_dir:
        Override for the artifact directory (defaults to the package's
        ``artifacts/``).

    Returns
    -------
    SentimentResult
        The summary, per-sentence predictions, and the two evaluation figures.

    Raises
    ------
    ValidationError
        If ``texts`` fails the 1..64 non-empty batch validation.
    ArtifactError
        If the committed metrics or the requested serve artifacts cannot load.
    """
    # Reproducibility provenance only: inference is deterministic, but record the
    # seed so callers can thread it through unchanged.
    del seed

    # LAZY: the predictor pulls in onnxruntime/tokenizers (ONNX backend) or the
    # pure-Python lexicon — never torch/transformers.
    from finbert_sentiment.inference.predictor import load_predictor

    metrics = load_committed_metrics(artifact_dir)
    predictor = load_predictor(model_pref, artifact_dir=str(artifact_dir) if artifact_dir else None)
    predictions = predictor.predict(texts)

    labels = list(LABELS)
    prediction_records = [
        {
            "text": p.text,
            "label": p.label,
            "scores": {name: _safe_float(p.scores.get(name, 0.0)) for name in labels},
        }
        for p in predictions
    ]

    confusion_fig, f1_fig = build_evaluation_figures(metrics)

    summary = SentimentSummary(
        served_model=predictor.backend,
        eval_macro_f1=_safe_float(metrics.get("eval_macro_f1")),
        eval_macro_f1_ci=_ci_tuple(metrics.get("eval_macro_f1_ci")),
        eval_accuracy=_safe_float(metrics.get("eval_accuracy")),
        lexicon_macro_f1=_safe_float(metrics.get("lexicon_macro_f1")),
        class_prior_macro_f1=_safe_float(metrics.get("class_prior_macro_f1")),
        beats_lexicon=metrics.get("beats_lexicon"),
        n_texts=len(prediction_records),
        data_source=str(metrics.get("data_source", "unknown")),
        transformer_measured=bool(
            metrics.get("transformer_published_macro_f1", {}).get("measured_in_this_build", False)
        ),
        notes=str(
            metrics.get(
                "notes",
                "Sentiment is a text label, not a tradable signal — no alpha is claimed.",
            )
        ),
    )
    return SentimentResult(
        summary=summary,
        predictions=prediction_records,
        confusion_figure=confusion_fig,
        per_class_f1_figure=f1_fig,
    )
