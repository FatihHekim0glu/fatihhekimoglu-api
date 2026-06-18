"""Unified predictor over the transformer-ONNX OR lexicon backend.

:class:`Predictor` exposes a single ``predict(texts) -> list[Prediction]``
interface and dispatches to whichever backend is available: the transformer-ONNX
session when the committed artifacts are present, else the torch-free lexicon
classifier. The selected backend is recorded so the API can report which model it
actually served. Nothing here imports a heavy dependency at module load.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

#: The backend identifiers surfaced to callers / the API ``served_model`` field.
BackendName = Literal["distilbert-onnx", "lexicon"]

#: The model the caller *requests* (the API may fall back to the lexicon).
RequestedModel = Literal["distilbert", "lexicon"]


@dataclass(frozen=True, slots=True)
class Prediction:
    """An immutable per-text prediction: label + the 3-way score vector.

    Attributes
    ----------
    text:
        The input sentence (echoed back for alignment in the UI table).
    label:
        The predicted class name (one of
        :data:`finbert_sentiment._constants.LABELS`).
    scores:
        ``{"negative": p, "neutral": p, "positive": p}`` scores summing to one.
    """

    text: str
    label: str
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this prediction."""
        return asdict(self)


class Predictor:
    """Backend-agnostic sentiment predictor (transformer-ONNX or lexicon).

    Construct via :func:`load_predictor` (which picks the backend by artifact
    availability) rather than directly. The chosen backend is fixed for the
    lifetime of the instance and exposed via :attr:`backend`.
    """

    def __init__(self, backend: BackendName, *, session: object | None = None) -> None:
        """Wire up the predictor around a concrete backend.

        Parameters
        ----------
        backend:
            Which backend this predictor serves.
        session:
            The backend object (an ``OnnxSentimentSession`` or a
            ``LexiconClassifier``); kept loosely typed so this module imports
            nothing heavy at load time.
        """
        self._backend: BackendName = backend
        self._session = session

    @property
    def backend(self) -> BackendName:
        """Return the backend identifier this predictor serves."""
        return self._backend

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        """Classify a batch and return one :class:`Prediction` per input text.

        Parameters
        ----------
        texts:
            The batch to classify (1..64 non-empty sentences).

        Returns
        -------
        list[Prediction]
            One prediction per input, in input order.

        Raises
        ------
        ValidationError
            If ``texts`` fails the batch validation.
        ArtifactError
            If the transformer backend is selected but its artifacts fail to load.
        """
        from finbert_sentiment._constants import INDEX_TO_LABEL, LABELS
        from finbert_sentiment._exceptions import ArtifactError
        from finbert_sentiment._validation import ensure_text_batch

        batch = ensure_text_batch(texts)
        session = self._session
        if session is None:  # pragma: no cover - load_predictor always wires a backend
            raise ArtifactError(
                f"Predictor.predict: backend {self._backend!r} has no session attached."
            )

        # Both backends expose the same ``predict_proba(texts) -> (n, N_CLASSES)``
        # row-stochastic interface, so dispatch is uniform.
        proba = session.predict_proba(batch)  # type: ignore[attr-defined]
        predictions: list[Prediction] = []
        for text, row in zip(batch, proba, strict=True):
            idx = int(row.argmax())
            scores = {label: float(row[i]) for i, label in enumerate(LABELS)}
            predictions.append(Prediction(text=text, label=INDEX_TO_LABEL[idx], scores=scores))
        return predictions


def load_predictor(
    requested: RequestedModel = "distilbert",
    *,
    artifact_dir: str | None = None,
) -> Predictor:
    """Build a :class:`Predictor`, selecting the backend by availability.

    If ``requested == "distilbert"`` AND the ONNX + tokenizer artifacts are
    present, the transformer backend is used; otherwise the predictor
    transparently falls back to the torch-free lexicon backend. ``requested ==
    "lexicon"`` always uses the lexicon.

    Parameters
    ----------
    requested:
        The model the caller would prefer.
    artifact_dir:
        Override for the serve-artifact directory (defaults to the package's
        ``artifacts/``).

    Returns
    -------
    Predictor
        A predictor bound to the resolved backend.
    """
    from finbert_sentiment.baselines.lexicon import LexiconClassifier
    from finbert_sentiment.inference.onnx_session import (
        OnnxSentimentSession,
        onnx_artifacts_present,
    )

    if requested == "distilbert" and onnx_artifacts_present(artifact_dir):
        session = OnnxSentimentSession(artifact_dir)
        return Predictor("distilbert-onnx", session=session)
    return Predictor("lexicon", session=LexiconClassifier())
