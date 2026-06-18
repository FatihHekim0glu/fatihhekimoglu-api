"""ONNX inference session — the transformer SERVE path (onnxruntime + tokenizers).

The container and the FastAPI router run the fine-tuned DistilBERT through this
module ONLY. It loads the committed ``artifacts/model*.onnx`` graph with
onnxruntime and the ``artifacts/tokenizer.json`` with the ``tokenizers`` library;
torch and transformers are NEVER imported here. Both heavy imports happen LAZILY
inside :meth:`OnnxSentimentSession.load`, so ``import finbert_sentiment`` stays
free of any inference engine.

Importing this module has no side effects.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray

#: Directory holding the committed, shipped serve artifacts (ONNX + tokenizer).
ARTIFACTS_DIR: Path = Path(__file__).resolve().parent.parent / "artifacts"

#: Filenames of the committed serve artifacts.
ONNX_ARTIFACT_NAME: str = "model.int8.onnx"
TOKENIZER_ARTIFACT_NAME: str = "tokenizer.json"


def default_artifact_dir() -> Path:
    """Return the package's ``artifacts/`` directory (pure path arithmetic).

    Does NOT check existence and imports nothing heavy, so it is safe to call at
    a caller's import time.

    Returns
    -------
    pathlib.Path
        ``<package>/artifacts``.
    """
    return ARTIFACTS_DIR


def onnx_artifacts_present(artifact_dir: str | Path | None = None) -> bool:
    """Return ``True`` iff BOTH the ONNX graph and the tokenizer.json exist.

    Used by the predictor to decide whether the transformer backend is available
    or it must fall back to the lexicon. Pure filesystem check — imports nothing.

    Parameters
    ----------
    artifact_dir:
        Directory to probe (defaults to :func:`default_artifact_dir`).

    Returns
    -------
    bool
        Whether the transformer serve artifacts are both present.
    """
    directory = Path(artifact_dir) if artifact_dir else default_artifact_dir()
    onnx_ok = (directory / ONNX_ARTIFACT_NAME).is_file()
    tok_ok = (directory / TOKENIZER_ARTIFACT_NAME).is_file()
    return onnx_ok and tok_ok


class OnnxSentimentSession:
    """A thin onnxruntime + tokenizers wrapper that serves the committed model.

    The onnxruntime session and the tokenizer are created LAZILY on first
    :meth:`predict_proba` (or via :meth:`load`), so constructing this object is
    cheap and import-pure.
    """

    def __init__(self, artifact_dir: str | Path | None = None) -> None:
        """Record the artifact directory; defer session/tokenizer creation to :meth:`load`.

        Parameters
        ----------
        artifact_dir:
            Directory holding ``model*.onnx`` + ``tokenizer.json``. Defaults to
            the package's shipped artifacts (:func:`default_artifact_dir`).
        """
        self._artifact_dir = Path(artifact_dir) if artifact_dir else default_artifact_dir()
        self._session: object | None = None
        self._tokenizer: object | None = None

    @property
    def artifact_dir(self) -> Path:
        """Return the resolved artifact directory this session will load from."""
        return self._artifact_dir

    def load(self) -> OnnxSentimentSession:
        """Create the onnxruntime session and load the tokenizer (lazy, idempotent).

        LAZY IMPORT: ``onnxruntime`` and ``tokenizers`` are imported inside this
        method. NO torch / transformers import occurs anywhere on this path.

        Returns
        -------
        OnnxSentimentSession
            ``self``, with an initialized session and tokenizer.

        Raises
        ------
        ArtifactError
            If either artifact is missing or initialization fails.
        """
        if self._session is not None and self._tokenizer is not None:
            return self

        from finbert_sentiment._exceptions import ArtifactError

        onnx_path = self._artifact_dir / ONNX_ARTIFACT_NAME
        tokenizer_path = self._artifact_dir / TOKENIZER_ARTIFACT_NAME
        if not onnx_path.is_file():
            raise ArtifactError(
                f"OnnxSentimentSession.load: ONNX artifact not found at {onnx_path}."
            )
        if not tokenizer_path.is_file():
            raise ArtifactError(
                f"OnnxSentimentSession.load: tokenizer.json not found at {tokenizer_path}."
            )

        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            self._session = ort.InferenceSession(
                str(onnx_path),
                providers=["CPUExecutionProvider"],
            )
            self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        except ArtifactError:
            raise
        except Exception as exc:  # normalize any onnxruntime/tokenizers error
            raise ArtifactError(
                f"OnnxSentimentSession.load: failed to initialize the onnxruntime "
                f"session or tokenizer from {self._artifact_dir}: {exc}"
            ) from exc
        return self

    def predict_proba(self, texts: Sequence[str]) -> NDArray[np.float64]:
        """Tokenize, run the ONNX forward pass, and softmax to class scores.

        Loads the session on first use. Returns one row of
        ``(neg, neutral, pos)`` softmax probabilities per input text, in the
        canonical :data:`finbert_sentiment._constants.LABELS` order.

        Parameters
        ----------
        texts:
            The batch to classify.

        Returns
        -------
        numpy.ndarray
            A ``(len(texts), N_CLASSES)`` row-stochastic score matrix.

        Raises
        ------
        ArtifactError
            If the session/tokenizer cannot be loaded or the forward pass fails.
        ValidationError
            If ``texts`` fails the batch validation.
        """
        import numpy as np

        from finbert_sentiment._constants import N_CLASSES
        from finbert_sentiment._exceptions import ArtifactError
        from finbert_sentiment._validation import ensure_text_batch

        batch = ensure_text_batch(texts)
        self.load()
        session = self._session
        tokenizer = self._tokenizer
        if session is None or tokenizer is None:  # pragma: no cover - load() guarantees both
            raise ArtifactError("OnnxSentimentSession.predict_proba: session not initialized.")

        # Enable right-padding so the batch tokenizes to one rectangular tensor.
        tokenizer.enable_padding()  # type: ignore[attr-defined]
        encodings = tokenizer.encode_batch(batch)  # type: ignore[attr-defined]
        input_ids = np.asarray([enc.ids for enc in encodings], dtype=np.int64)
        attention_mask = np.asarray([enc.attention_mask for enc in encodings], dtype=np.int64)

        # Build the feed dict by the session's declared input names so the export's
        # signature (input_ids / attention_mask) is honoured without hard-coding order.
        feed: dict[str, NDArray[np.int64]] = {}
        available = {"input_ids": input_ids, "attention_mask": attention_mask}
        for spec in session.get_inputs():  # type: ignore[attr-defined]
            if spec.name in available:
                feed[spec.name] = available[spec.name]
        if "input_ids" not in feed:
            raise ArtifactError(
                "OnnxSentimentSession.predict_proba: exported graph is missing an "
                "'input_ids' input; cannot run the forward pass."
            )

        try:
            outputs = session.run(None, feed)  # type: ignore[attr-defined]
        except Exception as exc:  # normalize onnxruntime runtime errors
            raise ArtifactError(
                f"OnnxSentimentSession.predict_proba: onnxruntime forward pass failed "
                f"(check the input signature matches the exported graph): {exc}"
            ) from exc

        logits = np.asarray(outputs[0], dtype=np.float64)
        if logits.ndim != 2 or logits.shape[1] != N_CLASSES:
            raise ArtifactError(
                f"OnnxSentimentSession.predict_proba: expected a "
                f"(n_texts, {N_CLASSES}) logit matrix, got shape {logits.shape}."
            )
        # Numerically stable row-wise softmax to class probabilities.
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        proba: NDArray[np.float64] = exp / exp.sum(axis=1, keepdims=True)
        return proba

    def predict(self, texts: Sequence[str]) -> NDArray[np.int64]:
        """Return the argmax class index per text (thin wrapper over scores).

        Parameters
        ----------
        texts:
            The batch to classify.

        Returns
        -------
        numpy.ndarray
            A length-``len(texts)`` ``int64`` vector of class indices.
        """
        import numpy as np

        proba = self.predict_proba(texts)
        return np.asarray(proba.argmax(axis=1), dtype=np.int64)
