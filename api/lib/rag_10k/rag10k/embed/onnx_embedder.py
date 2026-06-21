"""ONNX sentence-transformer embedder — the SERVE path (onnxruntime + tokenizers, NO torch).

The container and the FastAPI router embed both the query and any freshly-ingested
chunks through this module ONLY. It loads the committed
``artifacts/embedder.int8.onnx`` graph with onnxruntime and the
``artifacts/tokenizer.json`` with the ``tokenizers`` library; torch and
sentence-transformers are NEVER imported here. Both heavy imports happen LAZILY
inside :meth:`OnnxEmbedder.load`, so ``import rag10k`` stays free of any inference
engine.

The embedder mean-pools the token embeddings under the attention mask and
L2-normalizes each row, matching the reference ``sentence-transformers`` pooling
so the ONNX output is within :data:`rag10k._constants.ONNX_PARITY_ATOL` of torch.

Importing this module has no side effects.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rag10k._constants import EMBEDDING_DIM

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rag10k._typing import EmbeddingMatrix, EmbeddingVector

#: Directory holding the committed, shipped serve artifacts (ONNX + tokenizer).
ARTIFACTS_DIR: Path = Path(__file__).resolve().parent.parent / "artifacts"

#: Filenames of the committed serve artifacts (int8-quantized to stay <100MB).
ONNX_ARTIFACT_NAME: str = "embedder.int8.onnx"
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


def embedder_artifacts_present(artifact_dir: str | Path | None = None) -> bool:
    """Return ``True`` iff BOTH the ONNX embedder and the tokenizer.json exist.

    Used by the serve path to decide whether the ONNX embedder is available. Pure
    filesystem check — imports nothing heavy.

    Parameters
    ----------
    artifact_dir:
        Directory to probe (defaults to :func:`default_artifact_dir`).

    Returns
    -------
    bool
        Whether both serve artifacts are present.
    """
    directory = Path(artifact_dir) if artifact_dir else default_artifact_dir()
    onnx_ok = (directory / ONNX_ARTIFACT_NAME).is_file()
    tokenizer_ok = (directory / TOKENIZER_ARTIFACT_NAME).is_file()
    return onnx_ok and tokenizer_ok


class OnnxEmbedder:
    """A thin onnxruntime + tokenizers wrapper that serves the committed embedder.

    The onnxruntime session and the tokenizer are created LAZILY on first
    :meth:`embed` (or via :meth:`load`), so constructing this object is cheap and
    import-pure. torch / sentence-transformers are never imported on this path.
    """

    def __init__(self, artifact_dir: str | Path | None = None) -> None:
        """Record the artifact directory; defer session/tokenizer creation to :meth:`load`.

        Parameters
        ----------
        artifact_dir:
            Directory holding ``embedder.int8.onnx`` + ``tokenizer.json``.
            Defaults to the package's shipped artifacts (:func:`default_artifact_dir`).
        """
        self._artifact_dir = Path(artifact_dir) if artifact_dir else default_artifact_dir()
        self._session: object | None = None
        self._tokenizer: object | None = None

    @property
    def artifact_dir(self) -> Path:
        """Return the resolved artifact directory this embedder will load from."""
        return self._artifact_dir

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimensionality this embedder produces."""
        return EMBEDDING_DIM

    def load(self) -> OnnxEmbedder:
        """Create the onnxruntime session and load the tokenizer (lazy, idempotent).

        LAZY IMPORT: ``onnxruntime`` and ``tokenizers`` are imported inside this
        method. NO torch / sentence-transformers import occurs anywhere on this
        path.

        Returns
        -------
        OnnxEmbedder
            ``self``, with an initialized session and tokenizer.

        Raises
        ------
        ArtifactError
            If either artifact is missing or initialization fails.
        """
        if self._session is not None and self._tokenizer is not None:
            return self

        from rag10k._exceptions import ArtifactError

        onnx_path = self._artifact_dir / ONNX_ARTIFACT_NAME
        tokenizer_path = self._artifact_dir / TOKENIZER_ARTIFACT_NAME
        if not onnx_path.is_file():
            raise ArtifactError(
                f"OnnxEmbedder.load: ONNX embedder artifact not found at {onnx_path}."
            )
        if not tokenizer_path.is_file():
            raise ArtifactError(
                f"OnnxEmbedder.load: tokenizer.json not found at {tokenizer_path}."
            )

        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            self._session = ort.InferenceSession(
                str(onnx_path),
                providers=["CPUExecutionProvider"],
            )
            self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        except ArtifactError:  # pragma: no cover - defensive: re-raise our own errors verbatim
            raise
        except Exception as exc:  # normalize any onnxruntime/tokenizers error to ArtifactError
            raise ArtifactError(
                f"OnnxEmbedder.load: failed to initialize the onnxruntime session or "
                f"tokenizer from {self._artifact_dir}: {exc}"
            ) from exc
        return self

    def embed(self, texts: Sequence[str]) -> EmbeddingMatrix:
        """Embed a batch of texts to L2-normalized rows (mean-pooled, torch-free).

        Tokenizes the batch (with right-padding), runs the ONNX forward pass,
        mean-pools the token embeddings under the attention mask, and L2-normalizes
        each row — matching the reference sentence-transformers pooling so the
        output is within :data:`rag10k._constants.ONNX_PARITY_ATOL` of torch.

        Parameters
        ----------
        texts:
            The batch to embed (1..N non-empty strings).

        Returns
        -------
        EmbeddingMatrix
            A ``(len(texts), EMBEDDING_DIM)`` ``float32`` matrix of unit-norm rows.

        Raises
        ------
        ArtifactError
            If the session/tokenizer cannot be loaded or the forward pass fails.
        ValidationError
            If ``texts`` is empty or contains a non-string / empty entry.
        """
        import numpy as np

        from rag10k._constants import EPS
        from rag10k._exceptions import ArtifactError

        batch = _ensure_text_batch(texts)
        self.load()
        session = self._session
        tokenizer = self._tokenizer
        if session is None or tokenizer is None:  # pragma: no cover - load() guarantees both
            raise ArtifactError("OnnxEmbedder.embed: session/tokenizer not initialized.")

        # Right-pad the batch so it tokenizes to one rectangular tensor; the padded
        # positions are masked out of the mean pool below.
        tokenizer.enable_padding()  # type: ignore[attr-defined]
        encodings = tokenizer.encode_batch(batch)  # type: ignore[attr-defined]
        input_ids = np.asarray([enc.ids for enc in encodings], dtype=np.int64)
        attention_mask = np.asarray([enc.attention_mask for enc in encodings], dtype=np.int64)

        # Feed by the session's declared input names so the export's signature
        # (input_ids / attention_mask / optional token_type_ids) is honoured without
        # hard-coding feed order.
        available: dict[str, object] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": np.zeros_like(input_ids),
        }
        feed: dict[str, object] = {}
        for spec in session.get_inputs():  # type: ignore[attr-defined]
            if spec.name in available:
                feed[spec.name] = available[spec.name]
        if "input_ids" not in feed:
            raise ArtifactError(
                "OnnxEmbedder.embed: exported graph is missing an 'input_ids' input; "
                "cannot run the forward pass."
            )

        try:
            outputs = session.run(None, feed)  # type: ignore[attr-defined]
        except Exception as exc:  # normalize onnxruntime runtime errors to ArtifactError
            raise ArtifactError(
                f"OnnxEmbedder.embed: onnxruntime forward pass failed (check the input "
                f"signature matches the exported graph): {exc}"
            ) from exc

        token_embeddings = np.asarray(outputs[0], dtype=np.float32)
        if token_embeddings.ndim != 3 or token_embeddings.shape[2] != EMBEDDING_DIM:
            raise ArtifactError(
                f"OnnxEmbedder.embed: expected a (n_texts, seq_len, {EMBEDDING_DIM}) "
                f"token-embedding tensor, got shape {token_embeddings.shape}."
            )

        # Mean-pool the token embeddings under the attention mask (the canonical
        # sentence-transformers pooling), then L2-normalize each row so a cosine
        # top-k is a plain dot product. Padded positions contribute nothing.
        mask = attention_mask.astype(np.float32)[:, :, None]
        summed = (token_embeddings * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), EPS, None)
        pooled = summed / counts
        norms = np.sqrt((pooled * pooled).sum(axis=1, keepdims=True))
        normalized = pooled / np.clip(norms, EPS, None)
        return np.ascontiguousarray(normalized, dtype=np.float32)

    def embed_one(self, text: str) -> EmbeddingVector:
        """Embed a single text to one L2-normalized vector (thin wrapper over :meth:`embed`).

        Parameters
        ----------
        text:
            The query (or chunk) text to embed.

        Returns
        -------
        EmbeddingVector
            A ``(EMBEDDING_DIM,)`` ``float32`` unit-norm vector.

        Raises
        ------
        ArtifactError
            If the embedder cannot be loaded or the forward pass fails.
        ValidationError
            If ``text`` is empty.
        """
        from rag10k._exceptions import ValidationError

        if not isinstance(text, str) or not text.strip():
            raise ValidationError("OnnxEmbedder.embed_one: text must be a non-empty string.")
        vector: EmbeddingVector = self.embed([text])[0]
        return vector


def _ensure_text_batch(texts: Sequence[str]) -> list[str]:
    """Coerce ``texts`` to a non-empty list of non-empty strings (serve-path guard).

    Inlined (numpy-free) so the embedder validates its own batch without coupling
    to other groups' validation helpers. Whitespace-only / non-string entries are
    rejected so the tokenizer never sees a degenerate row.

    Raises
    ------
    ValidationError
        If ``texts`` is empty or any entry is not a non-empty string.
    """
    from rag10k._exceptions import ValidationError

    batch = list(texts)
    if not batch:
        raise ValidationError("OnnxEmbedder.embed: texts must contain at least one string.")
    cleaned: list[str] = []
    for i, item in enumerate(batch):
        if not isinstance(item, str) or not item.strip():
            raise ValidationError(
                f"OnnxEmbedder.embed: texts[{i}] must be a non-empty string, got {item!r}."
            )
        cleaned.append(item)
    return cleaned
