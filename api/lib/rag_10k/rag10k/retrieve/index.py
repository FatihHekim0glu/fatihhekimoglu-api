"""Committed read-only vector index (chunk vectors + text + provenance).

:class:`CorpusIndex` wraps the prebuilt ``artifacts/corpus_index.npz`` artifact:
an ``(n_chunks, EMBEDDING_DIM)`` ``float32`` matrix of L2-normalized chunk vectors
alongside the per-chunk text and provenance (chunk_id / cik / accession / item /
char-span). It is loaded LAZILY and READ-ONLY — the serve path never mutates it —
and exposes a cosine top-k (a plain dot product on unit-norm rows) plus chunk
lookup by id for citation validation.

The index can also be built in-memory from a list of chunks + a vector matrix (the
offline builder and the test fixtures use this), so a committed ``.npz`` is not
required to exercise retrieval in tests.

Importing this module has no side effects (numpy is imported lazily inside the
methods to keep the module import-pure under the lean serve extras).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rag10k._constants import DEFAULT_TOP_K, EMBEDDING_DIM, EPS

if TYPE_CHECKING:
    import numpy as np

    from rag10k._typing import EmbeddingMatrix, EmbeddingVector, IndexArray, ScoreArray
    from rag10k.chunk.splitter import Chunk

#: Directory holding the committed, shipped index artifact.
ARTIFACTS_DIR: Path = Path(__file__).resolve().parent.parent / "artifacts"

#: Filename of the committed read-only corpus index (the default bundle).
INDEX_ARTIFACT_NAME: str = "corpus_index.npz"


def _index_filename(ticker: str | None) -> str:
    """Return the index filename for ``ticker`` (per-ticker file or default bundle).

    Pure string arithmetic: an uppercased, alnum-only ticker selects a
    ``corpus_index_<TICKER>.npz`` file; ``None``/blank selects the default bundle.
    """
    if ticker is None:
        return INDEX_ARTIFACT_NAME
    slug = "".join(ch for ch in ticker.upper() if ch.isalnum())
    if not slug:
        return INDEX_ARTIFACT_NAME
    return f"corpus_index_{slug}.npz"


def default_index_path(ticker: str | None = None) -> Path:
    """Return the committed index path for a ticker (pure path arithmetic).

    A single committed corpus may host several filings; ``ticker`` selects the
    per-ticker index file when present, else the default bundle. Does NOT check
    existence and imports nothing heavy.

    Parameters
    ----------
    ticker:
        Optional ticker selecting a per-ticker index file.

    Returns
    -------
    pathlib.Path
        The resolved index artifact path.
    """
    return ARTIFACTS_DIR / _index_filename(ticker)


def index_present(ticker: str | None = None, *, artifact_dir: str | Path | None = None) -> bool:
    """Return ``True`` iff the committed index artifact exists for ``ticker``.

    Pure filesystem check — imports nothing heavy.

    Parameters
    ----------
    ticker:
        Optional ticker selecting a per-ticker index file.
    artifact_dir:
        Directory to probe (defaults to :data:`ARTIFACTS_DIR`).

    Returns
    -------
    bool
        Whether the index artifact is present.
    """
    directory = Path(artifact_dir) if artifact_dir is not None else ARTIFACTS_DIR
    return (directory / _index_filename(ticker)).is_file()


class CorpusIndex:
    """A committed, read-only cosine index over a corpus of 10-K chunks.

    Holds the L2-normalized chunk-vector matrix and the aligned chunk metadata.
    Cosine top-k is a dot product (rows are unit-norm). Build via :meth:`load`
    (from the committed ``.npz``) or :meth:`from_chunks` (in-memory, for the
    offline builder and tests).
    """

    def __init__(self, vectors: EmbeddingMatrix, chunks: list[Chunk]) -> None:
        """Wrap a vector matrix and its aligned chunk metadata.

        Parameters
        ----------
        vectors:
            An ``(n_chunks, EMBEDDING_DIM)`` ``float32`` matrix of L2-normalized
            chunk vectors.
        chunks:
            The ``n_chunks`` :class:`rag10k.chunk.splitter.Chunk` objects aligned
            row-for-row with ``vectors``.

        Raises
        ------
        ValidationError
            If ``vectors`` and ``chunks`` disagree in length, or ``vectors`` has
            the wrong dimensionality.
        """
        import numpy as np

        from rag10k._exceptions import ValidationError

        matrix = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
        if matrix.ndim != 2 or matrix.shape[1] != EMBEDDING_DIM:
            raise ValidationError(
                f"CorpusIndex: vectors must be a 2-D (n_chunks, {EMBEDDING_DIM}) float32 "
                f"matrix, got shape {matrix.shape}."
            )
        if matrix.shape[0] != len(chunks):
            raise ValidationError(
                f"CorpusIndex: vectors ({matrix.shape[0]} rows) and chunks "
                f"({len(chunks)}) disagree in length."
            )
        self._vectors: np.ndarray = matrix
        self._chunks: list[Chunk] = list(chunks)
        # Build the chunk_id -> row map once so citation lookup is O(1) and so a
        # fabricated citation is rejected without scanning the corpus.
        self._id_to_row: dict[str, int] = {chunk.chunk_id: row for row, chunk in enumerate(self._chunks)}

    @property
    def n_chunks(self) -> int:
        """Return the number of chunks in the index."""
        return len(self._chunks)

    @property
    def vectors(self) -> EmbeddingMatrix:
        """Return the ``(n_chunks, EMBEDDING_DIM)`` L2-normalized vector matrix."""
        return self._vectors

    @classmethod
    def load(
        cls,
        ticker: str | None = None,
        *,
        artifact_dir: str | Path | None = None,
    ) -> CorpusIndex:
        """Load the committed read-only index for ``ticker`` from its ``.npz`` artifact.

        LAZY: numpy is imported inside this method; the artifact is memory-mapped
        read-only. The chunk metadata is rehydrated into :class:`Chunk` objects
        aligned with the vector rows.

        Parameters
        ----------
        ticker:
            Optional ticker selecting a per-ticker index file.
        artifact_dir:
            Directory to load from (defaults to :data:`ARTIFACTS_DIR`).

        Returns
        -------
        CorpusIndex
            The loaded read-only index.

        Raises
        ------
        ArtifactError
            If the index artifact is missing or cannot be parsed.
        """
        import numpy as np

        from rag10k._exceptions import ArtifactError
        from rag10k.chunk.splitter import Chunk

        directory = Path(artifact_dir) if artifact_dir is not None else ARTIFACTS_DIR
        index_path = directory / _index_filename(ticker)
        if not index_path.is_file():
            raise ArtifactError(
                f"CorpusIndex.load: index artifact not found at {index_path}."
            )

        try:
            # ``mmap_mode='r'`` keeps the (potentially large) vector matrix read-only
            # and out of the heap; the object metadata arrays need pickle (committed
            # by the trusted offline builder, never user input).
            with np.load(index_path, allow_pickle=True, mmap_mode="r") as npz:
                vectors = np.ascontiguousarray(np.asarray(npz["vectors"], dtype=np.float32))
                chunk_ids = npz["chunk_ids"]
                ciks = npz["ciks"]
                accessions = npz["accessions"]
                items = npz["items"]
                texts = npz["texts"]
                ordinals = npz["ordinals"]
                char_starts = npz["char_starts"]
                char_ends = npz["char_ends"]
        except ArtifactError:  # pragma: no cover - defensive: re-raise our own errors verbatim
            raise
        except Exception as exc:  # normalize any numpy/IO error to ArtifactError
            raise ArtifactError(
                f"CorpusIndex.load: failed to read index artifact {index_path}: {exc}"
            ) from exc

        chunks = [
            Chunk(
                chunk_id=str(chunk_ids[i]),
                cik=str(ciks[i]),
                accession_no=str(accessions[i]),
                item=str(items[i]),
                char_span=(int(char_starts[i]), int(char_ends[i])),
                text=str(texts[i]),
                ordinal=int(ordinals[i]),
            )
            for i in range(len(chunk_ids))
        ]
        return cls(vectors, chunks)

    @classmethod
    def from_chunks(cls, chunks: list[Chunk], vectors: EmbeddingMatrix) -> CorpusIndex:
        """Build an in-memory index from chunks + their (already-embedded) vectors.

        Used by the offline builder and the test fixtures so retrieval can be
        exercised without a committed ``.npz``. The vectors are L2-normalized if
        not already unit-norm.

        Parameters
        ----------
        chunks:
            The corpus chunks.
        vectors:
            Their ``(n_chunks, EMBEDDING_DIM)`` embedding matrix.

        Returns
        -------
        CorpusIndex
            The in-memory index.

        Raises
        ------
        ValidationError
            If the inputs disagree in length or dimensionality.
        InsufficientDataError
            If ``chunks`` is empty.
        """
        from rag10k._exceptions import InsufficientDataError

        if not chunks:
            raise InsufficientDataError("CorpusIndex.from_chunks: chunks must be non-empty.")
        normalized = _l2_normalize_rows(vectors)
        return cls(normalized, list(chunks))

    def cosine_topk(self, query_vector: EmbeddingVector, *, top_k: int = DEFAULT_TOP_K) -> tuple[IndexArray, ScoreArray]:
        """Return the ``top_k`` chunk row indices and cosine scores for a query.

        Computes ``vectors @ query`` (rows are unit-norm, so this is cosine
        similarity), then selects the ``top_k`` highest scores in descending order.
        Deterministic; ties broken by ascending row index for reproducibility.

        Parameters
        ----------
        query_vector:
            The L2-normalized ``(EMBEDDING_DIM,)`` query embedding.
        top_k:
            Number of chunks to return (capped at ``n_chunks``).

        Returns
        -------
        tuple[IndexArray, ScoreArray]
            ``(row_indices, scores)`` sorted by descending score.

        Raises
        ------
        ValidationError
            If ``top_k <= 0`` or ``query_vector`` has the wrong dimensionality.
        RetrievalError
            If the index is empty.
        """
        import numpy as np

        from rag10k._exceptions import RetrievalError, ValidationError

        if top_k <= 0:
            raise ValidationError(f"cosine_topk: top_k must be > 0, got {top_k}.")
        if self.n_chunks == 0:  # pragma: no cover - __init__ forbids an empty index
            raise RetrievalError("cosine_topk: the index is empty.")

        query = np.ascontiguousarray(np.asarray(query_vector, dtype=np.float32)).reshape(-1)
        if query.shape[0] != EMBEDDING_DIM:
            raise ValidationError(
                f"cosine_topk: query_vector must have dimensionality {EMBEDDING_DIM}, "
                f"got {query.shape[0]}."
            )

        # Rows are unit-norm, so vectors @ query is the cosine similarity. Compute in
        # float64 for a stable, reproducible argsort.
        scores = (self._vectors.astype(np.float64) @ query.astype(np.float64))
        k = min(top_k, self.n_chunks)

        # Deterministic top-k: sort by descending score, breaking ties by ascending
        # row index. ``np.lexsort`` sorts by the last key primarily, so sort by
        # (-score) with row index as the tie-break (stable, ascending).
        order = np.lexsort((np.arange(self.n_chunks), -scores))[:k]
        top_indices: IndexArray = np.ascontiguousarray(order, dtype=np.int64)
        top_scores: ScoreArray = np.ascontiguousarray(scores[order], dtype=np.float64)
        return top_indices, top_scores

    def get_chunk(self, chunk_id: str) -> Chunk:
        """Return the chunk with ``chunk_id`` (for citation validation / lookup).

        Parameters
        ----------
        chunk_id:
            The deterministic chunk identifier.

        Returns
        -------
        Chunk
            The matching chunk.

        Raises
        ------
        CitationError
            If no chunk with ``chunk_id`` exists in the index (a fabricated /
            hallucinated citation).
        """
        from rag10k._exceptions import CitationError

        row = self._id_to_row.get(chunk_id)
        if row is None:
            raise CitationError(
                f"get_chunk: chunk_id {chunk_id!r} is not present in the index "
                "(a fabricated / non-retrievable citation)."
            )
        return self._chunks[row]

    def chunk_at(self, row: int) -> Chunk:
        """Return the chunk at vector row ``row`` (alignment helper for search).

        Parameters
        ----------
        row:
            The 0-based vector-matrix row index.

        Returns
        -------
        Chunk
            The chunk aligned with that row.

        Raises
        ------
        ValidationError
            If ``row`` is out of range.
        """
        from rag10k._exceptions import ValidationError

        if row < 0 or row >= self.n_chunks:
            raise ValidationError(
                f"chunk_at: row {row} is out of range [0, {self.n_chunks})."
            )
        return self._chunks[row]


def _l2_normalize_rows(vectors: EmbeddingMatrix) -> np.ndarray:
    """Return ``vectors`` as a ``float32`` matrix with each row scaled to unit L2 norm.

    A zero row is left unchanged (its norm is floored by :data:`rag10k._constants.EPS`
    to avoid a divide-by-zero), so cosine similarity against it is well-defined (zero).
    Inlined (numpy-only, lazy) so retrieval does not couple to other groups' helpers.
    """
    import numpy as np

    matrix = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
    if matrix.ndim != 2:  # pragma: no cover - defensive; __init__ rejects non-2-D downstream
        return matrix
    norms = np.sqrt((matrix * matrix).sum(axis=1, keepdims=True))
    normalized: np.ndarray = matrix / np.clip(norms, EPS, None)
    return np.ascontiguousarray(normalized, dtype=np.float32)
