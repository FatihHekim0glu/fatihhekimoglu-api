"""Query embedding → cosine top-k → similarity-threshold ABSTENTION gate.

:func:`search` is the retrieval entrypoint the serve path calls: embed the query
with the committed ONNX embedder (torch-free), run a cosine top-k over the
:class:`rag10k.retrieve.index.CorpusIndex`, and decide — purely from the top
score — whether to ABSTAIN. When the top retrieved chunk's cosine score is below
:data:`rag10k._constants.ABSTENTION_THRESHOLD`, the result is flagged ``abstain``
and downstream MUST return "not found in the filing" rather than answer. Retrieval
is deterministic (seeded embedder + deterministic top-k).

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rag10k._constants import ABSTENTION_THRESHOLD, DEFAULT_TOP_K

if TYPE_CHECKING:
    from rag10k.chunk.splitter import Chunk
    from rag10k.embed.onnx_embedder import OnnxEmbedder
    from rag10k.retrieve.index import CorpusIndex


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """An immutable retrieved chunk + its cosine score (a citation candidate).

    Attributes
    ----------
    chunk:
        The retrieved :class:`rag10k.chunk.splitter.Chunk` (carries provenance).
    score:
        Its cosine similarity to the query (``[-1, 1]``; unit-norm rows).
    rank:
        Its 0-based rank in the top-k (0 = most similar).
    """

    chunk: Chunk
    score: float
    rank: int

    def to_citation(self) -> dict[str, Any]:
        """Return the JSON citation dict (chunk_id, item, char_span, score, snippet).

        Mirrors the backend ``citations`` list element: ``{chunk_id, item,
        char_span, score, text_snippet}``.
        """
        return {
            "chunk_id": self.chunk.chunk_id,
            "item": self.chunk.item,
            "char_span": list(self.chunk.char_span),
            "score": float(self.score),
            "text_snippet": self.chunk.text,
        }


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Immutable result of a retrieval: the ranked chunks + the abstention decision.

    Attributes
    ----------
    query:
        The (normalized) query text.
    retrieved:
        The ranked :class:`RetrievedChunk` s (descending score).
    top_score:
        The cosine score of the top chunk (0.0 when nothing retrieved).
    abstain:
        ``True`` iff ``top_score < ABSTENTION_THRESHOLD`` (or nothing retrieved) —
        the system MUST NOT answer.
    threshold:
        The abstention threshold used (for transparency in the response).
    extra:
        Optional extra provenance.
    """

    query: str
    retrieved: list[RetrievedChunk]
    top_score: float
    abstain: bool
    threshold: float = ABSTENTION_THRESHOLD
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        return {
            "query": self.query,
            "retrieved": [hit.to_citation() for hit in self.retrieved],
            "top_score": float(self.top_score),
            "abstain": bool(self.abstain),
            "threshold": float(self.threshold),
            "extra": dict(self.extra),
        }


def search(
    query: str,
    index: CorpusIndex,
    embedder: OnnxEmbedder,
    *,
    top_k: int = DEFAULT_TOP_K,
    threshold: float = ABSTENTION_THRESHOLD,
) -> RetrievalResult:
    """Embed ``query``, retrieve top-k chunks, and apply the abstention gate.

    Normalizes and embeds the query with ``embedder`` (ONNX, torch-free), runs
    ``index.cosine_topk``, wraps each hit as a :class:`RetrievedChunk`, and sets
    ``abstain = top_score < threshold`` (also ``True`` when the index returns
    nothing). The decision is a PURE function of the top score and the threshold.

    Parameters
    ----------
    query:
        The user question.
    index:
        The committed read-only corpus index to search.
    embedder:
        The ONNX embedder used to embed the query.
    top_k:
        Number of chunks to retrieve (capped at the index size / ``MAX_TOP_K``).
    threshold:
        The cosine abstention floor.

    Returns
    -------
    RetrievalResult
        The ranked chunks and the abstention decision.

    Raises
    ------
    ValidationError
        If ``query`` is empty or ``top_k <= 0``.
    RetrievalError
        If the index is empty or the embedder fails.
    """
    from rag10k._constants import MAX_TOP_K
    from rag10k._exceptions import RetrievalError, ValidationError

    if not isinstance(query, str) or not query.strip():
        raise ValidationError("search: query must be a non-empty string.")
    if top_k <= 0:
        raise ValidationError(f"search: top_k must be > 0, got {top_k}.")
    if index.n_chunks == 0:  # pragma: no cover - CorpusIndex forbids an empty index
        raise RetrievalError("search: the corpus index is empty.")

    normalized_query = query.strip()
    capped_top_k = min(top_k, MAX_TOP_K, index.n_chunks)

    # Embed the query with the (torch-free) ONNX embedder; normalize any embedder
    # failure to RetrievalError so the engine surfaces a single, well-formed error
    # rather than leaking an onnxruntime/tokenizers exception.
    try:
        query_vector = embedder.embed_one(normalized_query)
    except ValidationError:  # pragma: no cover - guarded above; defensive
        raise
    except Exception as exc:
        raise RetrievalError(f"search: query embedding failed: {exc}") from exc

    rows, scores = index.cosine_topk(query_vector, top_k=capped_top_k)

    retrieved: list[RetrievedChunk] = [
        RetrievedChunk(chunk=index.chunk_at(int(row)), score=float(score), rank=rank)
        for rank, (row, score) in enumerate(zip(rows.tolist(), scores.tolist(), strict=True))
    ]

    top_score = float(retrieved[0].score) if retrieved else 0.0
    # The abstention decision is a PURE function of the top score and the threshold:
    # below the floor (or nothing retrieved) the system MUST NOT answer.
    abstain = (not retrieved) or top_score < threshold

    return RetrievalResult(
        query=normalized_query,
        retrieved=retrieved,
        top_score=top_score,
        abstain=abstain,
        threshold=threshold,
    )
