"""Key-free extractive answer (the deployed default when no LLM key is set).

When no LLM key is present (the deployed default), the system answers EXTRACTIVELY:
it returns the top retrieved chunks verbatim as the answer, each tagged with its
citation, with NO generation. This keeps the retrieval/eval half fully demoable
without a key while remaining honestly grounded — every returned sentence is a
verbatim span of a retrieved chunk, so there are no hallucinated claims and the
citations are sound by construction.

The inline citation marker for chunk ``c`` is ``[chunk_id | item | start-end]`` —
it carries the provenance (chunk id, 10-K item slug, char span) needed to resolve
the cited snippet back to the filing. Because every span of the composed answer is
a verbatim slice of a retrieved chunk and the cited ids are exactly the included
chunks, citation-soundness holds by construction.

The extractive answer respects the same abstention contract: if retrieval
abstained (top score below threshold), the extractive answer is the abstention
message with no citations.

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag10k._constants import ABSTENTION_MESSAGE, DEFAULT_TOP_K
from rag10k._exceptions import ValidationError

if TYPE_CHECKING:
    from rag10k.generate.answer import GeneratedAnswer
    from rag10k.retrieve.search import RetrievalResult, RetrievedChunk

#: The model id stamped on an extractive answer (no LLM ran).
EXTRACTIVE_MODEL: str = "extractive"


def format_citation_marker(hit: RetrievedChunk) -> str:
    """Return the inline citation marker ``[chunk_id | item | start-end]`` for a hit.

    The marker carries the chunk id (the citation key), the 10-K item slug, and the
    ``start-end`` character span so a reader can resolve the cited span back to the
    filing. PURE and deterministic.

    Parameters
    ----------
    hit:
        The retrieved chunk to render a citation marker for.

    Returns
    -------
    str
        The inline marker, e.g. ``"[acc:mda:3 | mda | 384-512]"``.
    """
    chunk = hit.chunk
    start, end = chunk.char_span
    return f"[{chunk.chunk_id} | {chunk.item} | {start}-{end}]"


def extractive_answer(
    retrieval: RetrievalResult,
    *,
    max_chunks: int = DEFAULT_TOP_K,
) -> GeneratedAnswer:
    """Return the top retrieved chunks verbatim as a cited extractive answer.

    The KEY-FREE fallback. If ``retrieval.abstain`` is ``True`` the answer is the
    abstention message with an empty citation list. Otherwise the top
    ``max_chunks`` retrieved chunks are concatenated verbatim (each prefixed with
    its ``[chunk_id | item | char-span]`` citation marker) into the answer, and
    their chunk ids are recorded as the citations — so every answer span maps to a
    retrieved chunk and the citations are sound by construction.

    Parameters
    ----------
    retrieval:
        The :class:`rag10k.retrieve.search.RetrievalResult` to answer from.
    max_chunks:
        Maximum number of top chunks to include verbatim.

    Returns
    -------
    GeneratedAnswer
        An extractive :class:`rag10k.generate.answer.GeneratedAnswer`
        (``model="extractive"``) whose citations are exactly the included chunk
        ids (or empty when abstaining).

    Raises
    ------
    ValidationError
        If ``max_chunks <= 0``.
    """
    # Local import keeps the module import-pure and avoids a circular import at
    # module load (answer.py and extractive.py both live under generate/).
    from rag10k.generate.answer import GeneratedAnswer

    if max_chunks <= 0:
        raise ValidationError(f"extractive_answer: max_chunks must be > 0, got {max_chunks}.")

    # Honour the abstention contract: a below-threshold retrieval NEVER produces an
    # answer, only the canonical abstention message with no citations.
    if retrieval.abstain or not retrieval.retrieved:
        return GeneratedAnswer(
            answer=ABSTENTION_MESSAGE,
            cited_chunk_ids=[],
            abstained=True,
            model=EXTRACTIVE_MODEL,
            extra={"n_chunks_used": 0},
        )

    selected = retrieval.retrieved[:max_chunks]

    # Compose the answer from the chunk bodies VERBATIM, each prefixed with its
    # inline citation marker. Every span of the answer is thus a literal slice of a
    # retrieved chunk → no hallucinated claims, citations sound by construction.
    blocks = [f"{format_citation_marker(hit)} {hit.chunk.text.strip()}" for hit in selected]
    answer = "\n\n".join(blocks)

    # Cited ids in first-appearance (rank) order, de-duplicated while preserving order
    # (a chunk id cannot legitimately repeat across distinct top-k rows, but de-dup
    # defensively so the citation list mirrors `validate_citations` expectations).
    cited_chunk_ids: list[str] = []
    seen: set[str] = set()
    for hit in selected:
        chunk_id = hit.chunk.chunk_id
        if chunk_id not in seen:
            seen.add(chunk_id)
            cited_chunk_ids.append(chunk_id)

    return GeneratedAnswer(
        answer=answer,
        cited_chunk_ids=cited_chunk_ids,
        abstained=False,
        model=EXTRACTIVE_MODEL,
        extra={"n_chunks_used": len(selected)},
    )
