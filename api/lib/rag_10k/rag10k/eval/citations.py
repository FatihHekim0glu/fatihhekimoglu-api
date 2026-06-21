"""Citation validation — every cited chunk id MUST exist and be retrievable.

The headline correctness guard against hallucinated citations: given an answer's
cited chunk ids and the retrieved set (and the backing index), assert that every
cited id (1) appears in the retrieved set for this query and (2) resolves to a real
chunk with returnable provenance in the index. A citation that fails either check
is a hallucinated/fabricated citation and is REJECTED (raises
:class:`rag10k._exceptions.CitationError`) — the property test asserts a fabricated
citation is rejected.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rag10k.retrieve.index import CorpusIndex
    from rag10k.retrieve.search import RetrievedChunk


@dataclass(frozen=True, slots=True)
class CitationReport:
    """Immutable result of validating an answer's citations.

    Attributes
    ----------
    valid:
        ``True`` iff every cited id is in the retrieved set AND resolves in the
        index (no hallucinated citations).
    cited_chunk_ids:
        The chunk ids the answer cited.
    retrieved_chunk_ids:
        The chunk ids actually retrieved for this query.
    invalid_chunk_ids:
        Cited ids that are NOT in the retrieved set (the fabricated ones).
    n_citations:
        Number of (valid) citations.
    extra:
        Optional extra provenance.
    """

    valid: bool
    cited_chunk_ids: list[str]
    retrieved_chunk_ids: list[str]
    invalid_chunk_ids: list[str]
    n_citations: int
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this report."""
        return asdict(self)


def validate_citations(
    cited_chunk_ids: Sequence[str],
    retrieved: Sequence[RetrievedChunk],
    index: CorpusIndex,
    *,
    strict: bool = True,
) -> CitationReport:
    """Validate that every cited chunk id is retrieved AND resolves in the index.

    For each cited id: it must be present in ``retrieved`` (cited from what was
    actually shown to the answerer) and must resolve via ``index.get_chunk`` (a
    real chunk with returnable provenance). Any id failing either check is a
    fabricated citation.

    Parameters
    ----------
    cited_chunk_ids:
        The chunk ids the answer cited.
    retrieved:
        The retrieved chunks shown to the answerer (the only valid citation set).
    index:
        The backing index, used to confirm each id resolves to a real chunk.
    strict:
        If ``True`` (default), a fabricated citation raises
        :class:`rag10k._exceptions.CitationError`; if ``False``, it is recorded in
        ``invalid_chunk_ids`` and ``valid`` is ``False`` without raising.

    Returns
    -------
    CitationReport
        The validation report.

    Raises
    ------
    CitationError
        If ``strict`` and any cited id is not retrieved or does not resolve.
    """
    from rag10k._exceptions import CitationError

    cited = list(cited_chunk_ids)
    # The retrieved set is the ONLY legitimate citation source: a citation must come
    # from what was actually shown to the answerer.
    retrieved_ids = [hit.chunk.chunk_id for hit in retrieved]
    retrieved_set = set(retrieved_ids)

    invalid: list[str] = []
    valid_ids: list[str] = []
    for chunk_id in cited:
        in_retrieved = chunk_id in retrieved_set
        resolves = in_retrieved and _resolves_in_index(index, chunk_id)
        if in_retrieved and resolves:
            valid_ids.append(chunk_id)
        else:
            invalid.append(chunk_id)

    valid = not invalid

    if strict and invalid:
        raise CitationError(
            "validate_citations: fabricated / non-retrievable citation(s) "
            f"{invalid!r} — every cited chunk id must be in the retrieved set and "
            "resolve in the index (no hallucinated citations)."
        )

    return CitationReport(
        valid=valid,
        cited_chunk_ids=cited,
        retrieved_chunk_ids=retrieved_ids,
        invalid_chunk_ids=invalid,
        n_citations=len(valid_ids),
        extra={"strict": strict},
    )


def _resolves_in_index(index: CorpusIndex, chunk_id: str) -> bool:
    """Return ``True`` iff ``chunk_id`` resolves to a real chunk in ``index``.

    ``index.get_chunk`` raises :class:`rag10k._exceptions.CitationError` for an id
    that is not present, so a fabricated id is caught here and reported (not raised)
    — :func:`validate_citations` owns the strict-mode raise decision.
    """
    from rag10k._exceptions import CitationError

    try:
        index.get_chunk(chunk_id)
    except CitationError:
        return False
    return True
