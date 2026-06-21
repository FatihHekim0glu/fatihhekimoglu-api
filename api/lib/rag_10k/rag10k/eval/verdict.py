"""The PURE ``answer_is_grounded`` verdict.

:func:`answer_is_grounded` is a PURE function (no I/O, no model, no randomness)
that decides whether an answer is grounded: it is ``True`` ONLY IF the system did
NOT abstain AND every emitted claim maps to a cited chunk that (a) was retrieved
above the similarity threshold and (b) entails the claim under the deterministic
proxy; otherwise the answer is NOT grounded and the system ABSTAINS. This is the
single boolean the backend surfaces as ``summary.answer_is_grounded``.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rag10k.eval.citations import CitationReport


@dataclass(frozen=True, slots=True)
class GroundingVerdict:
    """Immutable grounding verdict for a single answer.

    Attributes
    ----------
    answer_is_grounded:
        The PURE verdict: every claim cites a retrieved, entailing chunk and the
        system did not abstain.
    abstained:
        Whether the system abstained (retrieval below threshold).
    citations_valid:
        Whether all citations were sound (no hallucinated citations).
    all_claims_entailed:
        Whether every answer claim was entailed by its cited chunk (the proxy).
    n_claims:
        Number of answer claims examined.
    extra:
        Optional extra provenance.
    """

    answer_is_grounded: bool
    abstained: bool
    citations_valid: bool
    all_claims_entailed: bool
    n_claims: int
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this verdict."""
        return asdict(self)


def answer_is_grounded(
    *,
    abstained: bool,
    citation_report: CitationReport,
    claim_entailments: list[bool],
) -> GroundingVerdict:
    """Derive the PURE grounding verdict from abstention + citations + entailments.

    The verdict is ``True`` ONLY IF: the system did NOT abstain, the citation
    report is valid (no hallucinated citations), there is at least one claim, and
    EVERY claim was entailed by its cited chunk. Any failure → not grounded (and
    the system should abstain). PURE: a deterministic function of its arguments,
    with no I/O.

    Parameters
    ----------
    abstained:
        Whether retrieval abstained (top score below threshold).
    citation_report:
        The :class:`rag10k.eval.citations.CitationReport` for the answer.
    claim_entailments:
        Per-claim entailment booleans (claim i is entailed by its cited chunk).

    Returns
    -------
    GroundingVerdict
        The pure verdict bundle.
    """
    n_claims = len(claim_entailments)
    citations_valid = bool(citation_report.valid)
    # ``all()`` of an empty list is vacuously True; require >= 1 claim explicitly so
    # an empty answer is NOT counted as grounded (nothing was actually supported).
    all_claims_entailed = n_claims > 0 and all(claim_entailments)

    # The verdict is the conjunction: did not abstain AND citations are sound AND
    # there is at least one claim AND every claim is entailed by its cited chunk.
    # Any failure → not grounded (and the system should abstain).
    grounded = (not abstained) and citations_valid and all_claims_entailed

    return GroundingVerdict(
        answer_is_grounded=grounded,
        abstained=bool(abstained),
        citations_valid=citations_valid,
        all_claims_entailed=all_claims_entailed,
        n_claims=n_claims,
        extra={"n_invalid_citations": len(citation_report.invalid_chunk_ids)},
    )
