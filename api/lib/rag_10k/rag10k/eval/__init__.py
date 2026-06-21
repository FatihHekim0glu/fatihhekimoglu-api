"""Evaluation: the frozen 150-Q harness + citation validation + the pure verdict.

:mod:`rag10k.eval.harness` runs the FROZEN 150-question eval over the committed
corpus (retrieval recall@k, citation-soundness, abstention rate on out-of-document
questions). :mod:`rag10k.eval.citations` validates that every cited chunk id EXISTS
and is retrievable (NO hallucinated citations). :mod:`rag10k.eval.verdict` derives
the PURE ``answer_is_grounded`` boolean.

The harness, the entailment proxy, and the verdict are all DETERMINISTIC — no
model, no network, no LLM key. Importing this package has no side effects.
"""

from __future__ import annotations

from rag10k.eval.citations import CitationReport, validate_citations
from rag10k.eval.harness import (
    EvalQuestion,
    EvalSummary,
    entailment_proxy,
    recall_at_k,
    run_harness,
)
from rag10k.eval.verdict import GroundingVerdict, answer_is_grounded

__all__ = [
    "CitationReport",
    "EvalQuestion",
    "EvalSummary",
    "GroundingVerdict",
    "answer_is_grounded",
    "entailment_proxy",
    "recall_at_k",
    "run_harness",
    "validate_citations",
]
