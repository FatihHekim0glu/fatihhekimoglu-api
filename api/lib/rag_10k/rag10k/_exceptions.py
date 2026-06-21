"""Typed exception hierarchy for the rag10k library.

A single base (:class:`Rag10kError`) lets callers catch any library-raised error
with one ``except`` clause, while the specific subclasses let them distinguish
data-shape problems (validation), retrieval-engine problems, citation-soundness
problems, and missing-artifact / model-load problems. Importing this module has
no side effects.
"""

from __future__ import annotations

# quantcore-candidate: mirrors hrp-portfolio:src/hrp/_exceptions.py


class Rag10kError(Exception):
    """Base class for every exception raised by :mod:`rag10k`.

    Catching ``Rag10kError`` catches all library-specific failures while letting
    unrelated exceptions (e.g. ``KeyboardInterrupt``) propagate.
    """


class ValidationError(Rag10kError):
    """Raised when an input fails a shape, dtype, alignment, or domain check.

    Examples: an empty question, an unknown ticker, a non-positive ``top_k``, a
    chunk whose char-span is out of bounds, or a query/chunk embedding whose
    dimensionality does not match the committed index.
    """


class InsufficientDataError(ValidationError):
    """Raised when there is too little data to perform the requested operation.

    For example, an empty corpus index, a filing that parses to zero chunks, or
    an eval set with fewer questions than the harness requires. It subclasses
    :class:`ValidationError` because "not enough data" is a special case of a
    failed input precondition.
    """


class IngestError(Rag10kError):
    """Raised when an EDGAR fetch fails (network, HTTP status, or empty payload).

    The deployed endpoint never surfaces this to the caller: it catches
    ``IngestError`` and degrades to the cached/synthetic 10-K fixture. The
    library raises it so the fallback decision is made by the caller, not buried.
    """


class ParseError(Rag10kError):
    """Raised when a 10-K document cannot be parsed into the expected sections.

    For example, no anchored ``Item 1A``/``Item 7`` header is found and no
    fallback heuristic applies. Distinguishes a structural parse failure from a
    generic validation failure.
    """


class RetrievalError(Rag10kError):
    """Raised when the retrieval engine cannot serve a query.

    For example, a committed index that fails to load, a query embedding whose
    dimensionality is incompatible with the index, or a ``top_k`` request against
    an empty index. Distinct from :class:`ValidationError` (a malformed request)
    because the request was well-formed but the engine could not answer it.
    """


class CitationError(Rag10kError):
    """Raised when an answer's citations fail soundness validation.

    The headline correctness guard: every cited chunk id MUST exist in the
    retrieved set and be returnable with its provenance. A fabricated /
    non-retrieved citation raises this error rather than silently passing — there
    are NO hallucinated citations.
    """


class ArtifactError(Rag10kError):
    """Raised when a shipped serve artifact cannot be located, loaded, or run.

    Reserved for the serve path: a missing ``artifacts/embedder.onnx`` graph or
    ``artifacts/tokenizer.json``, a corrupt model, an onnxruntime session that
    fails to initialize, or an input whose shape does not match the exported
    graph's expected signature. The FastAPI router maps this to a 502
    (artifact-load failure), distinct from the 422 raised for request
    :class:`ValidationError`.
    """


class GenerationError(Rag10kError):
    """Raised when the (key-gated) LLM generation step fails irrecoverably.

    The deployed endpoint never surfaces this: an absent key or a transient LLM
    failure degrades to the KEY-FREE extractive answer. The library raises it so
    the fallback decision is made by the caller, not buried.
    """
