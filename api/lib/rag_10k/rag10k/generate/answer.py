"""Key-gated grounded LLM generation (lazy anthropic/openai; strict cite-or-abstain).

The LLM generation step. It is gated behind an ``ANTHROPIC_API_KEY`` /
``OPENAI_API_KEY`` env check: when no key is present the serve path NEVER calls
this and degrades to the key-free extractive answer. When a key IS present, the
client is constructed LAZILY (no anthropic/openai import at module load) and
driven by a STRICT grounded-answer prompt that:

- is given ONLY the retrieved chunk text (no parametric free-form context),
- MUST cite the chunk ids that support each answer sentence, and
- MUST output exactly "not found in the filing" when the retrieved context does
  not support an answer.

In tests the LLM is always MOCKED (no key, no network); the prompt-construction
and key-detection helpers are pure and unit-tested directly.

Importing this module has no side effects.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from rag10k._constants import ABSTENTION_MESSAGE, DEFAULT_TOP_K
from rag10k._exceptions import GenerationError, ValidationError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from rag10k.retrieve.search import RetrievalResult, RetrievedChunk

#: Env var overriding the default generation model id (so the deployed model can
#: be swapped without a code change). The brief's example default is a Haiku model.
MODEL_ENV_VAR: str = "RAG10K_LLM_MODEL"

#: Default LLM model id when :data:`MODEL_ENV_VAR` is unset. Configurable via env;
#: a small/cheap Anthropic model is sufficient for a strictly-grounded cite step.
DEFAULT_MODEL: str = "claude-haiku-4-5"

#: The env vars whose presence gates the generative path. Checked in order.
_KEY_ENV_VARS: tuple[str, ...] = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")

#: Maximum output tokens requested from the LLM (a cited 10-K answer is short).
_MAX_OUTPUT_TOKENS: int = 1024

#: Matches an inline ``[chunk_id]`` citation marker. Chunk ids are
#: ``<accession>:<item>:<ordinal>`` plus the ``item | span`` extractive form, so the
#: marker body is any run of non-``]`` characters; the leading token (before the
#: first ``|``/whitespace) is treated as the candidate chunk id.
_CITATION_RE = re.compile(r"\[([^\]]+)\]")

#: The strict system instruction prepended to every grounded-answer prompt.
GROUNDED_SYSTEM_INSTRUCTION: str = (
    "You answer questions about a company's 10-K filing using ONLY the numbered "
    "context chunks provided. Every sentence in your answer MUST cite the id(s) of "
    "the chunk(s) that support it, in square brackets, e.g. [chunk_id]. If the "
    f"context does not contain the answer, reply with exactly: {ABSTENTION_MESSAGE!r}. "
    "Never use knowledge outside the provided chunks. Never invent a citation."
)


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    """An immutable generated answer + the chunk ids it cited.

    Attributes
    ----------
    answer:
        The generated answer text (or the abstention message).
    cited_chunk_ids:
        The chunk ids the answer cited, in first-appearance order.
    abstained:
        ``True`` iff the model emitted the abstention message.
    model:
        The LLM model id used.
    extra:
        Optional extra provenance (token counts, provider).
    """

    answer: str
    cited_chunk_ids: list[str]
    abstained: bool
    model: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this answer."""
        return asdict(self)


def llm_key_present() -> bool:
    """Return ``True`` iff an LLM API key is set in the environment.

    Checks ``ANTHROPIC_API_KEY`` then ``OPENAI_API_KEY``. PURE (reads only
    ``os.environ``); imports no client. This is the single gate the serve path
    uses to decide generative-vs-extractive in ``mode="auto"``.

    Returns
    -------
    bool
        Whether a generation key is available.
    """
    return any(os.environ.get(name, "").strip() for name in _KEY_ENV_VARS)


def resolve_default_model() -> str:
    """Return the configured default generation model id.

    Reads :data:`MODEL_ENV_VAR` (so the deployed model can be swapped via a Fly
    secret) and falls back to :data:`DEFAULT_MODEL`. PURE (reads only the
    environment).

    Returns
    -------
    str
        The model id to use when the caller does not pass one explicitly.
    """
    return os.environ.get(MODEL_ENV_VAR, "").strip() or DEFAULT_MODEL


def resolve_mode(mode: str, *, key_present: bool | None = None) -> str:
    """Resolve the answer ``mode`` to ``"extractive"`` or ``"generative"``.

    The dispatcher rule: ``"auto"`` resolves to ``"generative"`` iff an LLM key is
    present, else ``"extractive"``; ``"extractive"`` and ``"generative"`` pass
    through. PURE given ``key_present`` (which defaults to a live
    :func:`llm_key_present` check when not supplied, e.g. in tests).

    Parameters
    ----------
    mode:
        ``"auto"``, ``"extractive"``, or ``"generative"``.
    key_present:
        Whether an LLM key is available. ``None`` defers to
        :func:`llm_key_present`.

    Returns
    -------
    str
        ``"extractive"`` or ``"generative"``.

    Raises
    ------
    ValidationError
        If ``mode`` is not one of the three supported values.
    """
    if mode not in ("auto", "extractive", "generative"):
        raise ValidationError(
            f"resolve_mode: mode must be 'auto'|'extractive'|'generative', got {mode!r}."
        )
    if mode != "auto":
        return mode
    available = llm_key_present() if key_present is None else key_present
    return "generative" if available else "extractive"


def build_grounded_prompt(question: str, retrieved: Sequence[RetrievedChunk]) -> str:
    """Build the STRICT grounded-answer user prompt from the retrieved chunks.

    Lays out the numbered context as ``[chunk_id] <chunk text>`` blocks followed
    by the question and the cite-or-abstain instruction. PURE and deterministic —
    the prompt contains the chunk text verbatim and the strict instruction (a test
    asserts both), and it never includes any text that was not retrieved.

    Parameters
    ----------
    question:
        The user question.
    retrieved:
        The retrieved chunks (citation candidates) to ground the answer in.

    Returns
    -------
    str
        The fully-rendered user prompt.

    Raises
    ------
    ValidationError
        If ``question`` is empty or ``retrieved`` is empty.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValidationError("build_grounded_prompt: question must be a non-empty string.")
    if not retrieved:
        raise ValidationError("build_grounded_prompt: retrieved must be non-empty.")

    # One numbered block per retrieved chunk: "[chunk_id] (Item <item>) <text>".
    # The chunk TEXT is laid out verbatim (a test asserts the prompt contains it),
    # so the model is grounded ONLY in what was retrieved — no parametric free-form.
    blocks = [
        f"[{hit.chunk.chunk_id}] (Item {hit.chunk.item}) {hit.chunk.text.strip()}"
        for hit in retrieved
    ]
    context = "\n\n".join(blocks)

    return (
        "Context chunks (cite by their bracketed id):\n"
        f"{context}\n\n"
        f"Question: {question.strip()}\n\n"
        "Answer using ONLY the context chunks above. Cite the supporting chunk id(s) "
        "in square brackets after each sentence. If the context does not contain the "
        f"answer, reply with exactly: {ABSTENTION_MESSAGE!r}."
    )


def generate_answer(
    question: str,
    retrieved: Sequence[RetrievedChunk],
    *,
    model: str = DEFAULT_MODEL,
    client: Callable[[str, str], str] | None = None,
) -> GeneratedAnswer:
    """Generate a grounded, cited answer from the retrieved chunks (LLM, key-gated).

    Builds the strict prompt via :func:`build_grounded_prompt`, calls the LLM
    (``client``, or a lazily-constructed anthropic/openai client when a key is
    present), and parses the cited chunk ids out of the response. The injectable
    ``client`` callable ``(system, user) -> str`` is how tests MOCK the LLM with
    no key and no network.

    Parameters
    ----------
    question:
        The user question.
    retrieved:
        The retrieved chunks to ground the answer in.
    model:
        The LLM model id.
    client:
        Optional injected ``(system_prompt, user_prompt) -> completion`` callable
        (mocked in tests). When ``None``, a real client is built lazily and
        requires a key.

    Returns
    -------
    GeneratedAnswer
        The generated answer + the chunk ids it cited.

    Raises
    ------
    ValidationError
        If the inputs are invalid.
    GenerationError
        If no key is present and no ``client`` is injected, or the LLM call fails.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValidationError("generate_answer: question must be a non-empty string.")
    if not retrieved:
        raise ValidationError("generate_answer: retrieved must be non-empty.")
    if not isinstance(model, str) or not model.strip():
        raise ValidationError("generate_answer: model must be a non-empty string.")

    user_prompt = build_grounded_prompt(question, retrieved)

    call = client if client is not None else _build_lazy_client(model)

    try:
        completion = call(GROUNDED_SYSTEM_INSTRUCTION, user_prompt)
    except (GenerationError, ValidationError):
        # Library errors (e.g. no-key from the lazy client) propagate unchanged so
        # the serve path can distinguish them and degrade to extractive.
        raise
    except Exception as exc:  # normalize any client/transport error to GenerationError
        raise GenerationError(f"generate_answer: LLM call failed: {exc}") from exc

    if not isinstance(completion, str):
        raise GenerationError(
            f"generate_answer: LLM client returned {type(completion).__name__}, expected str."
        )

    answer_text = completion.strip()

    # Strict abstention: an answer that IS the canonical abstention message carries
    # no citations and is flagged abstained (downstream MUST NOT treat it as grounded).
    if _is_abstention(answer_text):
        return GeneratedAnswer(
            answer=ABSTENTION_MESSAGE,
            cited_chunk_ids=[],
            abstained=True,
            model=model,
            extra={"provider": _provider_for_model(model)},
        )

    # Parse cited ids, keeping ONLY those that were actually retrieved — a model that
    # emits a fabricated/non-retrieved citation has it DROPPED here, so a generated
    # answer never reports a citation to a chunk that was not in its context. (The
    # serve path additionally runs validate_citations over these for a hard guard.)
    retrieved_ids = [hit.chunk.chunk_id for hit in retrieved]
    cited_chunk_ids = _parse_cited_chunk_ids(answer_text, allowed_ids=retrieved_ids)

    return GeneratedAnswer(
        answer=answer_text,
        cited_chunk_ids=cited_chunk_ids,
        abstained=False,
        model=model,
        extra={"provider": _provider_for_model(model)},
    )


def _is_abstention(text: str) -> bool:
    """Return ``True`` iff ``text`` is the canonical abstention message (case-folded)."""
    return text.strip().casefold().rstrip(".") == ABSTENTION_MESSAGE.casefold()


def _parse_cited_chunk_ids(answer: str, *, allowed_ids: Sequence[str]) -> list[str]:
    """Extract cited chunk ids from ``answer``, keeping only ids in ``allowed_ids``.

    Scans every ``[...]`` marker, takes the leading token (the id, before any
    ``" | "`` provenance suffix the extractive marker adds), and keeps it ONLY when
    it is one of the retrieved (``allowed_ids``) chunk ids — so a hallucinated /
    non-retrieved citation is silently rejected rather than reported. Returns ids in
    first-appearance order, de-duplicated.

    Parameters
    ----------
    answer:
        The model's answer text.
    allowed_ids:
        The chunk ids that were actually retrieved (the only valid citation set).

    Returns
    -------
    list[str]
        The retrieved chunk ids the answer cited, in first-appearance order.
    """
    allowed = set(allowed_ids)
    cited: list[str] = []
    seen: set[str] = set()
    for marker in _CITATION_RE.findall(answer):
        # The extractive marker is "[id | item | span]"; a generative answer cites
        # the bare id. Take the leading whitespace/pipe-delimited token either way.
        candidate = marker.split("|", 1)[0].strip()
        if candidate in allowed and candidate not in seen:
            seen.add(candidate)
            cited.append(candidate)
    return cited


def _provider_for_model(model: str) -> str:
    """Return ``"anthropic"`` or ``"openai"`` inferred from the model id (best effort)."""
    lowered = model.lower()
    if lowered.startswith("claude") or "anthropic" in lowered:
        return "anthropic"
    if lowered.startswith(("gpt", "o1", "o3", "o4")) or "openai" in lowered:
        return "openai"
    return "anthropic"


def _build_lazy_client(model: str) -> Callable[[str, str], str]:
    """Build a real LLM ``(system, user) -> str`` client LAZILY (key-gated).

    Constructs an anthropic or openai client only when an API key is present, and
    imports the provider SDK INSIDE this function (never at module load) to keep the
    package import-pure. In tests a mock ``client`` is always injected, so this code
    path is never exercised without a real key.

    Parameters
    ----------
    model:
        The model id (also selects the provider via :func:`_provider_for_model`).

    Returns
    -------
    Callable[[str, str], str]
        A ``(system_prompt, user_prompt) -> completion_text`` callable.

    Raises
    ------
    GenerationError
        If no LLM key is present (the serve path degrades to extractive instead).
    """
    if not llm_key_present():
        raise GenerationError(
            "generate_answer: no LLM key (ANTHROPIC_API_KEY / OPENAI_API_KEY) and no "
            "client injected; the serve path degrades to the extractive answer."
        )

    provider = _provider_for_model(model)
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    # Prefer the provider implied by the model id, but fall back to whichever key
    # is actually set so an OpenAI key + a claude default still resolves sensibly.
    if provider == "anthropic" and not anthropic_key:
        provider = "openai"

    if provider == "anthropic":
        return _anthropic_client(model)
    return _openai_client(model)


def _anthropic_client(model: str) -> Callable[[str, str], str]:  # pragma: no cover - needs a key
    """Return an Anthropic-backed ``(system, user) -> str`` callable (lazy import)."""
    import anthropic  # imported lazily to preserve import purity

    sdk = anthropic.Anthropic()

    def _call(system: str, user: str) -> str:
        message = sdk.messages.create(
            model=model,
            max_tokens=_MAX_OUTPUT_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        # message.content is a union of block types; pull text out of the text blocks
        # via getattr so mypy does not require narrowing every union member.
        parts = [
            str(getattr(block, "text", ""))
            for block in message.content
            if getattr(block, "type", None) == "text"
        ]
        return "".join(parts)

    return _call


def _openai_client(model: str) -> Callable[[str, str], str]:  # pragma: no cover - needs a key
    """Return an OpenAI-backed ``(system, user) -> str`` callable (lazy import)."""
    import openai  # imported lazily to preserve import purity

    sdk = openai.OpenAI()

    def _call(system: str, user: str) -> str:
        response = sdk.chat.completions.create(
            model=model,
            max_tokens=_MAX_OUTPUT_TOKENS,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = response.choices[0].message.content
        return content or ""

    return _call


def compose_answer(
    question: str,
    retrieval: RetrievalResult,
    *,
    mode: str = "auto",
    model: str | None = None,
    client: Callable[[str, str], str] | None = None,
    max_chunks: int = DEFAULT_TOP_K,
) -> GeneratedAnswer:
    """Dispatch to the generative or extractive answer per ``mode`` (the dispatcher).

    The single entrypoint the serve path uses to turn a retrieval into an answer:

    - if ``retrieval.abstain`` → the abstention message with no citations (NEVER an
      answer), regardless of ``mode``;
    - ``mode="auto"`` → generative iff an LLM key is present (or a ``client`` is
      injected) else extractive;
    - ``mode="generative"`` → generative when possible, else the key-free extractive
      fallback (NEVER a hard failure — the deployed default must work without a key);
    - ``mode="extractive"`` → always the key-free extractive answer.

    Parameters
    ----------
    question:
        The user question.
    retrieval:
        The :class:`rag10k.retrieve.search.RetrievalResult` to answer from.
    mode:
        ``"auto"`` | ``"extractive"`` | ``"generative"``.
    model:
        The generation model id (``None`` → :func:`resolve_default_model`).
    client:
        Optional injected LLM callable (mocked in tests). Its presence makes the
        generative path available even when no key is set.
    max_chunks:
        Max chunks for the extractive fallback.

    Returns
    -------
    GeneratedAnswer
        The composed (generative or extractive) answer.

    Raises
    ------
    ValidationError
        If ``mode`` is unknown or inputs are invalid.
    """
    from rag10k.generate.extractive import extractive_answer

    # Abstention dominates: a below-threshold retrieval is answered ONLY by the
    # extractive path, which emits the canonical abstention message with no citations.
    if retrieval.abstain or not retrieval.retrieved:
        return extractive_answer(retrieval, max_chunks=max_chunks)

    # A injected client makes generation available even without an env key (tests).
    key_present = client is not None or llm_key_present()
    resolved = resolve_mode(mode, key_present=key_present)

    if resolved == "generative":
        resolved_model = (model or "").strip() or resolve_default_model()
        try:
            return generate_answer(
                question, retrieval.retrieved, model=resolved_model, client=client
            )
        except GenerationError:
            # NEVER hard-fail the deployed endpoint: degrade to the key-free answer.
            return extractive_answer(retrieval, max_chunks=max_chunks)

    return extractive_answer(retrieval, max_chunks=max_chunks)
