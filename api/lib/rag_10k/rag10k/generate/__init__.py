"""Answer generation: the key-gated grounded LLM path + the key-free extractive fallback.

:mod:`rag10k.generate.answer` is the LLM generation step — a lazy anthropic/openai
client behind an ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` env check, driven by a
STRICT grounded-answer prompt that is given ONLY the retrieved chunks and must
cite chunk ids or say "not found in the filing". :mod:`rag10k.generate.extractive`
is the KEY-FREE fallback: it returns the top retrieved chunks verbatim as the
answer with their citations (the deployed default when no key is set).

IMPORT PURITY: no anthropic / openai import at module load; the LLM client is
constructed lazily inside the generation call, only when a key is present.
Importing this package has no side effects.
"""

from __future__ import annotations

from rag10k.generate.answer import (
    GeneratedAnswer,
    build_grounded_prompt,
    compose_answer,
    generate_answer,
    llm_key_present,
    resolve_default_model,
    resolve_mode,
)
from rag10k.generate.extractive import extractive_answer, format_citation_marker

__all__ = [
    "GeneratedAnswer",
    "build_grounded_prompt",
    "compose_answer",
    "extractive_answer",
    "format_citation_marker",
    "generate_answer",
    "llm_key_present",
    "resolve_default_model",
    "resolve_mode",
]
