"""Hand-rolled readability metrics (no textstat at runtime).

Gunning-Fog, Flesch-Kincaid grade, and SMOG indices, plus the word/sentence/
syllable counts they are built from. The syllable counter is a deterministic
heuristic (validated against ``textstat`` to a pinned tolerance in the parity
suite); there is no ML and no external runtime dependency. Importing this package
has no side effects.
"""

from __future__ import annotations

from edgar_nlp.readability.metrics import (
    ReadabilityScore,
    count_syllables,
    flesch_kincaid_grade,
    gunning_fog,
    score_text,
    smog_index,
)

__all__ = [
    "ReadabilityScore",
    "count_syllables",
    "flesch_kincaid_grade",
    "gunning_fog",
    "score_text",
    "smog_index",
]
