"""Hand-rolled readability indices and the counts behind them.

Implements three classic readability formulas from first principles:

- **Gunning-Fog**: ``0.4 * (words/sentences + 100 * complex_words/words)`` where a
  complex word has >= 3 syllables.
- **Flesch-Kincaid grade**: ``0.39 * (words/sentences) + 11.8 * (syllables/words)
  - 15.59``.
- **SMOG**: ``1.0430 * sqrt(polysyllables * 30 / sentences) + 3.1291`` where a
  polysyllable has >= 3 syllables.

The syllable counter (:func:`count_syllables`) is a deterministic vowel-group
heuristic with the standard silent-``e``, trailing-``le``, and silent
``-ed``/``-es`` adjustments, validated against ``textstat`` to a pinned tolerance
in the parity suite. NO ML, NO runtime third-party dependency.

Word and sentence counts are produced by the shared
:func:`edgar_nlp._validation.tokenize` / :func:`edgar_nlp._validation.split_sentences`
helpers so that they agree exactly with the Loughran-McDonald sentiment scorer.

Importing this module has no side effects.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from edgar_nlp._validation import split_sentences, tokenize

# quantcore-candidate: readability formulas; parity oracle is textstat.

#: Vowel characters (``y`` counts as a vowel, the standard heuristic choice).
_VOWELS: frozenset[str] = frozenset("aeiouy")

#: Vowel-group pattern: one syllable nucleus per contiguous run of vowels.
_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+")

#: A word is "complex"/polysyllabic (for Fog and SMOG) at this syllable count.
_COMPLEX_SYLLABLE_THRESHOLD: int = 3

# Gunning-Fog coefficients.
_FOG_WORD_SENTENCE_COEF: float = 0.4
_FOG_COMPLEX_COEF: float = 100.0

# Flesch-Kincaid grade coefficients.
_FK_WORDS_COEF: float = 0.39
_FK_SYLLABLES_COEF: float = 11.8
_FK_CONSTANT: float = 15.59

# SMOG coefficients.
_SMOG_COEF: float = 1.0430
_SMOG_SAMPLE: float = 30.0
_SMOG_CONSTANT: float = 3.1291


@dataclass(frozen=True, slots=True)
class ReadabilityScore:
    """Immutable readability score for one body of text.

    Attributes
    ----------
    word_count:
        Number of scored word tokens.
    n_sentences:
        Number of sentences detected by the shared splitter.
    syllable_count:
        Total syllables across all scored words.
    complex_word_count:
        Number of words with >= 3 syllables (the Fog/SMOG polysyllable count).
    gunning_fog:
        The Gunning-Fog index.
    flesch_kincaid:
        The Flesch-Kincaid grade level.
    smog:
        The SMOG index.
    extra:
        Optional extra provenance (reserved).
    """

    word_count: int
    n_sentences: int
    syllable_count: int
    complex_word_count: int
    gunning_fog: float
    flesch_kincaid: float
    smog: float
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this score."""
        return {
            "word_count": int(self.word_count),
            "n_sentences": int(self.n_sentences),
            "syllable_count": int(self.syllable_count),
            "complex_word_count": int(self.complex_word_count),
            "gunning_fog": float(self.gunning_fog),
            "flesch_kincaid": float(self.flesch_kincaid),
            "smog": float(self.smog),
            "extra": dict(self.extra),
        }


def count_syllables(word: str) -> int:
    """Count the syllables in a single word with a deterministic heuristic.

    Counts contiguous vowel groups (``aeiouy``), removes a silent trailing ``e``
    (while preserving a consonant + ``le`` ending such as "table"), drops a
    silent ``-ed``/``-es`` inflection where it does not add a syllable, and
    floors the result at ``1`` for any non-empty alphabetic word. Words of three
    or fewer letters are treated as a single syllable (the standard short-word
    rule). Validated against ``textstat`` to a pinned tolerance in the parity
    suite.

    Parameters
    ----------
    word:
        A single word token (case-insensitive; non-alphabetic characters are
        ignored).

    Returns
    -------
    int
        The estimated syllable count (``>= 1`` for a non-empty word, ``0`` for
        an empty/non-alphabetic string).
    """
    cleaned = "".join(ch for ch in word.lower() if ch.isalpha())
    if not cleaned:
        return 0
    if len(cleaned) <= 3:
        return 1

    stem = cleaned
    # Silent trailing 'e' (but a consonant + 'le' keeps its own syllable).
    if stem.endswith("e") and not (stem.endswith("le") and stem[-3] not in _VOWELS):
        stem = stem[:-1]
    # Silent '-ed'/'-es' inflection: drop it when the preceding consonant does
    # not voice the suffix into its own syllable ('-ed' after t/d, '-es' after a
    # sibilant c/s/x/z stay).
    if _has_silent_inflection(stem):
        stem = stem[:-2]

    count = len(_VOWEL_GROUP_RE.findall(stem))
    return max(count, 1)


def _has_silent_inflection(stem: str) -> bool:
    """Return True if ``stem`` ends in a silent ``-ed``/``-es`` inflection."""
    if len(stem) <= 2 or stem[-3] in _VOWELS:
        return False
    return (stem.endswith("ed") and stem[-3] not in "td") or (
        stem.endswith("es") and stem[-3] not in "csxz"
    )


def _counts(text: str) -> tuple[int, int, int, int]:
    """Return ``(word_count, n_sentences, syllable_count, complex_word_count)``.

    Tokenizes and sentence-splits ``text`` once using the shared validation
    helpers so the counts feeding every index are mutually consistent and agree
    with the sentiment scorer's word count.
    """
    tokens = tokenize(text)
    n_sentences = len(split_sentences(text))
    syllable_count = 0
    complex_word_count = 0
    for token in tokens:
        syllables = count_syllables(token)
        syllable_count += syllables
        if syllables >= _COMPLEX_SYLLABLE_THRESHOLD:
            complex_word_count += 1
    return len(tokens), n_sentences, syllable_count, complex_word_count


def _gunning_fog(words: int, sentences: int, complex_words: int) -> float:
    """Compute the Gunning-Fog index from raw counts (``0.0`` if degenerate)."""
    if words == 0 or sentences == 0:
        return 0.0
    return _FOG_WORD_SENTENCE_COEF * (words / sentences + _FOG_COMPLEX_COEF * complex_words / words)


def _flesch_kincaid(words: int, sentences: int, syllables: int) -> float:
    """Compute the Flesch-Kincaid grade from raw counts (``0.0`` if degenerate)."""
    if words == 0 or sentences == 0:
        return 0.0
    return (
        _FK_WORDS_COEF * (words / sentences)
        + _FK_SYLLABLES_COEF * (syllables / words)
        - _FK_CONSTANT
    )


def _smog(sentences: int, complex_words: int) -> float:
    """Compute the SMOG index from raw counts (``0.0`` if no sentences)."""
    if sentences == 0:
        return 0.0
    return _SMOG_COEF * math.sqrt(complex_words * _SMOG_SAMPLE / sentences) + _SMOG_CONSTANT


def gunning_fog(text: str) -> float:
    """Return the Gunning-Fog index of ``text``.

    ``0.4 * (words / sentences + 100 * complex_words / words)``, where a complex
    word has >= 3 syllables. Returns ``0.0`` for empty text.

    Parameters
    ----------
    text:
        Raw text to score.

    Returns
    -------
    float
        The Gunning-Fog index.
    """
    words, sentences, _syllables, complex_words = _counts(text)
    return _gunning_fog(words, sentences, complex_words)


def flesch_kincaid_grade(text: str) -> float:
    """Return the Flesch-Kincaid grade level of ``text``.

    ``0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59``. Returns
    ``0.0`` for empty text.

    Parameters
    ----------
    text:
        Raw text to score.

    Returns
    -------
    float
        The Flesch-Kincaid grade level.
    """
    words, sentences, syllables, _complex_words = _counts(text)
    return _flesch_kincaid(words, sentences, syllables)


def smog_index(text: str) -> float:
    """Return the SMOG index of ``text``.

    ``1.0430 * sqrt(polysyllables * 30 / sentences) + 3.1291``, where a
    polysyllable has >= 3 syllables. Returns ``0.0`` for empty text.

    Parameters
    ----------
    text:
        Raw text to score.

    Returns
    -------
    float
        The SMOG index.
    """
    _words, sentences, _syllables, complex_words = _counts(text)
    return _smog(sentences, complex_words)


def score_text(text: str) -> ReadabilityScore:
    """Compute all readability indices and their underlying counts in one pass.

    Tokenizes and sentence-splits ``text`` once (via the shared helpers in
    :mod:`edgar_nlp._validation`) so the word and sentence counts feeding Fog,
    Flesch-Kincaid, and SMOG are mutually consistent and agree with the
    Loughran-McDonald sentiment scorer.

    Parameters
    ----------
    text:
        Raw section text to score.

    Returns
    -------
    ReadabilityScore
        The frozen bundle of counts and indices. Empty/whitespace-only text
        yields all-zero counts and indices (no division by zero).
    """
    words, sentences, syllables, complex_words = _counts(text)
    return ReadabilityScore(
        word_count=words,
        n_sentences=sentences,
        syllable_count=syllables,
        complex_word_count=complex_words,
        gunning_fog=_gunning_fog(words, sentences, complex_words),
        flesch_kincaid=_flesch_kincaid(words, sentences, syllables),
        smog=_smog(sentences, complex_words),
    )
