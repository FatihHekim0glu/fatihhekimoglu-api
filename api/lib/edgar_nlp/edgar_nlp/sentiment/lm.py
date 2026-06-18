"""Loughran-McDonald dictionary sentiment scorer.

Looks up each token against the FIXED LM lexicon
(:mod:`edgar_nlp.sentiment.lexicon`) and reports the per-category word fraction
plus the net tone ``(neg - pos) / total``. There is NO fitting and NO ML: the
dictionary is reference data, applied verbatim, so the scorer is deterministic
and reproducible. Validated against ``pysentiment2`` in the parity suite.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from edgar_nlp._validation import tokenize
from edgar_nlp.sentiment.lexicon import (
    LM_CONSTRAINING,
    LM_LITIGIOUS,
    LM_NEGATIVE,
    LM_POSITIVE,
    LM_UNCERTAINTY,
)

# quantcore-candidate: LM fraction scorer; parity oracle is pysentiment2.


@dataclass(frozen=True, slots=True)
class LMScore:
    """Immutable Loughran-McDonald sentiment score for one body of text.

    All ``*_pct`` fields are fractions of total scored word tokens in ``[0, 1]``.
    ``tone`` is the net negativity ``(neg_pct - pos_pct)`` so that more-negative
    documents score higher.

    Attributes
    ----------
    n_words:
        Total scored word tokens (the denominator for every fraction).
    neg_pct:
        Fraction of tokens in the LM negative category.
    pos_pct:
        Fraction of tokens in the LM positive category.
    uncertainty_pct:
        Fraction of tokens in the LM uncertainty category.
    litigious_pct:
        Fraction of tokens in the LM litigious category.
    constraining_pct:
        Fraction of tokens in the LM constraining category.
    tone:
        Net tone ``neg_pct - pos_pct`` in ``[-1, 1]`` (``0.0`` for empty text).
    extra:
        Optional extra provenance (reserved).
    """

    n_words: int
    neg_pct: float
    pos_pct: float
    uncertainty_pct: float
    litigious_pct: float
    constraining_pct: float
    tone: float
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this score."""
        return {
            "n_words": int(self.n_words),
            "neg_pct": float(self.neg_pct),
            "pos_pct": float(self.pos_pct),
            "uncertainty_pct": float(self.uncertainty_pct),
            "litigious_pct": float(self.litigious_pct),
            "constraining_pct": float(self.constraining_pct),
            "tone": float(self.tone),
            "extra": dict(self.extra),
        }


def score_tokens(tokens: list[str]) -> LMScore:
    """Score a list of word tokens against the fixed LM lexicon.

    Counts category membership over ``tokens`` (already lowercased by
    :func:`edgar_nlp._validation.tokenize`) and divides by the token count to get
    per-category fractions. The net ``tone`` is ``neg_pct - pos_pct``. An empty
    token list yields an all-zero score (``tone = 0.0``) rather than dividing by
    zero.

    Parameters
    ----------
    tokens:
        Lowercased word tokens to score.

    Returns
    -------
    LMScore
        The frozen per-category fractions and net tone.
    """
    n_words = len(tokens)
    if n_words == 0:
        return LMScore(
            n_words=0,
            neg_pct=0.0,
            pos_pct=0.0,
            uncertainty_pct=0.0,
            litigious_pct=0.0,
            constraining_pct=0.0,
            tone=0.0,
        )

    neg = pos = unc = lit = con = 0
    for token in tokens:
        if token in LM_NEGATIVE:
            neg += 1
        if token in LM_POSITIVE:
            pos += 1
        if token in LM_UNCERTAINTY:
            unc += 1
        if token in LM_LITIGIOUS:
            lit += 1
        if token in LM_CONSTRAINING:
            con += 1

    denom = float(n_words)
    neg_pct = neg / denom
    pos_pct = pos / denom
    return LMScore(
        n_words=n_words,
        neg_pct=neg_pct,
        pos_pct=pos_pct,
        uncertainty_pct=unc / denom,
        litigious_pct=lit / denom,
        constraining_pct=con / denom,
        tone=neg_pct - pos_pct,
    )


def score_text(text: str) -> LMScore:
    """Tokenize ``text`` (via the shared tokenizer) and score it.

    Convenience wrapper over :func:`score_tokens` that funnels ``text`` through
    :func:`edgar_nlp._validation.tokenize` so the word count agrees exactly with
    the readability metrics.

    Parameters
    ----------
    text:
        Raw section text to score.

    Returns
    -------
    LMScore
        The frozen per-category fractions and net tone.
    """
    return score_tokens(tokenize(text))
