"""Loughran-McDonald dictionary-based sentiment scoring (no ML).

The LM lexicon is a FIXED external dictionary (Loughran & McDonald, 2011): word
categories are looked up, never fitted. The scorer returns the per-category word
fractions (negative, positive, uncertainty, litigious, constraining) and a net
tone ``(neg - pos) / total``. A small curated subset of the LM word-list ships as
reference data (:mod:`edgar_nlp.sentiment.lexicon`). Importing this package has no
side effects.
"""

from __future__ import annotations

from edgar_nlp.sentiment.lm import LMScore, score_text, score_tokens

__all__ = ["LMScore", "score_text", "score_tokens"]
