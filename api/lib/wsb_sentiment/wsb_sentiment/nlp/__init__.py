"""Lexicon-based sentiment scoring (finance-augmented VADER, with TextBlob parity).

Both scorers are LEXICON-BASED — there is NO model fit, NO transformers/torch/TF,
and NO NLTK download at import. The VADER analyzer is the primary scorer; TextBlob
provides an independent cross-check used by the parity test suite. Scoring happens
in the OFFLINE ``score`` CLI path on the ingested text, never at request time.

Importing this subpackage has no side effects.
"""

from __future__ import annotations

from wsb_sentiment.nlp.textblob_parity import (
    TextBlobScore,
    score_textblob,
    score_textblob_batch,
)
from wsb_sentiment.nlp.vader import (
    FINANCE_LEXICON,
    VaderScore,
    score_vader,
    score_vader_batch,
)

__all__ = [
    "FINANCE_LEXICON",
    "TextBlobScore",
    "VaderScore",
    "score_textblob",
    "score_textblob_batch",
    "score_vader",
    "score_vader_batch",
]
