"""TextBlob sentiment cross-check (parity oracle for VADER).

TextBlob's pattern-based polarity is an independent, LEXICON-based scorer used as
a cross-check against the primary VADER score in the parity test suite: the two
should agree in SIGN on clearly-polar text and correlate across a corpus. It is
NOT the production signal — VADER is — but disagreement flags lexicon bugs.

IMPORT PURITY: ``textblob`` is imported LAZILY inside :func:`score_textblob`.
Importing this module triggers no textblob import and no corpus download.

Importing this module has no side effects.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TextBlobScore:
    """A TextBlob polarity/subjectivity score for one piece of text.

    Attributes
    ----------
    polarity:
        Sentiment polarity in ``[-1, 1]`` (the comparable axis vs. VADER
        ``compound``).
    subjectivity:
        Subjectivity in ``[0, 1]`` (``0`` objective, ``1`` subjective).
    """

    polarity: float
    subjectivity: float

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this score."""
        return asdict(self)


def score_textblob(text: str) -> TextBlobScore:
    """Score a single text with TextBlob (LAZY import).

    ``textblob`` is imported here, never at module import, so importing this module
    triggers no textblob import and no corpus download.

    Parameters
    ----------
    text:
        The text to score.

    Returns
    -------
    TextBlobScore
        The polarity/subjectivity pair.
    """
    from textblob import TextBlob

    sentiment = TextBlob(text).sentiment
    return TextBlobScore(
        polarity=float(sentiment.polarity),
        subjectivity=float(sentiment.subjectivity),
    )


def score_textblob_batch(texts: Iterable[str]) -> list[TextBlobScore]:
    """Score many texts with TextBlob, in order.

    Imports ``textblob`` once (lazily) and scores every text with it.

    Parameters
    ----------
    texts:
        The texts to score.

    Returns
    -------
    list[TextBlobScore]
        One score per input text, in order.
    """
    from textblob import TextBlob

    out: list[TextBlobScore] = []
    for text in texts:
        sentiment = TextBlob(text).sentiment
        out.append(
            TextBlobScore(
                polarity=float(sentiment.polarity),
                subjectivity=float(sentiment.subjectivity),
            )
        )
    return out
