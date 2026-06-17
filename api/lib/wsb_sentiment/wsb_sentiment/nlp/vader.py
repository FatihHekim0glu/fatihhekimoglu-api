"""Finance-augmented VADER sentiment scoring (primary, lexicon-based).

VADER (Hutto & Gilbert, 2014) is a rule-based, LEXICON sentiment analyzer: there
is NO model fit and NO network/NLTK download. We augment the stock lexicon with a
small, STATIC finance-slang booster table (e.g. ``"moon" -> +``, ``"bagholder"
-> -``, ``"tendies" -> +``) so r/wallstreetbets jargon scores sensibly. The
augmentation is a constant dict applied to a freshly-constructed analyzer; it is
never re-estimated.

IMPORT PURITY: ``vaderSentiment`` is imported LAZILY inside the scoring functions.
Importing this module constructs no analyzer and touches no network/disk.

Importing this module has no side effects.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any, Final

#: STATIC finance-slang lexicon booster, merged into VADER's lexicon at analyzer
#: construction time. Values are VADER valence scores in roughly ``[-4, 4]``.
#: This table is a CONSTANT — it is never fit or learned from data.
FINANCE_LEXICON: Final[dict[str, float]] = {
    "moon": 2.5,
    "mooning": 2.5,
    "tendies": 2.0,
    "rocket": 2.0,
    "squeeze": 1.2,
    "diamond": 1.5,
    "hold": 0.8,
    "bullish": 2.5,
    "calls": 1.0,
    "bagholder": -2.0,
    "bagholding": -2.0,
    "puts": -1.0,
    "bearish": -2.5,
    "drilling": -1.5,
    "rug": -2.5,
    "rugged": -2.5,
    "dump": -1.5,
    "dumping": -1.5,
}


@dataclass(frozen=True, slots=True)
class VaderScore:
    """A VADER polarity score for one piece of text.

    Attributes
    ----------
    compound:
        The normalized aggregate score in ``[-1, 1]`` (the primary signal).
    positive:
        The proportion of text rated positive in ``[0, 1]``.
    neutral:
        The proportion rated neutral in ``[0, 1]``.
    negative:
        The proportion rated negative in ``[0, 1]``.
    """

    compound: float
    positive: float
    neutral: float
    negative: float

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this score."""
        return asdict(self)


def _build_analyzer(finance_lexicon: dict[str, float] | None) -> Any:
    """Construct a finance-augmented ``SentimentIntensityAnalyzer`` (LAZY import).

    ``vaderSentiment`` is imported here, never at module import, so importing this
    module touches no network/disk. The supplied (or default static) finance-slang
    booster is merged into a FRESH analyzer's lexicon; nothing is fit or mutated
    on the module-level constant.

    Parameters
    ----------
    finance_lexicon:
        Override of the static finance-slang booster table, or ``None`` to use
        :data:`FINANCE_LEXICON`.

    Returns
    -------
    SentimentIntensityAnalyzer
        A fresh analyzer whose lexicon includes the finance boosters.
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    booster = FINANCE_LEXICON if finance_lexicon is None else finance_lexicon
    # Augment a copy via ``update`` on the freshly-built analyzer's own lexicon;
    # the module-level constant is never mutated.
    analyzer.lexicon.update({word: float(value) for word, value in booster.items()})
    return analyzer


def _score_with(analyzer: Any, text: str) -> VaderScore:
    """Score ``text`` with a prepared ``analyzer`` into a :class:`VaderScore`."""
    scores = analyzer.polarity_scores(text)
    return VaderScore(
        compound=float(scores["compound"]),
        positive=float(scores["pos"]),
        neutral=float(scores["neu"]),
        negative=float(scores["neg"]),
    )


def score_vader(
    text: str,
    *,
    finance_lexicon: dict[str, float] | None = None,
) -> VaderScore:
    """Score a single text with finance-augmented VADER (LAZY analyzer).

    Constructs a ``SentimentIntensityAnalyzer`` LAZILY, merges the STATIC
    :data:`FINANCE_LEXICON` (or ``finance_lexicon`` override) into its lexicon,
    and returns the four polarity components. No model is fit.

    Parameters
    ----------
    text:
        The text to score.
    finance_lexicon:
        Optional override of the static finance-slang booster table.

    Returns
    -------
    VaderScore
        The compound score plus positive/neutral/negative proportions.
    """
    analyzer = _build_analyzer(finance_lexicon)
    return _score_with(analyzer, text)


def score_vader_batch(
    texts: Iterable[str],
    *,
    finance_lexicon: dict[str, float] | None = None,
) -> list[VaderScore]:
    """Score many texts with finance-augmented VADER, reusing one analyzer.

    Constructs the augmented analyzer ONCE (lazily) and scores every text with it,
    which is materially faster than calling :func:`score_vader` per text.

    Parameters
    ----------
    texts:
        The texts to score.
    finance_lexicon:
        Optional override of the static finance-slang booster table.

    Returns
    -------
    list[VaderScore]
        One score per input text, in order.
    """
    analyzer = _build_analyzer(finance_lexicon)
    return [_score_with(analyzer, text) for text in texts]
