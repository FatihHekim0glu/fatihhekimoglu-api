"""Loughran-McDonald-style 3-way lexicon classifier (the torch-free LIVE model).

This classifier counts domain-specific positive/negative cue words in each
sentence and decides ``positive`` / ``negative`` by the signed count, defaulting
to ``neutral`` when the net tone is zero. It is the honest lexical floor the
transformer is measured against AND the always-available served model the FastAPI
tool falls back to when no ONNX artifact is present (it needs only Python — no
torch, no onnxruntime, no network).

The word lists below are a compact, finance-oriented subset in the spirit of the
Loughran-McDonald master dictionary; they are illustrative, not the full LM list.
Importing this module has no side effects.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from finbert_sentiment._validation import ensure_text_batch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

#: Tokenizer: split on any run of non-alphanumeric characters. Lowercasing is
#: applied before matching so the (lowercase) cue-word sets match case-insensitively.
_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: Positive finance cue words (lowercase stems). Compact LM-style subset.
LEXICON_POSITIVE: frozenset[str] = frozenset(
    {
        "gain",
        "gains",
        "profit",
        "profits",
        "growth",
        "rose",
        "rise",
        "rises",
        "rising",
        "up",
        "surge",
        "surged",
        "beat",
        "beats",
        "strong",
        "improved",
        "improvement",
        "record",
        "higher",
        "positive",
        "upgrade",
        "outperform",
        "boost",
        "boosted",
        "exceeded",
    }
)

#: Negative finance cue words (lowercase stems). Compact LM-style subset.
LEXICON_NEGATIVE: frozenset[str] = frozenset(
    {
        "loss",
        "losses",
        "decline",
        "declined",
        "fell",
        "fall",
        "falls",
        "drop",
        "dropped",
        "down",
        "plunge",
        "plunged",
        "miss",
        "missed",
        "weak",
        "weaker",
        "cut",
        "cuts",
        "lower",
        "negative",
        "downgrade",
        "underperform",
        "warning",
        "lawsuit",
        "bankruptcy",
    }
)


@dataclass(frozen=True, slots=True)
class LexiconClassifier:
    """A stateless, deterministic lexicon sentiment classifier.

    The classifier holds no fitted parameters (the word lists are fixed), so it
    is constructible directly and is safe to share across requests. ``predict``
    is a pure function of the input text and the two frozen word sets.

    Attributes
    ----------
    positive:
        Positive cue-word set (defaults to :data:`LEXICON_POSITIVE`).
    negative:
        Negative cue-word set (defaults to :data:`LEXICON_NEGATIVE`).
    """

    positive: frozenset[str] = field(default_factory=lambda: LEXICON_POSITIVE)
    negative: frozenset[str] = field(default_factory=lambda: LEXICON_NEGATIVE)

    def _score_one(self, text: str) -> tuple[int, int]:
        """Return the ``(pos_count, neg_count)`` cue-word counts for one sentence."""
        pos = 0
        neg = 0
        for token in _TOKEN_RE.findall(text.lower()):
            if token in self.positive:
                pos += 1
            if token in self.negative:
                neg += 1
        return pos, neg

    def predict(self, texts: Sequence[str]) -> NDArray[np.int64]:
        """Classify each text as negative/neutral/positive by net cue tone.

        Decision rule: ``positive`` if ``pos_count > neg_count``, ``negative`` if
        ``neg_count > pos_count``, else ``neutral``. Deterministic and content-only.

        Parameters
        ----------
        texts:
            The batch to classify.

        Returns
        -------
        numpy.ndarray
            A length-``len(texts)`` ``int64`` vector of class indices.

        Raises
        ------
        ValidationError
            If ``texts`` fails the batch validation.
        """
        batch = ensure_text_batch(texts)
        out = np.empty(len(batch), dtype=np.int64)
        for i, text in enumerate(batch):
            pos, neg = self._score_one(text)
            if pos > neg:
                out[i] = 2  # positive
            elif neg > pos:
                out[i] = 0  # negative
            else:
                out[i] = 1  # neutral
        return out

    def predict_proba(self, texts: Sequence[str]) -> NDArray[np.float64]:
        """Return pseudo-probabilities derived from normalized cue counts.

        The scores are a softmax-like normalization of ``(neg, neutral, pos)``
        pseudo-logits; they are NOT calibrated probabilities (the lexicon has no
        likelihood model) and are documented as such wherever surfaced.

        Parameters
        ----------
        texts:
            The batch to classify.

        Returns
        -------
        numpy.ndarray
            A ``(len(texts), N_CLASSES)`` row-stochastic score matrix.
        """
        batch = ensure_text_batch(texts)
        # Pseudo-logits ordered (negative, neutral, positive), built from the
        # signed cue difference ``d = pos - neg`` so the row argmax is provably
        # identical to :meth:`predict`:
        #   positive logit = +d, negative logit = -d, neutral logit = bias.
        # The neutral bias lies in ``(0, 1)``; since cue counts are integers the
        # smallest non-tie ``|d|`` is 1 > bias, so positive/negative win whenever
        # ``d != 0`` and neutral wins exactly on the tie ``d == 0`` (incl. no cues).
        bias = 0.5
        logits = np.empty((len(batch), 3), dtype=np.float64)
        for i, text in enumerate(batch):
            pos, neg = self._score_one(text)
            d = float(pos - neg)
            logits[i, 0] = -d
            logits[i, 1] = bias
            logits[i, 2] = d
        # Numerically stable row-wise softmax.
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        proba: NDArray[np.float64] = exp / exp.sum(axis=1, keepdims=True)
        return proba

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` (word sets as sorted lists)."""
        out: dict[str, Any] = asdict(self)
        out["positive"] = sorted(self.positive)
        out["negative"] = sorted(self.negative)
        return out
