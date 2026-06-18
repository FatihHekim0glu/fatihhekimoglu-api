"""Majority / class-prior baseline classifier (the trivial floor).

The class-prior classifier predicts the most frequent TRAIN class for every
input and emits the train class-frequency vector as its score. On the
neutral-heavy PhraseBank this already reaches ~60% accuracy, which is exactly why
accuracy alone is a dishonest headline — this baseline exists to make that
visible and to anchor the macro-F1 floor.

The prior is computed on TRAIN labels ONLY; no val/test label ever touches the
fit. Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from finbert_sentiment._constants import N_CLASSES
from finbert_sentiment._validation import ensure_labels, ensure_text_batch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ClassPriorClassifier:
    """A fitted majority/prior classifier over the 3-way label space.

    Construct via :meth:`fit` (never directly); the frozen fields capture the
    train-fold prior so prediction is a pure function of the fitted state.

    Attributes
    ----------
    prior:
        Train class-frequency vector of length ``N_CLASSES`` summing to one.
    majority_index:
        The argmax of ``prior`` (the always-predicted class index).
    """

    prior: tuple[float, ...] = field(default_factory=lambda: (1.0 / N_CLASSES,) * N_CLASSES)
    majority_index: int = 1

    @classmethod
    def fit(cls, labels: Sequence[int]) -> ClassPriorClassifier:
        """Fit the prior on TRAIN labels only.

        Parameters
        ----------
        labels:
            Integer TRAIN class indices.

        Returns
        -------
        ClassPriorClassifier
            The fitted classifier.

        Raises
        ------
        ValidationError
            If ``labels`` is empty or contains out-of-range indices.
        """
        arr = ensure_labels(labels)
        counts = np.bincount(arr, minlength=N_CLASSES).astype(np.float64)
        prior = counts / counts.sum()
        # ``argmax`` breaks ties toward the lowest index, which is deterministic.
        majority = int(np.argmax(counts))
        return cls(prior=tuple(float(p) for p in prior), majority_index=majority)

    def predict(self, texts: Sequence[str]) -> NDArray[np.int64]:
        """Return the majority class index for every input text (ignores content).

        Parameters
        ----------
        texts:
            The batch to classify (only its length is used).

        Returns
        -------
        numpy.ndarray
            A length-``len(texts)`` ``int64`` vector of the majority index.
        """
        batch = ensure_text_batch(texts)
        return np.full(len(batch), self.majority_index, dtype=np.int64)

    def predict_proba(self, texts: Sequence[str]) -> NDArray[np.float64]:
        """Return the (broadcast) train prior as the score for every input.

        Parameters
        ----------
        texts:
            The batch to classify (only its length is used).

        Returns
        -------
        numpy.ndarray
            A ``(len(texts), N_CLASSES)`` matrix; every row equals ``prior``.
        """
        batch = ensure_text_batch(texts)
        row = np.asarray(self.prior, dtype=np.float64)
        return np.tile(row, (len(batch), 1))

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this classifier."""
        return asdict(self)
