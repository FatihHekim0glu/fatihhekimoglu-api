"""Sentence-level deduplication and the normalized-sentence hash.

Near-duplicate sentences are the real leakage risk in the Financial PhraseBank:
the same headline appears in slightly different surface forms. We normalize each
sentence (lowercase, collapse whitespace, strip punctuation noise) and hash it;
exact-normalized duplicates collapse to one example, and the hash later groups
near-duplicates so they never straddle the train/val/test split.

Importing this module has no side effects.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

from finbert_sentiment._exceptions import ValidationError
from finbert_sentiment.data.load import LabelledDataset

#: Collapse any run of Unicode whitespace to a single ASCII space.
_WHITESPACE_RE = re.compile(r"\s+")

#: Trailing/leading punctuation noise that distinguishes otherwise-identical
#: sentences (e.g. a missing full stop on a near-duplicate). Stripping it makes
#: the hash group near-duplicates that differ only in terminal punctuation.
_PUNCT_NOISE_RE = re.compile(r"[.,;:!?…\"'`)(\]\[}{]+")

#: Digest size (bytes) for the BLAKE2b group key -> 16 bytes -> 32 hex chars.
_DIGEST_BYTES = 16


def normalize_sentence(text: str) -> str:
    """Return the canonical normalized form of ``text`` for hashing/dedup.

    The normalization is deterministic and idempotent: lowercase, Unicode-NFKC,
    collapse internal whitespace to single spaces, and strip leading/trailing
    whitespace. Two sentences that differ only in case or spacing map to the same
    normalized string (and therefore the same hash).

    Parameters
    ----------
    text:
        A raw sentence.

    Returns
    -------
    str
        The normalized sentence.
    """
    if not isinstance(text, str):
        raise ValidationError(f"text must be a str, got {type(text).__name__}.")
    folded = unicodedata.normalize("NFKC", text).casefold()
    # Drop punctuation noise so a sentence and its end-period-stripped twin map
    # to the same key, then collapse the whitespace that stripping may leave.
    stripped = _PUNCT_NOISE_RE.sub(" ", folded)
    collapsed = _WHITESPACE_RE.sub(" ", stripped)
    return collapsed.strip()


def sentence_hash(text: str) -> str:
    """Return a stable BLAKE2b-16 hex digest of ``normalize_sentence(text)``.

    The digest is the *group key* for the leakage-safe split: all sentences with
    the same hash are placed in the same fold.

    Parameters
    ----------
    text:
        A raw sentence.

    Returns
    -------
    str
        A 32-character hex digest.
    """
    normalized = normalize_sentence(text)
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=_DIGEST_BYTES).hexdigest()


@dataclass(frozen=True, slots=True)
class DedupResult:
    """Immutable outcome of a dedup pass.

    Attributes
    ----------
    dataset:
        The deduplicated :class:`~finbert_sentiment.data.load.LabelledDataset`
        (one example per unique normalized sentence).
    group_hashes:
        The normalized-sentence hash for each surviving example, parallel to
        ``dataset.texts``. Reused as the group key by the split.
    n_dropped:
        How many exact-normalized duplicate rows were removed.
    """

    dataset: LabelledDataset
    group_hashes: tuple[str, ...]
    n_dropped: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` (dataset flattened)."""
        out: dict[str, Any] = asdict(self)
        out["dataset"] = self.dataset.to_dict()
        return out


def dedup_sentences(dataset: LabelledDataset) -> DedupResult:
    """Collapse exact-normalized duplicate sentences, keeping the first occurrence.

    A label conflict among duplicates (the same normalized sentence carrying two
    different labels) is resolved by keeping the first occurrence and is counted
    in the result; the ``allagree`` PhraseBank config makes this rare.

    Parameters
    ----------
    dataset:
        The (pre-dedup) labelled corpus.

    Returns
    -------
    DedupResult
        The deduplicated dataset, its per-example group hashes, and the drop count.

    Raises
    ------
    ValidationError
        If ``dataset`` is empty.
    """
    if dataset.n == 0:  # pragma: no cover - LabelledDataset forbids empty construction
        raise ValidationError("cannot dedup an empty dataset.")

    seen: set[str] = set()
    kept_texts: list[str] = []
    kept_labels: list[int] = []
    kept_hashes: list[str] = []
    n_dropped = 0

    for text, label in zip(dataset.texts, dataset.labels, strict=True):
        digest = sentence_hash(text)
        if digest in seen:
            n_dropped += 1
            continue
        seen.add(digest)
        kept_texts.append(text)
        kept_labels.append(label)
        kept_hashes.append(digest)

    deduped = LabelledDataset(
        texts=tuple(kept_texts),
        labels=tuple(kept_labels),
        source=dataset.source,
    )
    return DedupResult(
        dataset=deduped,
        group_hashes=tuple(kept_hashes),
        n_dropped=n_dropped,
    )
