"""Data layer: PhraseBank loading, sentence-level dedup, and seeded group split.

This subpackage owns everything between "raw Financial PhraseBank" and a locked,
leakage-safe train/val/test split. Importing it has no side effects and pulls in
no heavy dependency (``datasets`` is imported lazily inside the loader).
"""

from __future__ import annotations

from finbert_sentiment.data.dedup import (
    DedupResult,
    dedup_sentences,
    normalize_sentence,
    sentence_hash,
)
from finbert_sentiment.data.load import LabelledDataset, load_phrasebank, sample_dataset
from finbert_sentiment.data.split import (
    SplitIndices,
    assert_no_group_overlap,
    stratified_group_split,
)

__all__ = [
    "DedupResult",
    "LabelledDataset",
    "SplitIndices",
    "assert_no_group_overlap",
    "dedup_sentences",
    "load_phrasebank",
    "normalize_sentence",
    "sample_dataset",
    "sentence_hash",
    "stratified_group_split",
]
