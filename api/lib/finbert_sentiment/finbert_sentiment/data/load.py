"""Financial PhraseBank loader and the in-memory labelled-dataset container.

The loader pulls the HuggingFace ``financial_phrasebank`` dataset (config
``sentences_allagree``) and normalizes it into a :class:`LabelledDataset` of
``(texts, labels)`` with the canonical 3-way label space. ``datasets`` is
imported LAZILY inside :func:`load_phrasebank` so that ``import
finbert_sentiment`` never triggers a network call or a heavy import. Tests use an
offline-cached sample baked into the conftest fixture, never the network.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from finbert_sentiment._constants import INDEX_TO_LABEL, LABELS, N_CLASSES, PHRASEBANK_CONFIG
from finbert_sentiment._exceptions import FinbertSentimentError, ValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence

#: HuggingFace PhraseBank exposes ``label`` as ``0=negative, 1=neutral,
#: 2=positive`` which already matches our canonical :data:`LABELS` order, so the
#: identity map is used. Kept explicit so a future HF re-ordering is a one-line
#: fix rather than a silent label swap.
_HF_LABEL_TO_CANONICAL: dict[int, int] = {0: 0, 1: 1, 2: 2}

#: Parquet mirror of ``financial_phrasebank/sentences_allagree`` used as a
#: fallback when ``datasets >= 3`` rejects the canonical script-based loader
#: ("Dataset scripts are no longer supported"). The mirror ships seed-named
#: configs (each a pre-shuffled train/test split of the SAME 2264-sentence
#: allagree corpus) with the identical ``sentence`` + ``label`` (0/1/2) schema;
#: we concatenate train+test of one config to recover the full corpus and apply
#: our OWN dedup + leakage-guarded group split downstream.
_PHRASEBANK_PARQUET_MIRROR: str = "gtfintechlab/financial_phrasebank_sentences_allagree"
_PHRASEBANK_MIRROR_CONFIG: str = "5768"


def _load_canonical(
    load_dataset: Any, cfg: str, cache_dir: str | None
) -> tuple[list[str], list[int], str]:
    """Load the canonical script-based ``financial_phrasebank`` (older ``datasets``)."""
    from finbert_sentiment._constants import PHRASEBANK_DATASET

    ds = load_dataset(PHRASEBANK_DATASET, cfg, split="train", cache_dir=cache_dir)
    return list(ds["sentence"]), list(ds["label"]), f"{PHRASEBANK_DATASET}/{cfg}"


def _load_parquet_mirror(
    load_dataset: Any, cache_dir: str | None
) -> tuple[list[str], list[int], str]:
    """Load the parquet mirror and recover the full allagree corpus (train+test).

    The mirror's seed-named configs are each a pre-shuffled split of the SAME
    2264-sentence corpus, so concatenating one config's ``train`` and ``test``
    folds reconstructs the full set; our own dedup + group split partitions it.
    """
    texts: list[str] = []
    labels: list[int] = []
    for split in ("train", "test"):
        ds = load_dataset(
            _PHRASEBANK_PARQUET_MIRROR,
            _PHRASEBANK_MIRROR_CONFIG,
            split=split,
            cache_dir=cache_dir,
        )
        texts.extend(str(s) for s in ds["sentence"])
        labels.extend(int(v) for v in ds["label"])
    return texts, labels, f"{_PHRASEBANK_PARQUET_MIRROR}/{_PHRASEBANK_MIRROR_CONFIG}"


@dataclass(frozen=True, slots=True)
class LabelledDataset:
    """An immutable, aligned ``(texts, labels)`` corpus with provenance.

    Attributes
    ----------
    texts:
        The raw sentences, one per example (parallel to ``labels``).
    labels:
        Integer class indices in ``{0, 1, 2}`` (see
        :data:`finbert_sentiment._constants.LABELS`), one per text.
    source:
        Human-readable provenance string (e.g.
        ``"financial_phrasebank/sentences_allagree"`` or
        ``"offline-cached-sample"``).
    """

    texts: tuple[str, ...]
    labels: tuple[int, ...]
    source: str = "unknown"

    def __post_init__(self) -> None:
        """Validate that ``texts`` and ``labels`` are aligned and non-empty."""
        if len(self.texts) == 0:
            raise ValidationError("LabelledDataset must contain at least one example.")
        if len(self.texts) != len(self.labels):
            raise ValidationError(
                f"texts/labels misaligned: {len(self.texts)} texts vs {len(self.labels)} labels."
            )
        for i, text in enumerate(self.texts):
            if not isinstance(text, str):
                raise ValidationError(f"texts[{i}] must be a str, got {type(text).__name__}.")
            if not text.strip():
                raise ValidationError(f"texts[{i}] must not be blank.")
        for i, label in enumerate(self.labels):
            if isinstance(label, bool) or not isinstance(label, int):
                raise ValidationError(
                    f"labels[{i}] must be an int index, got {type(label).__name__}."
                )
            if not 0 <= label < N_CLASSES:
                raise ValidationError(f"labels[{i}]={label} is out of range [0, {N_CLASSES}).")

    @property
    def n(self) -> int:
        """Return the number of examples."""
        return len(self.texts)

    def class_counts(self) -> dict[str, int]:
        """Return a ``{class_name: count}`` mapping over the 3-way label space."""
        counts = {label: 0 for label in LABELS}
        for label_idx in self.labels:
            counts[INDEX_TO_LABEL[label_idx]] += 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this dataset."""
        return asdict(self)


def load_phrasebank(
    *,
    config: str | None = None,
    cache_dir: str | None = None,
) -> LabelledDataset:
    """Load the Financial PhraseBank into a :class:`LabelledDataset`.

    LAZY IMPORT: ``datasets`` is imported inside this function. The HuggingFace
    label ids (``0=negative, 1=neutral, 2=positive`` for PhraseBank) are mapped
    onto the canonical :data:`finbert_sentiment._constants.LABELS` order.

    Parameters
    ----------
    config:
        Dataset config name (defaults to
        :data:`finbert_sentiment._constants.PHRASEBANK_CONFIG`).
    cache_dir:
        Optional HuggingFace datasets cache directory.

    Returns
    -------
    LabelledDataset
        The full PhraseBank corpus (pre-dedup), tagged with its source.

    Raises
    ------
    FinbertSentimentError
        If the dataset cannot be downloaded or parsed.
    """
    cfg = config if config is not None else PHRASEBANK_CONFIG
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only without [data]
        raise FinbertSentimentError(
            "the 'datasets' package is required to load the Financial PhraseBank; "
            "install the [data] extra."
        ) from exc

    from finbert_sentiment._constants import PHRASEBANK_DATASET

    try:
        raw_texts, raw_labels, source = _load_canonical(load_dataset, cfg, cache_dir)
    except Exception as canonical_exc:
        # ``datasets >= 3`` rejects the script-based canonical loader. Fall back
        # to the parquet mirror of the same 2264-sentence allagree corpus.
        try:
            raw_texts, raw_labels, source = _load_parquet_mirror(load_dataset, cache_dir)
        except Exception as mirror_exc:  # normalize any HF/IO failure to our error type
            raise FinbertSentimentError(
                f"failed to load {PHRASEBANK_CONFIG!r} from the Financial PhraseBank "
                f"(canonical {PHRASEBANK_DATASET!r}: {canonical_exc}; "
                f"parquet mirror {_PHRASEBANK_PARQUET_MIRROR!r}: {mirror_exc})."
            ) from mirror_exc

    texts: list[str] = []
    labels: list[int] = []
    for text, hf_label in zip(raw_texts, raw_labels, strict=True):
        canonical = _HF_LABEL_TO_CANONICAL.get(int(hf_label))
        if canonical is None:
            raise FinbertSentimentError(
                f"unexpected PhraseBank label id {hf_label!r}; "
                f"expected one of {sorted(_HF_LABEL_TO_CANONICAL)}."
            )
        texts.append(str(text))
        labels.append(canonical)

    return LabelledDataset(
        texts=tuple(texts),
        labels=tuple(labels),
        source=source,
    )


def sample_dataset(
    texts: Sequence[str],
    labels: Sequence[int | str],
    *,
    source: str = "offline-cached-sample",
) -> LabelledDataset:
    """Build a :class:`LabelledDataset` from in-memory texts/labels (no network).

    The constructor used by the offline test fixture and by the FastAPI demo's
    bundled example headlines. String labels are normalized to integer indices.

    Parameters
    ----------
    texts:
        The sentences.
    labels:
        Integer indices or canonical string class names, one per text.
    source:
        Provenance tag stored on the returned dataset.

    Returns
    -------
    LabelledDataset
        The validated, aligned corpus.

    Raises
    ------
    ValidationError
        If the inputs are misaligned, empty, or contain invalid labels.
    """
    from finbert_sentiment._validation import ensure_labels

    if isinstance(texts, str):
        raise ValidationError("texts must be a sequence of strings, not a bare str.")
    text_list = list(texts)
    if len(text_list) == 0:
        raise ValidationError("texts must be non-empty.")
    label_arr = ensure_labels(labels, n_expected=len(text_list))
    return LabelledDataset(
        texts=tuple(text_list),
        labels=tuple(int(v) for v in label_arr.tolist()),
        source=source,
    )
