"""DistilBERT 3-way fine-tune on the Financial PhraseBank (``[train]`` only).

This module fine-tunes ``distilbert-base-uncased`` for 3-way sentiment using the
HuggingFace ``Trainer``, seeded and early-stopped on validation macro-F1. torch
and transformers are imported LAZILY inside :func:`train_distilbert` so that
``import finbert_sentiment`` never pulls in a deep-learning framework; the serve
container never installs ``[train]`` and never imports this module.

The label encoder and any class-weight statistics are computed on TRAIN ONLY
(the pretrained tokenizer is frozen, so it touching all text is fine). Importing
this module has no side effects.
"""

from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from finbert_sentiment._constants import DEFAULT_SEED, LABELS, N_CLASSES
from finbert_sentiment._exceptions import FinbertSentimentError, InsufficientDataError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from finbert_sentiment.data.load import LabelledDataset
    from finbert_sentiment.data.split import SplitIndices


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Immutable hyper-parameters for the DistilBERT fine-tune.

    Attributes
    ----------
    model_name:
        HuggingFace base checkpoint to fine-tune.
    epochs:
        Maximum training epochs (early-stopping may stop sooner).
    batch_size:
        Per-device train/eval batch size.
    learning_rate:
        AdamW peak learning rate.
    max_length:
        Tokenizer truncation length.
    early_stopping_patience:
        Epochs without val-macro-F1 improvement before stopping.
    seed:
        Master seed (set on torch, numpy, and the Trainer).
    """

    model_name: str = "distilbert-base-uncased"
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    max_length: int = 128
    early_stopping_patience: int = 1
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        """Validate the hyper-parameters are in admissible ranges."""
        if not self.model_name.strip():
            raise FinbertSentimentError("model_name must be a non-empty string.")
        if self.epochs < 1:
            raise FinbertSentimentError(f"epochs must be >= 1, got {self.epochs}.")
        if self.batch_size < 1:
            raise FinbertSentimentError(f"batch_size must be >= 1, got {self.batch_size}.")
        if not self.learning_rate > 0.0:
            raise FinbertSentimentError(f"learning_rate must be > 0, got {self.learning_rate}.")
        if self.max_length < 1:
            raise FinbertSentimentError(f"max_length must be >= 1, got {self.max_length}.")
        if self.early_stopping_patience < 0:
            raise FinbertSentimentError(
                f"early_stopping_patience must be >= 0, got {self.early_stopping_patience}."
            )
        if self.seed < 0:
            raise FinbertSentimentError(f"seed must be non-negative, got {self.seed}.")

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this config."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Immutable outcome of a fine-tune run.

    Attributes
    ----------
    output_dir:
        Directory holding the saved torch model + tokenizer.
    best_val_macro_f1:
        Best validation macro-F1 reached during training.
    epochs_run:
        Number of epochs actually executed (<= ``config.epochs``).
    config:
        The :class:`TrainConfig` used.
    label_order:
        The class-name order the model's logits correspond to.
    """

    output_dir: str
    best_val_macro_f1: float
    epochs_run: int
    config: TrainConfig
    label_order: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` (config flattened)."""
        out: dict[str, Any] = asdict(self)
        out["config"] = self.config.to_dict()
        return out


def train_available() -> bool:
    """Return ``True`` iff the heavy ``[train]`` extra (torch + transformers) is importable.

    Probes with :func:`importlib.util.find_spec` so the check itself never imports
    torch/transformers and the module stays import-pure. The CLI and the export
    path call this to decide whether the fine-tune branch can run here at all.

    Returns
    -------
    bool
        Whether both ``torch`` and ``transformers`` can be imported.
    """
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("transformers") is not None
    )


def build_label_arrays(
    dataset: LabelledDataset,
    split: SplitIndices,
) -> tuple[list[str], list[int], list[str], list[int]]:
    """Slice ``(train_texts, train_labels, val_texts, val_labels)`` from a locked split.

    A pure, torch-free helper (the "dry-run config builder" the tests exercise
    without any deep-learning framework). The locked TEST fold is deliberately NOT
    returned: training and early stopping must never see it. Train/val label index
    vectors are read straight off the dataset; the canonical label *order* is fixed
    by :data:`finbert_sentiment._constants.LABELS`, so no encoder is fit on text.

    Parameters
    ----------
    dataset:
        The deduplicated labelled corpus.
    split:
        The locked train/val/test partition (only train + val are used).

    Returns
    -------
    tuple[list[str], list[int], list[str], list[int]]
        ``(train_texts, train_labels, val_texts, val_labels)``.

    Raises
    ------
    InsufficientDataError
        If either the train or val fold is empty, or the train fold does not
        contain every one of the ``N_CLASSES`` classes (the model could not learn
        a class it never sees).
    """
    texts = dataset.texts
    labels = dataset.labels
    n = len(texts)
    for fold_name, idxs in (("train", split.train), ("val", split.val)):
        for idx in idxs:
            if not 0 <= idx < n:
                raise FinbertSentimentError(
                    f"{fold_name} index {idx} is out of range for {n} examples."
                )

    train_texts = [texts[i] for i in split.train]
    train_labels = [labels[i] for i in split.train]
    val_texts = [texts[i] for i in split.val]
    val_labels = [labels[i] for i in split.val]

    if len(train_texts) == 0:
        raise InsufficientDataError("train fold is empty; cannot fine-tune.")
    if len(val_texts) == 0:
        raise InsufficientDataError("val fold is empty; cannot early-stop on val macro-F1.")
    present = set(train_labels)
    missing = set(range(N_CLASSES)) - present
    if missing:
        missing_names = sorted(LABELS[i] for i in missing)
        raise InsufficientDataError(
            f"train fold is missing class(es) {missing_names}; "
            "every class must appear in train to be learned."
        )
    return train_texts, train_labels, val_texts, val_labels


def _compute_macro_f1(label_ids: Sequence[int], pred_ids: Sequence[int]) -> float:
    """Return the (sklearn-free) macro-F1 over the 3-way label space.

    Kept dependency-light (numpy only) so the Trainer's compute-metrics callback
    introduces no sklearn requirement on the ``[train]`` path.
    """
    import numpy as np

    y_true = np.asarray(list(label_ids), dtype=np.int64)
    y_pred = np.asarray(list(pred_ids), dtype=np.int64)
    f1s: list[float] = []
    for cls in range(N_CLASSES):
        tp = int(np.sum((y_pred == cls) & (y_true == cls)))
        fp = int(np.sum((y_pred == cls) & (y_true != cls)))
        fn = int(np.sum((y_pred != cls) & (y_true == cls)))
        denom = 2 * tp + fp + fn
        f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1s))


def train_distilbert(
    dataset: LabelledDataset,
    split: SplitIndices,
    config: TrainConfig | None = None,
    *,
    output_dir: str,
) -> TrainResult:
    """Fine-tune DistilBERT on the PhraseBank train fold, early-stopped on val macro-F1.

    LAZY IMPORT: ``torch`` and ``transformers`` are imported inside this function.
    The val fold drives early stopping; the locked test fold is NOT touched here.

    Parameters
    ----------
    dataset:
        The deduplicated labelled corpus.
    split:
        The locked train/val/test partition (only train+val are used here).
    config:
        Hyper-parameters (defaults to :class:`TrainConfig`).
    output_dir:
        Where to save the best model + tokenizer.

    Returns
    -------
    TrainResult
        The training outcome (best val macro-F1, epochs run, saved-model dir).

    Raises
    ------
    FinbertSentimentError
        If the ``[train]`` dependencies are missing or training fails.
    """
    import os

    cfg = config if config is not None else TrainConfig()

    # Pure, torch-free slicing + leakage guards FIRST: a bad split fails fast
    # before we pay the cost of importing torch / downloading a checkpoint.
    train_texts, train_labels, val_texts, val_labels = build_label_arrays(dataset, split)

    if not train_available():  # pragma: no cover - exercised only without [train]
        raise FinbertSentimentError(
            "the [train] extra (torch + transformers) is required to fine-tune "
            "DistilBERT; install it with `uv pip install -e '.[train]'`. The lexicon "
            "baseline is the torch-free LIVE served model and needs none of this."
        )

    try:  # pragma: no cover - requires the heavy [train] extra (torch/transformers)
        import numpy as np
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
            set_seed,
        )
        from transformers.trainer_utils import EvalPrediction

        set_seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        id2label = dict(enumerate(LABELS))
        label2id = {name: i for i, name in id2label.items()}

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name,
            num_labels=N_CLASSES,
            id2label=id2label,
            label2id=label2id,
        )

        def _encode(texts: list[str], labels: list[int]) -> list[dict[str, Any]]:
            enc = tokenizer(
                texts,
                truncation=True,
                max_length=cfg.max_length,
                padding=False,
            )
            rows: list[dict[str, Any]] = []
            for i, label in enumerate(labels):
                row = {key: enc[key][i] for key in enc}
                row["labels"] = int(label)
                rows.append(row)
            return rows

        train_rows = _encode(train_texts, train_labels)
        val_rows = _encode(val_texts, val_labels)

        def _compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
            logits = eval_pred.predictions
            if isinstance(logits, tuple):
                logits = logits[0]
            preds: list[int] = [int(v) for v in np.asarray(logits).argmax(axis=-1).tolist()]
            label_ids: list[int] = [int(v) for v in np.asarray(eval_pred.label_ids).tolist()]
            return {"macro_f1": _compute_macro_f1(label_ids, preds)}

        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            save_total_limit=1,
            seed=cfg.seed,
            data_seed=cfg.seed,
            report_to=[],
            disable_tqdm=True,
        )

        from transformers import TrainerCallback

        callbacks: list[TrainerCallback] = []
        if cfg.early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)
            )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_rows,
            eval_dataset=val_rows,
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=_compute_metrics,
            callbacks=callbacks,
        )

        trainer.train()

        best_metrics = trainer.evaluate()
        best_val_macro_f1 = float(best_metrics.get("eval_macro_f1", 0.0))
        epochs_run = round(float(trainer.state.epoch or 0.0))

        os.makedirs(output_dir, exist_ok=True)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
    except FinbertSentimentError:
        raise
    except Exception as exc:  # pragma: no cover - heavy [train]-only path
        raise FinbertSentimentError(f"DistilBERT fine-tune failed: {exc}") from exc

    return TrainResult(  # pragma: no cover - heavy [train]-only path
        output_dir=output_dir,
        best_val_macro_f1=best_val_macro_f1,
        epochs_run=epochs_run,
        config=cfg,
        label_order=LABELS,
    )
