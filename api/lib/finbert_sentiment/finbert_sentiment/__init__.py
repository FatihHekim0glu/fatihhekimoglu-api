"""finbert-sentiment — a from-scratch 3-way financial-sentiment classifier.

A DistilBERT fine-tune on the Financial PhraseBank (``sentences_allagree``),
served via ONNX + onnxruntime, benchmarked HONESTLY against a class-prior
baseline and a Loughran-McDonald-style lexicon baseline. The headline metric is
**macro-F1** (with per-class precision/recall, a confusion matrix, and bootstrap
CIs) — never accuracy alone, because the PhraseBank neutral class is ~60%. The
lexicon baseline doubles as a torch-free LIVE served model the tool falls back to
when no ONNX artifact is present.

Honesty notes baked into the design:

* Sentiment is a TEXT LABEL, not a tradable signal — no alpha is claimed.
* Walk-forward / purge / Deflated-Sharpe DO NOT apply here (there is no return
  series); using them would be a category error.
* A transformer macro-F1 is reported ONLY if it was actually measured in this
  build; otherwise the published ProsusAI/finbert figure is cited as
  "expected, not measured here".

The package has ZERO import-time side effects and ZERO UI coupling: importing it
pulls in no torch / transformers / onnxruntime / network call (those are imported
lazily, behind functions). The same functions back a CLI and a hosted FastAPI
tool unchanged. Public API is curated below; see :data:`__all__`.
"""

from __future__ import annotations

from finbert_sentiment._constants import (
    EPS,
    INDEX_TO_LABEL,
    LABEL_TO_INDEX,
    LABELS,
    N_CLASSES,
    PHRASEBANK_CONFIG,
    PHRASEBANK_DATASET,
)
from finbert_sentiment._exceptions import (
    ArtifactError,
    FinbertSentimentError,
    InsufficientDataError,
    ValidationError,
)
from finbert_sentiment._manifest import RunManifest, config_hash
from finbert_sentiment._rng import make_rng, spawn_substreams
from finbert_sentiment._validation import (
    ensure_labels,
    ensure_score_matrix,
    ensure_text_batch,
    validate_min_per_class,
)
from finbert_sentiment.baselines import ClassPriorClassifier, LexiconClassifier
from finbert_sentiment.data import (
    DedupResult,
    LabelledDataset,
    SplitIndices,
    dedup_sentences,
    load_phrasebank,
    normalize_sentence,
    sample_dataset,
    stratified_group_split,
)
from finbert_sentiment.evaluation import (
    ClassificationReport,
    McNemarResult,
    Verdict,
    VerdictResult,
    bootstrap_macro_f1_ci,
    confusion_matrix,
    derive_verdict,
    macro_f1,
    mcnemar_test,
    per_class_precision_recall_f1,
)
from finbert_sentiment.inference import Prediction, Predictor, load_predictor
from finbert_sentiment.plots import confusion_matrix_figure, per_class_f1_figure
from finbert_sentiment.service import (
    SentimentResult,
    SentimentSummary,
    build_evaluation_figures,
    load_committed_metrics,
    run_sentiment,
)

__version__ = "0.1.0"

#: Curated public API (sorted; see the import groups above for provenance:
#: constants, exceptions, reproducibility, validation, data, baselines,
#: inference, evaluation, plots).
__all__ = [
    "EPS",
    "INDEX_TO_LABEL",
    "LABELS",
    "LABEL_TO_INDEX",
    "N_CLASSES",
    "PHRASEBANK_CONFIG",
    "PHRASEBANK_DATASET",
    "ArtifactError",
    "ClassPriorClassifier",
    "ClassificationReport",
    "DedupResult",
    "FinbertSentimentError",
    "InsufficientDataError",
    "LabelledDataset",
    "LexiconClassifier",
    "McNemarResult",
    "Prediction",
    "Predictor",
    "RunManifest",
    "SentimentResult",
    "SentimentSummary",
    "SplitIndices",
    "ValidationError",
    "Verdict",
    "VerdictResult",
    "__version__",
    "bootstrap_macro_f1_ci",
    "build_evaluation_figures",
    "config_hash",
    "confusion_matrix",
    "confusion_matrix_figure",
    "dedup_sentences",
    "derive_verdict",
    "ensure_labels",
    "ensure_score_matrix",
    "ensure_text_batch",
    "load_committed_metrics",
    "load_phrasebank",
    "load_predictor",
    "macro_f1",
    "make_rng",
    "mcnemar_test",
    "normalize_sentence",
    "per_class_f1_figure",
    "per_class_precision_recall_f1",
    "run_sentiment",
    "sample_dataset",
    "spawn_substreams",
    "stratified_group_split",
    "validate_min_per_class",
]
