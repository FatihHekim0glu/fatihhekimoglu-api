"""Project-wide constants for the finbert-sentiment library.

Single source of truth for the 3-way sentiment label space, numerical
tolerances, and the seeded-split defaults so that no magic value is duplicated
across modules. Importing this module has no side effects.
"""

from __future__ import annotations

from typing import Final

# quantcore-candidate: mirrors risk-metrics:src/riskmetrics/_constants.py

#: Canonical ordered class names for the 3-way financial-sentiment task. The
#: order is fixed (``negative < neutral < positive``) so that label-index
#: encodings, confusion-matrix axes, and score vectors line up everywhere.
LABELS: Final[tuple[str, str, str]] = ("negative", "neutral", "positive")

#: Number of sentiment classes (always 3 for this task).
N_CLASSES: Final[int] = 3

#: Stable label -> integer-index map (matches :data:`LABELS` order).
LABEL_TO_INDEX: Final[dict[str, int]] = {label: i for i, label in enumerate(LABELS)}

#: Stable integer-index -> label map (inverse of :data:`LABEL_TO_INDEX`).
INDEX_TO_LABEL: Final[dict[int, str]] = dict(enumerate(LABELS))

#: Small positive floor used to guard divisions and log/sqrt arguments (e.g.
#: when normalizing score vectors). Chosen well above float64 round-off but far
#: below any meaningful probability mass.
EPS: Final[float] = 1e-12

#: Default master seed for the stratified group split and bootstrap resampling
#: (overridable by callers; pinned so a split is reproducible by default).
DEFAULT_SEED: Final[int] = 20260618

#: Default test-set fraction for the locked, grouped, stratified split.
DEFAULT_TEST_FRACTION: Final[float] = 0.2

#: Default validation-set fraction (carved from the non-test remainder).
DEFAULT_VAL_FRACTION: Final[float] = 0.1

#: Number of bootstrap resamples used for metric confidence intervals.
DEFAULT_BOOTSTRAP_RESAMPLES: Final[int] = 2000

#: HuggingFace dataset id + config for the Financial PhraseBank (all-annotators
#: agree subset). Documented here so the loader and README cite one source.
PHRASEBANK_DATASET: Final[str] = "financial_phrasebank"
PHRASEBANK_CONFIG: Final[str] = "sentences_allagree"
