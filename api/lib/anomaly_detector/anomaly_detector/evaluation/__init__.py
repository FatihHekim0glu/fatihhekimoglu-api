"""Descriptive evaluation: detector agreement, proxy labels, and the toy overlay.

- :mod:`agreement` — Jaccard/overlap between detectors, regime alignment with
  known stress windows, and precision/recall against a transparent
  ``|z-return| > 3`` proxy label (the DESCRIPTIVE headline).
- :mod:`overlay` — the optional, clearly-labeled-diagnostic fade-the-anomaly toy
  with a Deflated Sharpe over the full configuration grid.
- :mod:`dsr` — Probabilistic / Deflated Sharpe (Bailey & Lopez de Prado 2014),
  reused verbatim for the overlay's multiplicity-aware yardstick.

Importing this subpackage has no side effects.
"""

from __future__ import annotations

from anomaly_detector.evaluation.agreement import (
    AgreementResult,
    compute_agreement,
    jaccard_index,
    proxy_precision_recall,
    regime_alignment,
)
from anomaly_detector.evaluation.dsr import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from anomaly_detector.evaluation.overlay import OverlayResult, fade_the_anomaly_overlay

__all__ = [
    "AgreementResult",
    "OverlayResult",
    "compute_agreement",
    "deflated_sharpe_ratio",
    "fade_the_anomaly_overlay",
    "jaccard_index",
    "probabilistic_sharpe_ratio",
    "proxy_precision_recall",
    "regime_alignment",
]
