"""Shared, frozen result dataclass for every anomaly detector.

:class:`AnomalyResult` is the single immutable container returned by both the
Isolation Forest and the PCA-reconstruction autoencoder. It carries the
out-of-sample per-day anomaly scores, the boolean flags derived from a
train-fitted threshold, and the threshold itself, plus provenance metadata
(detector name, contamination, train/test span sizes). Its :meth:`to_dict`
renders to plain JSON-serializable types so the result crosses the API boundary
cleanly.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


def _iso(label: object) -> str:
    """Render an index label as an ISO date string when possible, else ``str``.

    Pandas ``Timestamp``/``datetime`` labels expose ``.isoformat``; everything
    else (a positional int index, a plain string) falls back to ``str`` so the
    dict keys are always JSON-safe regardless of how the OOS slice is indexed.
    """
    isoformat = getattr(label, "isoformat", None)
    if callable(isoformat):
        return str(isoformat())
    return str(label)


def _safe_float(value: object) -> float | None:
    """Coerce ``value`` to a finite float, mapping NaN/Inf/None to ``None``.

    The API contract forbids non-finite floats in JSON; this helper is the single
    chokepoint every scalar passes through on the way out.

    Parameters
    ----------
    value:
        Any scalar coercible to ``float``.

    Returns
    -------
    float | None
        The finite float, or ``None`` for NaN/Inf/non-numeric input.
    """
    import math

    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


@dataclass(frozen=True, slots=True)
class AnomalyResult:
    """Immutable out-of-sample result of a single detector.

    The score and flag series are indexed by the OUT-OF-SAMPLE dates only; the
    threshold was fitted on the disjoint TRAIN slice (full-sample-leakage guard).
    A day is flagged anomalous when its score exceeds ``threshold``.

    Attributes
    ----------
    scores:
        Per-day anomaly score over the OOS slice (higher = more anomalous),
        indexed by date.
    flags:
        Boolean per-day flag over the OOS slice (``True`` = anomalous), indexed
        by the same dates as ``scores``. A day's flag uses only information
        available strictly before it (``.shift(1)`` chokepoint upstream).
    threshold:
        The score cutoff fitted on the TRAIN slice (the contamination quantile
        for Isolation Forest, the train-error quantile for the autoencoder).
    detector:
        The detector name (``"iforest"`` or ``"autoencoder"``).
    contamination:
        The contamination / quantile parameter used to set the threshold.
    n_train:
        Number of TRAIN-slice observations the detector was fitted on.
    n_test:
        Number of OOS observations scored.
    meta:
        Free-form JSON-serializable provenance (e.g. feature names, n_components).
    """

    scores: pd.Series
    flags: pd.Series
    threshold: float
    detector: str
    contamination: float
    n_train: int
    n_test: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result.

        Series are rendered with ISO-formatted date keys; scores pass through
        :func:`_safe_float` (NaN/Inf -> ``None``) and flags through ``bool``.

        Returns
        -------
        dict[str, Any]
            A mapping with ``scores``, ``flags``, ``threshold``, ``detector``,
            ``contamination``, ``n_train``, ``n_test``, and ``meta`` keys.
        """
        return {
            "scores": {_iso(k): _safe_float(v) for k, v in self.scores.items()},
            "flags": {_iso(k): bool(v) for k, v in self.flags.items()},
            "threshold": _safe_float(self.threshold),
            "detector": str(self.detector),
            "contamination": _safe_float(self.contamination),
            "n_train": int(self.n_train),
            "n_test": int(self.n_test),
            "meta": dict(self.meta),
        }

    def flagged_dates(self) -> list[str]:
        """Return the ISO-formatted dates flagged anomalous, in score order.

        Returns
        -------
        list[str]
            The OOS dates where ``flags`` is ``True``, sorted by descending
            anomaly score (most anomalous first).
        """
        flagged = self.flags.fillna(False).astype(bool)
        if not bool(flagged.any()):
            return []
        flagged_index = self.flags.index[flagged.to_numpy()]
        ordered = self.scores.reindex(flagged_index).sort_values(ascending=False)
        return [_iso(label) for label in ordered.index]
