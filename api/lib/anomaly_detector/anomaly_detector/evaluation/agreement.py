"""Detector-agreement and proxy-label diagnostics.

The HEADLINE of this tool is descriptive, not predictive: how much do the two
independent detectors agree, and how well do their flags line up with a
TRANSPARENT proxy label and a set of KNOWN macro-stress windows? There is no
ground-truth anomaly label, so nothing here implies a tradable signal.

Quantities computed
-------------------
- Jaccard / overlap between the two detectors' OOS flag sets (the primary
  stability metric — expected to be MODEST, ~0.3-0.5).
- Regime alignment: overlap of each detector's flags with KNOWN stress windows
  (e.g. 2020-03, 2018-Q4, the 2022 selloff) or a VIX-spike proxy.
- Precision / recall of the flags against a transparent proxy label
  ``|causal z-return| > 3`` (expected LOW precision — flags are diagnostic).

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from anomaly_detector.detectors.result import _safe_float

if TYPE_CHECKING:
    from anomaly_detector.detectors.result import AnomalyResult

#: Z-score cutoff for the transparent proxy anomaly label ``|z-return| > Z``.
PROXY_Z_THRESHOLD: float = 3.0

#: Known macro-stress windows (inclusive ISO date ranges) used for regime
#: alignment. These are TRANSPARENT, well-documented historical stress periods,
#: NOT a fitted ground-truth label.
KNOWN_STRESS_WINDOWS: tuple[tuple[str, str], ...] = (
    ("2018-10-01", "2018-12-31"),  # 2018-Q4 selloff
    ("2020-02-20", "2020-04-30"),  # COVID crash
    ("2022-01-01", "2022-10-31"),  # 2022 rate-driven selloff
)


@dataclass(frozen=True, slots=True)
class AgreementResult:
    """Immutable descriptive-agreement summary for a pair of detectors.

    Attributes
    ----------
    jaccard:
        Jaccard index of the two detectors' OOS flag sets
        ``|A and B| / |A or B|`` (the primary stability metric).
    overlap_count:
        Number of OOS days flagged by BOTH detectors (``|A and B|``).
    n_flags_a:
        Number of OOS days flagged by detector A.
    n_flags_b:
        Number of OOS days flagged by detector B.
    proxy_precision:
        Precision of the (intersection or union) flags against the
        ``|z-return| > PROXY_Z_THRESHOLD`` proxy label (expected LOW).
    proxy_recall:
        Recall of those flags against the same proxy label.
    regime_alignment:
        Fraction of flagged days that fall inside a KNOWN stress window.
    top_anomaly_dates:
        The most anomalous OOS dates (ISO strings), agreed by both detectors
        where possible, for the summary block.
    meta:
        Free-form JSON-serializable provenance.
    """

    jaccard: float
    overlap_count: int
    n_flags_a: int
    n_flags_b: int
    proxy_precision: float
    proxy_recall: float
    regime_alignment: float
    top_anomaly_dates: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this summary.

        Scalars pass through :func:`_safe_float` (NaN/Inf -> ``None``); counts are
        plain ``int``; ``top_anomaly_dates`` is a list of ISO strings.

        Returns
        -------
        dict[str, Any]
            A mapping with ``jaccard``, ``overlap_count``, ``n_flags_a``,
            ``n_flags_b``, ``proxy_precision``, ``proxy_recall``,
            ``regime_alignment``, ``top_anomaly_dates``, and ``meta`` keys.
        """
        return {
            "jaccard": _safe_float(self.jaccard),
            "overlap_count": int(self.overlap_count),
            "n_flags_a": int(self.n_flags_a),
            "n_flags_b": int(self.n_flags_b),
            "proxy_precision": _safe_float(self.proxy_precision),
            "proxy_recall": _safe_float(self.proxy_recall),
            "regime_alignment": _safe_float(self.regime_alignment),
            "top_anomaly_dates": [str(d) for d in self.top_anomaly_dates],
            "meta": dict(self.meta),
        }


def _flagged_dates(flags: pd.Series) -> set[Any]:
    """Return the set of index labels where ``flags`` is truthy.

    NaN entries are treated as not-flagged, so a partially-warm-up flag series
    contributes only its concrete ``True`` days to the set.
    """
    mask = flags.fillna(False).astype(bool)
    return set(mask.index[mask.to_numpy()])


def jaccard_index(flags_a: pd.Series, flags_b: pd.Series) -> float:
    """Jaccard index of two boolean OOS flag series.

    Returns ``|A and B| / |A or B|`` over the aligned dates, where ``A``/``B``
    are the sets of dates each detector flagged. Defined as ``0.0`` when neither
    detector flags anything (empty union).

    Parameters
    ----------
    flags_a, flags_b:
        Boolean per-day flag series, aligned (or alignable) on their date index.

    Returns
    -------
    float
        The Jaccard index in ``[0, 1]``.
    """
    set_a = _flagged_dates(flags_a)
    set_b = _flagged_dates(flags_b)
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def proxy_precision_recall(
    flags: pd.Series,
    returns: pd.Series,
    *,
    window: int,
    z_threshold: float = PROXY_Z_THRESHOLD,
) -> tuple[float, float]:
    """Precision/recall of ``flags`` against the ``|causal z-return| > z`` proxy.

    The proxy label is TRANSPARENT and causal (built from
    :func:`anomaly_detector.features.engineer.causal_return_zscore`), NOT a
    ground-truth anomaly label. Low precision is the honest expected outcome.

    Parameters
    ----------
    flags:
        Boolean per-day flag series over the OOS slice.
    returns:
        The per-day return series over the same span (for the proxy label).
    window:
        Rolling window for the causal z-score.
    z_threshold:
        Absolute z-score cutoff defining the proxy positive class.

    Returns
    -------
    tuple[float, float]
        ``(precision, recall)`` of the flags against the proxy label.

    Raises
    ------
    ValidationError
        If ``window`` is not an int ``>= 2`` (propagated from the z-score helper).
    """
    from anomaly_detector.features.engineer import causal_return_zscore

    # TRANSPARENT, CAUSAL proxy label: a day is a "proxy anomaly" when its return,
    # standardized by a strictly-trailing rolling mean/std (no peeking at the day
    # it normalizes), is more than ``z_threshold`` sigmas from zero. This is NOT a
    # ground-truth label; low precision against it is the honest expected outcome.
    zscore = causal_return_zscore(returns, window=window)
    proxy_label = zscore.abs() > z_threshold  # NaN warm-up rows -> False

    # Restrict the comparison to dates where BOTH the flag and the proxy are
    # defined (the proxy warm-up rows carry no information either way).
    common = flags.index.intersection(zscore.index)
    defined = common[zscore.reindex(common).notna().to_numpy()]
    if len(defined) == 0:
        return 0.0, 0.0

    flagged = flags.reindex(defined).fillna(False).astype(bool)
    positive = proxy_label.reindex(defined).fillna(False).astype(bool)

    flagged_arr = flagged.to_numpy()
    positive_arr = positive.to_numpy()
    true_positive = int((flagged_arr & positive_arr).sum())
    n_flagged = int(flagged_arr.sum())
    n_positive = int(positive_arr.sum())

    precision = true_positive / n_flagged if n_flagged > 0 else 0.0
    recall = true_positive / n_positive if n_positive > 0 else 0.0
    return precision, recall


def regime_alignment(
    flags: pd.Series,
    *,
    windows: tuple[tuple[str, str], ...] = KNOWN_STRESS_WINDOWS,
) -> float:
    """Fraction of flagged days that fall inside a KNOWN stress window.

    Parameters
    ----------
    flags:
        Boolean per-day flag series over the OOS slice.
    windows:
        Inclusive ISO date ranges of known macro-stress periods.

    Returns
    -------
    float
        The fraction of flagged days inside any window (``0.0`` if no flags).
    """
    flagged = sorted(_flagged_dates(flags))
    if not flagged:
        return 0.0

    bounds = [(pd.Timestamp(lo), pd.Timestamp(hi)) for lo, hi in windows]
    in_window = 0
    for label in flagged:
        ts = pd.Timestamp(label)
        if any(lo <= ts <= hi for lo, hi in bounds):
            in_window += 1
    return in_window / len(flagged)


def _top_anomaly_dates(
    result_a: AnomalyResult,
    result_b: AnomalyResult,
    *,
    n_top: int = 10,
) -> list[str]:
    """ISO dates of the strongest anomalies, agreed dates first.

    Ranks by the mean of the two detectors' rank-normalized scores so the two
    very different score scales are comparable, and lists days flagged by BOTH
    detectors ahead of days flagged by only one.
    """
    scores_a = result_a.scores
    scores_b = result_b.scores
    common = scores_a.index.intersection(scores_b.index)
    if len(common) == 0:
        return []

    # Rank-normalize each detector's scores to [0, 1] before averaging so neither
    # scale dominates (IsolationForest and the AE produce different magnitudes).
    rank_a = scores_a.reindex(common).rank(pct=True)
    rank_b = scores_b.reindex(common).rank(pct=True)
    combined = ((rank_a + rank_b) / 2.0).sort_values(ascending=False)

    set_a = _flagged_dates(result_a.flags)
    set_b = _flagged_dates(result_b.flags)
    both = set_a & set_b
    either = set_a | set_b

    # Stable two-tier ordering: agreed-upon flags first (by combined score), then
    # the remaining single-detector flags (by combined score).
    agreed = [d for d in combined.index if d in both]
    single = [d for d in combined.index if d in either and d not in both]
    ordered = agreed + single
    return [pd.Timestamp(d).date().isoformat() for d in ordered[:n_top]]


def compute_agreement(
    result_a: AnomalyResult,
    result_b: AnomalyResult,
    returns: pd.Series,
    *,
    window: int,
) -> AgreementResult:
    """Assemble the full descriptive-agreement summary for two detector results.

    Combines the Jaccard index, proxy precision/recall, and regime alignment
    into a single :class:`AgreementResult` for the API summary block. Proxy
    precision/recall and regime alignment are measured against the UNION of the
    two detectors' flags (the tool's overall flag set), while ``jaccard`` measures
    the pairwise agreement between them.

    Parameters
    ----------
    result_a, result_b:
        The two detectors' :class:`~anomaly_detector.detectors.result.AnomalyResult`
        objects over the SAME OOS slice.
    returns:
        The per-day return series over the OOS slice (for the proxy label).
    window:
        Rolling window for the causal z-score proxy.

    Returns
    -------
    AgreementResult
        The descriptive agreement summary.
    """
    set_a = _flagged_dates(result_a.flags)
    set_b = _flagged_dates(result_b.flags)

    jaccard = jaccard_index(result_a.flags, result_b.flags)

    # Combined (union) flag series over the aligned OOS dates: a day is in the
    # tool's overall flag set when EITHER detector flags it.
    union_index = result_a.flags.index.union(result_b.flags.index)
    union_dates = set_a | set_b
    union_flags = pd.Series(
        [idx in union_dates for idx in union_index],
        index=union_index,
        name="anomaly_flag",
    )

    proxy_precision, proxy_recall = proxy_precision_recall(union_flags, returns, window=window)
    regime = regime_alignment(union_flags)

    top_dates = _top_anomaly_dates(result_a, result_b)

    return AgreementResult(
        jaccard=jaccard,
        overlap_count=len(set_a & set_b),
        n_flags_a=len(set_a),
        n_flags_b=len(set_b),
        proxy_precision=proxy_precision,
        proxy_recall=proxy_recall,
        regime_alignment=regime,
        top_anomaly_dates=top_dates,
        meta={
            "detector_a": result_a.detector,
            "detector_b": result_b.detector,
            "window": int(window),
            "n_union_flags": len(union_dates),
            "proxy_z_threshold": PROXY_Z_THRESHOLD,
        },
    )
