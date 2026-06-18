"""Diebold-Mariano (1995) test on the per-period IC / return differential.

The DM test compares two strategies' out-of-sample per-period skill. With a GNN's
per-period rank-IC series ``ic_gnn_t`` and the best baseline's ``ic_base_t``, the
differential ``d_t = ic_gnn_t - ic_base_t`` has mean ``d_bar``; the DM statistic is
``d_bar / HAC_SE(d)``, asymptotically standard normal under the null of equal
predictive accuracy. A POSITIVE statistic with a small p-value means the GNN beats
the baseline on rank-IC; a p-value ``>= alpha`` means the difference is
INSIGNIFICANT (the honest NULL on a leakage-free panel). The same test runs on the
long-short return differential.

NOTE the sign convention difference from the forecast-error DM: here the
differential is a SKILL series (higher is better), so the GNN beats the baseline
when the statistic is POSITIVE (a *higher* mean IC/return), not negative.

The HAC standard error of the differential mean uses a Newey-West Bartlett
long-run variance with the Andrews automatic lag — reused from
:func:`gnnstocks.evaluation.metrics.hac_standard_error`.

Importing this module has no side effects.
"""

from __future__ import annotations

import math

import numpy as np

from gnnstocks._exceptions import ValidationError
from gnnstocks._typing import FloatArray
from gnnstocks.evaluation.metrics import _coerce_pair, hac_standard_error

# quantcore-candidate: mirrors mvts-forecast:evaluation/diebold_mariano.py, with the
# loss-differential replaced by a SKILL (IC / return) differential (sign flipped).


def _norm_sf(x: float) -> float:
    """Standard-normal survival function ``1 - Phi(x)`` via the error function."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def diebold_mariano(
    skill_model: FloatArray,
    skill_baseline: FloatArray,
    *,
    lag: int | None = None,
) -> tuple[float, float]:
    r"""Diebold-Mariano test on a per-period SKILL differential (IC or LS return).

    With per-period skill series ``skill_model`` and ``skill_baseline`` (each a
    per-period rank-IC or long-short return), the differential
    ``d_t = skill_model_t - skill_baseline_t`` has mean ``d_bar``; the DM statistic
    is ``d_bar / HAC_SE(d)``, asymptotically standard normal under the null of
    equal predictive accuracy. A POSITIVE statistic with a small p-value means the
    model beats the baseline (a *higher* mean skill); a p-value ``>= alpha`` means
    the difference is insignificant (the honest NULL).

    Parameters
    ----------
    skill_model:
        The model's per-period skill series (rank-IC or long-short return).
    skill_baseline:
        The best baseline's per-period skill series (same length).
    lag:
        HAC Bartlett lag; ``None`` => Andrews automatic rule.

    Returns
    -------
    tuple[float, float]
        ``(dm_statistic, two_sided_pvalue)``. A positive statistic favours the
        model; the p-value is clipped to ``[0, 1]``.

    Raises
    ------
    ValidationError
        If inputs are empty/mismatched, or the differential HAC variance is zero
        with a non-zero mean (the statistic is undefined).
    """
    model, baseline = _coerce_pair(
        skill_model, skill_baseline, scores_name="skill_model", returns_name="skill_baseline"
    )
    # SKILL differential (higher is better): d_t = skill_model_t - skill_baseline_t.
    # A POSITIVE mean means the model has the higher mean rank-IC / long-short
    # return, the opposite sign convention to the forecast-error DM.
    diff = model - baseline
    if diff.size < 2:
        raise ValidationError("diebold_mariano needs at least two periods.")

    d_bar = float(np.mean(diff))
    # A scale-aware degeneracy check: a skill differential with no dispersion is
    # effectively constant. Comparing the peak-to-peak range to a tolerance scaled
    # by the skill magnitude is robust to the float noise that a raw
    # ``HAC_SE == 0.0`` equality check would miss (centering a constant array
    # leaves a ~1e-20 residue rather than an exact zero).
    spread = float(np.ptp(diff))
    scale = max(float(np.max(np.abs(diff))), 1.0)
    if spread <= 1e-12 * scale:
        # No detectable dispersion in the skill differential.
        if abs(d_bar) <= 1e-12 * scale:
            # The two skill series are pointwise identical (model == baseline): no
            # difference in predictive skill.
            return 0.0, 1.0
        # A non-zero CONSTANT differential: every period agrees the model is
        # uniformly better/worse, but with zero variance the asymptotic DM
        # statistic is undefined (it would diverge).
        raise ValidationError(
            "diebold_mariano: the skill differential has zero dispersion with a "
            "non-zero mean; the statistic is undefined (degenerate skill series)."
        )

    se = hac_standard_error(diff, lag=lag)
    if se == 0.0:  # pragma: no cover - defensive: spread guard catches this first
        raise ValidationError(
            "diebold_mariano: the skill-differential HAC variance is zero with a "
            "non-zero mean; the statistic is undefined."
        )

    dm_stat = d_bar / se
    pvalue = 2.0 * _norm_sf(abs(dm_stat))
    return dm_stat, min(1.0, pvalue)


def dm_favours_model(dm_statistic: float, dm_pvalue: float, *, alpha: float = 0.05) -> bool:
    """Return ``True`` iff DM is significant AND signed in the model's favour.

    The model beats the baseline only when the two-sided p-value clears the
    significance threshold (``dm_pvalue < alpha``) AND the statistic is strictly
    positive (a *higher* mean rank-IC / long-short return than the baseline). This
    is the DM-side of the pure
    :func:`gnnstocks.evaluation.verdict.derive_verdict` gate.

    Parameters
    ----------
    dm_statistic:
        The Diebold-Mariano statistic (POSITIVE favours the model on a skill
        differential).
    dm_pvalue:
        The two-sided DM p-value.
    alpha:
        Significance level (default ``0.05``).

    Returns
    -------
    bool
        ``True`` iff ``dm_pvalue < alpha and dm_statistic > 0``.
    """
    return bool(dm_pvalue < alpha and dm_statistic > 0.0)
