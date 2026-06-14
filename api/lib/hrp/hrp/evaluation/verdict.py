"""Pure-function verdict derivation.

The headline verdict is a PURE FUNCTION of the inference outputs
``(jkm_pvalue, deflated_sharpe, ci_low, ci_high)`` from a fixed enum. It cannot
emit "HRP beats 1/N" while the bootstrap confidence interval straddles zero — the
truth table is unit-tested. This is what keeps the README honest: the verdict is
derived, not narrated.

Importing this module has no side effects.
"""

from __future__ import annotations

import math
from enum import Enum

from hrp._exceptions import ValidationError


class Verdict(str, Enum):
    """Possible headline verdicts for the HRP-vs-1/N comparison.

    The values are stable string identifiers safe to serialize across the API
    boundary and render in the frontend.
    """

    #: The Sharpe gap is positive AND statistically significant (CI strictly > 0,
    #: small JKM p-value, positive deflated Sharpe).
    HRP_BEATS_1N = "hrp_beats_1n"

    #: The Sharpe gap is negative AND statistically significant (CI strictly < 0).
    HRP_LOSES_TO_1N = "hrp_loses_to_1n"

    #: The gap is not statistically distinguishable from zero (CI straddles zero
    #: or JKM is insignificant) — the expected, literature-consistent outcome.
    NO_SIGNIFICANT_DIFFERENCE = "no_significant_difference"


def derive_verdict(
    jkm_pvalue: float,
    deflated_sharpe: float,
    ci_low: float,
    ci_high: float,
    *,
    alpha: float = 0.05,
    dsr_threshold: float = 0.95,
) -> Verdict:
    r"""Derive the headline verdict from inference outputs (pure function).

    Decision rule (truth-table unit-tested):

    1. If the bootstrap CI straddles zero (``ci_low <= 0 <= ci_high``) OR the JKM
       test is insignificant (``jkm_pvalue >= alpha``) OR the deflated Sharpe
       fails its threshold (``deflated_sharpe < dsr_threshold``), return
       :attr:`Verdict.NO_SIGNIFICANT_DIFFERENCE`. (A significant claim requires
       ALL three to agree.)
    2. Otherwise, if the CI is strictly positive (``ci_low > 0``), return
       :attr:`Verdict.HRP_BEATS_1N`.
    3. Otherwise (CI strictly negative, ``ci_high < 0``), return
       :attr:`Verdict.HRP_LOSES_TO_1N`.

    HONESTY REQUIREMENT: this function MUST NOT return :attr:`Verdict.HRP_BEATS_1N`
    whenever the CI includes zero, regardless of the point estimate. The verdict
    is a deterministic consequence of the evidence, never a narrative choice.

    Parameters
    ----------
    jkm_pvalue:
        The Jobson-Korkie-Memmel two-sided p-value for the Sharpe gap.
    deflated_sharpe:
        The deflated Sharpe ratio (FULL-grid ``n_trials``) of the HRP strategy.
    ci_low, ci_high:
        The bootstrap confidence-interval bounds on the Sharpe gap
        (``HRP - 1/N``).
    alpha:
        Significance level for the JKM test (default ``0.05``).
    dsr_threshold:
        Minimum deflated Sharpe required to support a positive claim
        (default ``0.95``).

    Returns
    -------
    Verdict
        The derived headline verdict.

    Raises
    ------
    ValidationError
        If ``ci_low > ci_high`` or ``jkm_pvalue`` is outside ``[0, 1]``.
    """
    if not math.isfinite(jkm_pvalue) or not 0.0 <= jkm_pvalue <= 1.0:
        raise ValidationError(f"jkm_pvalue must be in [0, 1], got {jkm_pvalue}.")
    if ci_low > ci_high:
        raise ValidationError(
            f"ci_low ({ci_low}) must not exceed ci_high ({ci_high})."
        )

    # A significant directional claim requires ALL three lines of evidence to
    # agree: a CI that does not straddle zero, a significant JKM test, and a
    # deflated Sharpe clearing its threshold. Any one failing -> no difference.
    ci_straddles_zero = ci_low <= 0.0 <= ci_high
    jkm_insignificant = jkm_pvalue >= alpha
    dsr_fails = deflated_sharpe < dsr_threshold

    if ci_straddles_zero or jkm_insignificant or dsr_fails:
        return Verdict.NO_SIGNIFICANT_DIFFERENCE

    # Past the gate the CI is strictly one-signed (it cannot include zero here).
    if ci_low > 0.0:
        return Verdict.HRP_BEATS_1N
    return Verdict.HRP_LOSES_TO_1N
