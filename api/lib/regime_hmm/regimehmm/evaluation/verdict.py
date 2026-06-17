"""Pure-function verdict on the regime-timing overlay (honest-null discipline).

The headline verdict is a PURE FUNCTION of the out-of-sample inference outputs
``(jk_pvalue, deflated_sharpe, sharpe_diff)``. It is STRUCTURALLY UNABLE to claim
the overlay beats buy-and-hold whenever the Memmel-Jobson-Korkie test is
insignificant or the Deflated Sharpe (computed with the correct effective
``n_trials``) is non-positive. This is what keeps the README honest: the verdict is
derived from the evidence, never narrated.

The effective number of trials counts the FULL explored grid:
``n_effective_trials = |n_states grid| x |feature-set variants| x |cost grid|``.

Importing this module has no side effects.
"""

from __future__ import annotations

import math
from enum import StrEnum

from regimehmm._exceptions import ValidationError


class TimingVerdict(StrEnum):
    """Possible headline verdicts for the regime-timing overlay.

    Stable string identifiers, safe to serialize across the API boundary and
    render in the frontend.
    """

    #: The Memmel-JK test is insignificant OR the Deflated Sharpe is non-positive
    #: — the overlay shows no reliable edge over buy-and-hold (the honest,
    #: literature-consistent outcome and the project's headline).
    NO_TIMING_EDGE = "no_timing_edge"

    #: A positive Sharpe gap that is statistically detectable (significant Memmel-JK)
    #: but does NOT survive Deflated-Sharpe multiple-testing deflation — a fragile,
    #: likely-overfit edge.
    MARGINAL = "marginal"

    #: A positive Sharpe gap that is BOTH significant under Memmel-JK AND survives
    #: the Deflated Sharpe with the full effective ``n_trials``.
    TIMING_EDGE = "timing_edge"


def effective_n_trials(
    n_states_grid: int,
    n_feature_variants: int,
    n_cost_levels: int,
) -> int:
    """Compute the effective number of trials for Deflated-Sharpe deflation.

    The honest multiplicity count is the size of the FULL explored configuration
    grid:

    ``n_effective_trials = n_states_grid * n_feature_variants * n_cost_levels``.

    A guard in :func:`derive_timing_verdict` asserts this is at least 1 and is not
    silently collapsed to 1 (which would defeat the deflation).

    Parameters
    ----------
    n_states_grid:
        The number of ``n_states`` values explored (e.g. ``{2, 3, 4}`` -> 3).
    n_feature_variants:
        The number of feature-set variants explored (e.g.
        ``{returns, returns_vol, returns_vol_macro}`` -> 3).
    n_cost_levels:
        The number of cost levels in the sensitivity grid.

    Returns
    -------
    int
        The product ``n_states_grid * n_feature_variants * n_cost_levels``.

    Raises
    ------
    ValidationError
        If any factor is less than 1.
    """
    factors = {
        "n_states_grid": n_states_grid,
        "n_feature_variants": n_feature_variants,
        "n_cost_levels": n_cost_levels,
    }
    for label, value in factors.items():
        if value < 1:
            raise ValidationError(f"effective_n_trials requires {label} >= 1, got {value}.")

    return n_states_grid * n_feature_variants * n_cost_levels


def derive_timing_verdict(
    jk_pvalue: float,
    deflated_sharpe: float,
    sharpe_diff: float,
    *,
    alpha: float = 0.05,
    dsr_threshold: float = 0.95,
) -> TimingVerdict:
    r"""Derive the regime-timing verdict from OOS inference (pure function).

    Decision rule (truth-table unit-tested):

    1. If the Memmel-JK test is insignificant (``jk_pvalue >= alpha``) OR the
       Sharpe gap is not positive (``sharpe_diff <= 0``) OR the Deflated Sharpe is
       non-positive (a degenerate-deflation guard), return
       :attr:`TimingVerdict.NO_TIMING_EDGE`.
    2. Else, if the gap is significant but the Deflated Sharpe fails its threshold
       (``deflated_sharpe < dsr_threshold``), return
       :attr:`TimingVerdict.MARGINAL`.
    3. Else (significant gap AND Deflated Sharpe clears the threshold), return
       :attr:`TimingVerdict.TIMING_EDGE`.

    HONESTY REQUIREMENT: this function MUST NOT return
    :attr:`TimingVerdict.TIMING_EDGE` whenever Memmel-JK is insignificant or the
    Deflated Sharpe is ``<= 0``; the verdict is a deterministic consequence of the
    OOS evidence, never a narrative choice. On the synthetic ``regime_switch``
    fixture (overlay does not beat buy-and-hold after costs) it returns
    ``NO_TIMING_EDGE`` (regression-pinned).

    Parameters
    ----------
    jk_pvalue:
        The Memmel-Jobson-Korkie two-sided p-value for the overlay-vs-buy-and-hold
        Sharpe gap.
    deflated_sharpe:
        The Deflated Sharpe ratio of the overlay, deflated by the FULL effective
        ``n_trials`` (:func:`effective_n_trials`).
    sharpe_diff:
        The point Sharpe gap ``overlay - buy_and_hold``.
    alpha:
        Significance level for the Memmel-JK test (default ``0.05``).
    dsr_threshold:
        Minimum Deflated Sharpe required to support a ``timing_edge`` claim
        (default ``0.95``).

    Returns
    -------
    TimingVerdict
        The derived headline verdict.

    Raises
    ------
    ValidationError
        If ``jk_pvalue`` is outside ``[0, 1]``.
    """
    if math.isnan(jk_pvalue) or not 0.0 <= jk_pvalue <= 1.0:
        raise ValidationError(
            f"derive_timing_verdict requires jk_pvalue in [0, 1], got {jk_pvalue}."
        )

    # Honest-null guards. ANY of these collapses the verdict to NO_TIMING_EDGE,
    # which is what makes a ``timing_edge`` claim STRUCTURALLY impossible when the
    # Memmel-JK test is insignificant or the Deflated Sharpe is non-positive:
    #   * the Sharpe gap is not strictly positive (a non-edge by definition);
    #   * the Memmel-JK test is insignificant at ``alpha`` (gap indistinguishable
    #     from zero);
    #   * the Deflated Sharpe is non-positive (degenerate-deflation guard) or NaN.
    # NaN ``deflated_sharpe`` / ``sharpe_diff`` are treated as failing their
    # respective positivity checks (``NaN <= 0`` is False, so test explicitly).
    gap_is_edge = sharpe_diff > 0.0  # NaN -> False
    jk_significant = jk_pvalue < alpha
    dsr_positive = deflated_sharpe > 0.0  # NaN -> False

    if not gap_is_edge or not jk_significant or not dsr_positive:
        return TimingVerdict.NO_TIMING_EDGE

    # A significant, positive gap that does NOT clear the Deflated-Sharpe
    # threshold is a fragile, likely-overfit edge.
    if deflated_sharpe < dsr_threshold:
        return TimingVerdict.MARGINAL

    # Significant gap AND Deflated Sharpe clears its threshold.
    return TimingVerdict.TIMING_EDGE
