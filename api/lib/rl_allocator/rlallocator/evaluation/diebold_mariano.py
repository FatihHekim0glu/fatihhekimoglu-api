"""Diebold-Mariano (1995) test on the RL-vs-best-baseline per-bar net-return differential.

The DM test compares two strategies' out-of-sample per-bar performance. With the
RL allocator's per-bar net-return series ``r_rl_t`` and the BEST baseline's
``r_base_t``, the differential ``d_t = r_rl_t - r_base_t`` has mean ``d_bar``; the
DM statistic is ``d_bar / HAC_SE(d)``, asymptotically standard normal under the null
of equal performance. A POSITIVE statistic with a small p-value means the RL
allocator beats the best baseline on net return; a p-value ``>= alpha`` means the
difference is INSIGNIFICANT (the honest NULL — the expected outcome net of costs).

NOTE the sign convention: the differential is a PERFORMANCE series (higher is
better), so the RL allocator beats the baseline when the statistic is POSITIVE (a
*higher* mean net return), not negative.

The HAC standard error of the differential mean uses a Newey-West Bartlett
long-run variance with the Andrews automatic lag — reused from
:func:`rlallocator.evaluation.metrics.hac_standard_error`.

Importing this module has no side effects.
"""

from __future__ import annotations

import math

import numpy as np

from rlallocator._exceptions import ValidationError
from rlallocator._typing import FloatArray
from rlallocator.evaluation.metrics import _coerce_pair, hac_standard_error

# quantcore-candidate: mirrors algo-system:evaluation/diebold_mariano.py +
# rl-trader:evaluation/diebold_mariano.py with the per-bar net-RETURN differential
# (same sign: higher is better, so a positive statistic favours the RL allocator).


def _norm_sf(x: float) -> float:
    """Standard-normal survival function ``1 - Phi(x)`` via the error function."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def diebold_mariano(
    net_returns_rl: FloatArray,
    net_returns_baseline: FloatArray,
    *,
    lag: int | None = None,
) -> tuple[float, float]:
    r"""Diebold-Mariano test on the RL-vs-baseline per-bar net-return differential.

    With per-bar net-return series ``net_returns_rl`` and ``net_returns_baseline``,
    the differential ``d_t = rl_t - baseline_t`` has mean ``d_bar``; the DM statistic
    is ``d_bar / HAC_SE(d)``, asymptotically standard normal under the null of equal
    performance. A POSITIVE statistic with a small p-value means the RL allocator
    beats the baseline (a *higher* mean net return); a p-value ``>= alpha`` means the
    difference is insignificant (the honest NULL).

    Parameters
    ----------
    net_returns_rl:
        The RL allocator's per-bar net-return series.
    net_returns_baseline:
        The (best) baseline's per-bar net-return series (same length).
    lag:
        HAC Bartlett lag; ``None`` => Andrews automatic rule.

    Returns
    -------
    tuple[float, float]
        ``(dm_statistic, two_sided_pvalue)``. A positive statistic favours the RL
        allocator; the p-value is clipped to ``[0, 1]``.

    Raises
    ------
    ValidationError
        If inputs are empty/mismatched, or the differential HAC variance is zero
        with a non-zero mean (the statistic is undefined).
    """
    rl, baseline = _coerce_pair(
        net_returns_rl,
        net_returns_baseline,
        a_name="net_returns_rl",
        b_name="net_returns_baseline",
    )
    # Net-return differential (higher is better): d_t = rl_t - baseline_t. A POSITIVE
    # mean means the RL allocator has the higher mean net return.
    diff = rl - baseline
    if diff.size < 2:
        raise ValidationError("diebold_mariano needs at least two bars.")

    d_bar = float(np.mean(diff))
    # A scale-aware degeneracy check: a differential with no dispersion is
    # effectively constant. Comparing the peak-to-peak range to a tolerance scaled
    # by the magnitude is robust to the float noise a raw ``HAC_SE == 0.0`` check
    # would miss (centering a constant array leaves a ~1e-20 residue, not an exact
    # zero).
    spread = float(np.ptp(diff))
    scale = max(float(np.max(np.abs(diff))), 1.0)
    if spread <= 1e-12 * scale:
        if abs(d_bar) <= 1e-12 * scale:
            # The two series are pointwise identical (rl == baseline): no difference.
            return 0.0, 1.0
        # A non-zero CONSTANT differential: every bar agrees one is uniformly
        # better, but with zero variance the asymptotic DM statistic is undefined.
        raise ValidationError(
            "diebold_mariano: the net-return differential has zero dispersion with a "
            "non-zero mean; the statistic is undefined (degenerate series)."
        )

    se = hac_standard_error(diff, lag=lag)
    if se == 0.0:  # pragma: no cover - defensive: spread guard catches this first
        raise ValidationError(
            "diebold_mariano: the net-return-differential HAC variance is zero with a "
            "non-zero mean; the statistic is undefined."
        )

    dm_stat = d_bar / se
    pvalue = 2.0 * _norm_sf(abs(dm_stat))
    return dm_stat, min(1.0, pvalue)


def dm_favours_model(dm_statistic: float, dm_pvalue: float, *, alpha: float = 0.05) -> bool:
    """Return ``True`` iff DM is significant AND signed in the RL allocator's favour.

    The RL allocator beats the baseline only when the two-sided p-value clears the
    significance threshold (``dm_pvalue < alpha``) AND the statistic is strictly
    positive (a *higher* mean net return than the baseline). This is the DM-side of
    the pure :func:`rlallocator.evaluation.verdict.derive_verdict` gate.

    Parameters
    ----------
    dm_statistic:
        The Diebold-Mariano statistic (POSITIVE favours the RL allocator).
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
