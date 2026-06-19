"""Diebold-Mariano (1995) test on the system-vs-buy-hold per-bar net-return differential.

The DM test compares two strategies' out-of-sample per-bar performance. With the
system's per-bar net-return series ``r_sys_t`` and buy-and-hold's ``r_bh_t``, the
differential ``d_t = r_sys_t - r_bh_t`` has mean ``d_bar``; the DM statistic is
``d_bar / HAC_SE(d)``, asymptotically standard normal under the null of equal
performance. A POSITIVE statistic with a small p-value means the system beats
buy-and-hold on net return; a p-value ``>= alpha`` means the difference is
INSIGNIFICANT (the honest NULL — the expected outcome net of costs).

NOTE the sign convention: the differential is a PERFORMANCE series (higher is
better), so the system beats buy-and-hold when the statistic is POSITIVE (a
*higher* mean net return), not negative.

The HAC standard error of the differential mean uses a Newey-West Bartlett
long-run variance with the Andrews automatic lag, reused from
:func:`algosystem.evaluation.hac.newey_west_se`.

Importing this module has no side effects.
"""

from __future__ import annotations

import math

import numpy as np

from algosystem._exceptions import ValidationError
from algosystem._typing import FloatArray
from algosystem.evaluation.hac import newey_west_se

# quantcore-candidate: mirrors rl-trader:evaluation/diebold_mariano.py with the
# RL net-return differential replaced by the system-vs-buy-hold per-bar net-RETURN
# differential (same sign: higher is better, so a positive statistic favours the
# system).


def _norm_sf(x: float) -> float:
    """Standard-normal survival function ``1 - Phi(x)`` via the error function."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def _coerce_pair(
    series_a: FloatArray,
    series_b: FloatArray,
    *,
    a_name: str,
    b_name: str,
) -> tuple[FloatArray, FloatArray]:
    """Coerce a pair of per-bar series to aligned, finite, equal-length float64 vectors."""
    a = np.asarray(series_a, dtype=np.float64).ravel()
    b = np.asarray(series_b, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0:
        raise ValidationError(f"{a_name} and {b_name} must be non-empty.")
    if a.size != b.size:
        raise ValidationError(
            f"{a_name} (len {a.size}) and {b_name} (len {b.size}) must have the same length."
        )
    if not np.isfinite(a).all():
        raise ValidationError(f"{a_name} contains non-finite values.")
    if not np.isfinite(b).all():
        raise ValidationError(f"{b_name} contains non-finite values.")
    return a, b


def diebold_mariano(
    net_returns_system: FloatArray,
    net_returns_baseline: FloatArray,
    *,
    lag: int | None = None,
) -> tuple[float, float]:
    r"""Diebold-Mariano test on the system-vs-baseline per-bar net-return differential.

    With per-bar net-return series ``net_returns_system`` and
    ``net_returns_baseline``, the differential ``d_t = system_t - baseline_t`` has
    mean ``d_bar``; the DM statistic is ``d_bar / HAC_SE(d)``, asymptotically
    standard normal under the null of equal performance. A POSITIVE statistic with
    a small p-value means the system beats the baseline (a *higher* mean net
    return); a p-value ``>= alpha`` means the difference is insignificant (the
    honest NULL).

    Parameters
    ----------
    net_returns_system:
        The system's per-bar net-return series.
    net_returns_baseline:
        The baseline's (buy-and-hold's) per-bar net-return series (same length).
    lag:
        HAC Bartlett lag; ``None`` => Andrews automatic rule.

    Returns
    -------
    tuple[float, float]
        ``(dm_statistic, two_sided_pvalue)``. A positive statistic favours the
        system; the p-value is clipped to ``[0, 1]``.

    Raises
    ------
    ValidationError
        If inputs are empty/mismatched, fewer than two bars, or the differential
        HAC variance is zero with a non-zero mean (the statistic is undefined).
    """
    system, baseline = _coerce_pair(
        net_returns_system,
        net_returns_baseline,
        a_name="net_returns_system",
        b_name="net_returns_baseline",
    )
    # Net-return differential (higher is better): d_t = system_t - baseline_t. A
    # POSITIVE mean means the system has the higher mean net return.
    diff = system - baseline
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
            # The two series are pointwise identical (system == baseline): no diff.
            return 0.0, 1.0
        # A non-zero CONSTANT differential: every bar agrees one is uniformly
        # better, but with zero variance the asymptotic DM statistic is undefined.
        raise ValidationError(
            "diebold_mariano: the net-return differential has zero dispersion with a "
            "non-zero mean; the statistic is undefined (degenerate series)."
        )

    se = newey_west_se(diff, lag=lag)
    if se == 0.0:  # pragma: no cover - defensive: spread guard catches this first
        raise ValidationError(
            "diebold_mariano: the net-return-differential HAC variance is zero with a "
            "non-zero mean; the statistic is undefined."
        )

    dm_stat = d_bar / se
    pvalue = 2.0 * _norm_sf(abs(dm_stat))
    return dm_stat, min(1.0, pvalue)


def dm_favours_system(dm_statistic: float, dm_pvalue: float, *, alpha: float = 0.05) -> bool:
    """Return ``True`` iff DM is significant AND signed in the system's favour.

    The system beats the baseline only when the two-sided p-value clears the
    significance threshold (``dm_pvalue < alpha``) AND the statistic is strictly
    positive (a *higher* mean net return than the baseline). This is the DM-side of
    the pure :func:`algosystem.evaluation.verdict.system_has_edge` gate.

    Parameters
    ----------
    dm_statistic:
        The Diebold-Mariano statistic (POSITIVE favours the system).
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
