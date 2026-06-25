"""Diebold-Mariano (1995) test of equal predictive accuracy, with HAC variance.

The DM test compares two series' out-of-sample per-period performance. With the
differential ``d_t = a_t - b_t`` (mean ``d_bar``), the DM statistic is
``d_bar / HAC_SE(d)`` — asymptotically standard normal under the null of equal
performance, with the HAC (Newey-West / Bartlett / Andrews-lag) denominator so it
is valid under autocorrelated differentials.

SIGN IS CALLER-CONTROLLED. The statistic carries the sign of ``mean(a - b)``;
whether "positive favours ``a``" depends entirely on whether the inputs are a
PERFORMANCE series (higher better → positive favours ``a``) or a LOSS series
(lower better → negative favours ``a``). quantcore does NOT bake a sign in. The
:func:`dm_favours_a` helper takes an explicit ``greater_is_better`` flag.

REFERENCE-DISTRIBUTION SPLIT (flagged in DRIFT_REPORT §5). The portfolio carried
two conventions: the asymptotic **Normal** reference (4 repos: algo-system /
rl-trader / mvts-forecast / gnn-stocks) and a **Student-t(T-1)** reference WITH
the Harvey-Leybourne-Newbold (1997) small-sample correction (volforecast). They
give DIFFERENT finite-sample p-values. quantcore defaults to **Normal, no HLN**
(the majority / verdict-gate convention) and exposes ``reference="t"`` +
``hln_correction=True`` so volforecast's exact behaviour is reproducible without
silently mixing the two. Importing this module has no side effects.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from scipy import stats

from quantcore._exceptions import ValidationError
from quantcore._typing import Series1D
from quantcore._validation import coerce_pair
from quantcore.hac import newey_west_se


def _norm_sf(x: float) -> float:
    """Standard-normal survival function ``1 - Phi(x)`` via the error function."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def diebold_mariano(
    series_a: Series1D,
    series_b: Series1D,
    *,
    lag: int | None = None,
    reference: Literal["normal", "t"] = "normal",
    hln_correction: bool = False,
) -> tuple[float, float]:
    r"""Diebold-Mariano test on the differential ``d_t = a_t - b_t`` (HAC variance).

    The statistic is ``d_bar / HAC_SE(d)`` with the Newey-West HAC standard error
    (:func:`quantcore.hac.newey_west_se`) so it is valid under autocorrelated
    differentials. The returned statistic carries the sign of ``mean(a - b)``
    (see the module docstring: the favourable sign is caller-determined by whether
    the inputs are performance vs loss series).

    Degeneracy guard: a differential with no dispersion is effectively constant;
    if its mean is also ~0 the two series are pointwise identical and the test
    returns ``(0.0, 1.0)``; a non-zero constant differential has an undefined
    (diverging) asymptotic statistic and raises.

    Parameters
    ----------
    series_a, series_b:
        The two per-period series (e.g. model and baseline). Same length, finite.
    lag:
        HAC Bartlett lag; ``None`` selects the Andrews automatic rule.
    reference:
        ``"normal"`` (default) for the asymptotic standard-normal reference, or
        ``"t"`` for a Student-t with ``T - 1`` degrees of freedom (the
        volforecast convention).
    hln_correction:
        If ``True``, apply the Harvey-Leybourne-Newbold (1997) one-step-ahead
        small-sample scaling ``sqrt((T - 1) / T)`` to the statistic (default
        ``False``). The volforecast behaviour is ``reference="t",
        hln_correction=True``.

    Returns
    -------
    tuple[float, float]
        ``(dm_statistic, two_sided_pvalue)``; the p-value is clipped to ``[0, 1]``.

    Raises
    ------
    ValidationError
        If inputs are empty / mismatched / non-finite, fewer than two
        observations, or the differential is a non-zero constant (undefined
        statistic).
    """
    a, b = coerce_pair(series_a, series_b, a_name="series_a", b_name="series_b")
    diff = a - b
    if diff.size < 2:
        raise ValidationError("diebold_mariano needs at least two observations.")

    d_bar = float(np.mean(diff))
    # Scale-aware degeneracy check: a constant differential has zero dispersion.
    # The peak-to-peak vs magnitude-scaled tolerance is robust to the float noise
    # a raw HAC_SE == 0 check would miss (centering a constant leaves ~1e-20).
    spread = float(np.ptp(diff))
    scale = max(float(np.max(np.abs(diff))), 1.0)
    if spread <= 1e-12 * scale:
        if abs(d_bar) <= 1e-12 * scale:
            return 0.0, 1.0
        raise ValidationError(
            "diebold_mariano: the differential has zero dispersion with a non-zero "
            "mean; the statistic is undefined (degenerate series)."
        )

    se = newey_west_se(diff, lag=lag)
    if se == 0.0:  # pragma: no cover - defensive: spread guard catches this first
        raise ValidationError(
            "diebold_mariano: the differential HAC variance is zero with a non-zero "
            "mean; the statistic is undefined."
        )

    t = diff.size
    dm_stat = d_bar / se
    if hln_correction:
        # Harvey-Leybourne-Newbold (1997) one-step-ahead small-sample scaling.
        dm_stat *= math.sqrt((t - 1) / t)

    if reference == "normal":
        pvalue = 2.0 * _norm_sf(abs(dm_stat))
    elif reference == "t":
        df = t - 1
        pvalue = 2.0 * float(stats.t.sf(abs(dm_stat), df))
    else:  # pragma: no cover - guarded by the Literal type at call sites
        raise ValidationError(f"reference must be 'normal' or 't', got {reference!r}.")

    return dm_stat, min(1.0, pvalue)


def dm_favours_a(
    dm_statistic: float,
    dm_pvalue: float,
    *,
    greater_is_better: bool = True,
    alpha: float = 0.05,
) -> bool:
    """Return ``True`` iff the DM test is significant AND signed in ``a``'s favour.

    ``a`` (the first series passed to :func:`diebold_mariano`) beats ``b`` only
    when the two-sided p-value clears ``alpha`` AND the statistic is signed in
    ``a``'s favour. The favourable sign depends on the input space:

    - performance series (``greater_is_better=True``, the default): ``a`` wins
      when ``dm_statistic > 0`` (a higher mean ``a - b``);
    - loss series (``greater_is_better=False``): ``a`` wins when
      ``dm_statistic < 0`` (a lower mean loss).

    Parameters
    ----------
    dm_statistic:
        The DM statistic of the differential ``a - b``.
    dm_pvalue:
        The two-sided DM p-value.
    greater_is_better:
        Whether the inputs are a performance series (``True``) or a loss series
        (``False``).
    alpha:
        Significance level (default ``0.05``).

    Returns
    -------
    bool
        ``True`` iff ``dm_pvalue < alpha`` and the statistic favours ``a``.
    """
    if dm_pvalue >= alpha:
        return False
    return bool(dm_statistic > 0.0) if greater_is_better else bool(dm_statistic < 0.0)
