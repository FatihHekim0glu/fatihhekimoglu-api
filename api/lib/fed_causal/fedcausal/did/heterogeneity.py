"""Honest cross-sectional-heterogeneity framing (SCM demoted to descriptive).

This module wraps the DiD estimate (:mod:`fedcausal.did.model`) in the brief's
honest interpretation and computes the one quantity the verdict actually needs
from the DiD: the **net-of-cost tradable spread** implied by sorting names on
rate-sensitivity and going long/short around announcements.

The synthetic-control method (SCM) is DEMOTED to a descriptive diagnostic: it can
illustrate the treated-vs-synthetic-control gap, but it is NEVER presented as a
profit claim. The deliverable is the statement that FOMC moves are cross-sectional
HETEROGENEITY (rate-sensitive names move more, mechanically), not a placebo-robust
tradable alpha.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

from fedcausal._constants import DEFAULT_COST_BPS, EPS
from fedcausal._exceptions import InsufficientDataError, ValidationError
from fedcausal._validation import validate_alpha

if TYPE_CHECKING:
    from fedcausal.did.model import DiDResult

#: Basis points per unit (a 1.0 decimal return == 10,000 bps). Used to convert the
#: round-trip ``cost_bps`` hurdle into the same decimal units as the spread.
_BPS_PER_UNIT: float = 10_000.0


@dataclass(frozen=True, slots=True)
class HeterogeneitySpread:
    """The descriptive heterogeneity spread and its net-of-cost tradability.

    Attributes
    ----------
    gross_spread:
        The gross treated-minus-control abnormal-return spread per announcement
        (the long/short rotation's average gross move).
    cost_bps:
        The round-trip transaction cost (basis points) charged against the spread.
    net_spread:
        ``gross_spread`` minus the per-rotation cost — the spread an actual
        long/short would capture net of costs.
    net_pvalue:
        Two-sided p-value of the per-announcement NET-of-cost spread against
        zero — the significance the verdict's condition 4 must gate on (NOT the
        gross clustered-DiD interaction p-value).
    is_tradable_spread:
        ``True`` only if ``net_spread`` is positive AND ``net_pvalue < alpha``
        (this is one of FOUR independent conditions the overall verdict requires;
        it is FALSE by default for the honest-null deliverable).
    """

    gross_spread: float
    cost_bps: float
    net_spread: float
    net_pvalue: float
    is_tradable_spread: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the spread."""
        return asdict(self)


def _net_spread_pvalue(net_samples: np.ndarray) -> float:
    """Two-sided one-sample t-test p-value of the per-announcement net spread.

    Returns ``1.0`` for a degenerate (zero-variance or single-observation)
    sample rather than dividing by zero, so a constant net spread is never
    spuriously flagged significant.
    """
    n = net_samples.size
    if n < 2:
        return 1.0
    mean = float(net_samples.mean())
    sd = float(net_samples.std(ddof=1))
    se = sd / np.sqrt(n)
    if se <= EPS:
        return 1.0
    t_stat = mean / se
    return float(2.0 * stats.t.sf(abs(t_stat), df=n - 1))


def heterogeneity_spread(
    did: DiDResult,
    spread_samples: np.ndarray,
    *,
    cost_bps: float = DEFAULT_COST_BPS,
    alpha: float = 0.05,
) -> HeterogeneitySpread:
    """Translate the DiD heterogeneity into a NET-OF-COST tradable spread test.

    The gross spread is the per-announcement treated-minus-control move (its
    sample mean equals the DiD interaction coefficient by construction). A
    round-trip ``cost_bps`` is converted to decimal units and subtracted from
    EVERY per-announcement spread observation; the result is flagged tradable
    ONLY if the net spread is BOTH positive (in the mean) AND statistically
    distinct from zero (two-sided t-test at ``alpha``). By construction
    (mechanical heterogeneity, fragile after costs) this reads ``False`` on the
    honest-null deliverable.

    Parameters
    ----------
    did:
        The DiD interaction result (descriptive heterogeneity magnitude).
    spread_samples:
        Per-announcement treated-minus-control spread observations (for the
        net-of-cost significance test).
    cost_bps:
        Round-trip transaction cost in basis points.
    alpha:
        Significance level for the net-spread test.

    Returns
    -------
    HeterogeneitySpread
        The gross/net spread and its (default-``False``) tradability flag.

    Raises
    ------
    ValidationError
        If ``cost_bps`` is negative or non-finite, or ``alpha`` is out of range.
    InsufficientDataError
        If ``spread_samples`` is empty after dropping non-finite values.
    """
    validate_alpha(alpha)
    cost = float(cost_bps)
    if not np.isfinite(cost) or cost < 0.0:
        raise ValidationError(f"cost_bps must be a non-negative finite float, got {cost_bps}.")

    samples = np.asarray(spread_samples, dtype=np.float64).ravel()
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        raise InsufficientDataError("heterogeneity_spread: spread_samples is empty.")

    cost_decimal = cost / _BPS_PER_UNIT
    gross_spread = float(samples.mean())
    net_samples = samples - cost_decimal
    net_spread = float(net_samples.mean())

    p_value = _net_spread_pvalue(net_samples)
    # Tradable ONLY if the net spread is positive in the mean AND distinguishable
    # from zero — both gates must clear (the honest-null reads False).
    is_tradable_spread = bool(net_spread > 0.0 and p_value < alpha)

    return HeterogeneitySpread(
        gross_spread=gross_spread,
        cost_bps=cost,
        net_spread=net_spread,
        net_pvalue=p_value,
        is_tradable_spread=is_tradable_spread,
    )


def describe_heterogeneity(
    did: DiDResult,
    spread: HeterogeneitySpread,
) -> str:
    """Return the honest, non-promotional one-line interpretation of the DiD.

    Produces text framing the result as descriptive cross-sectional heterogeneity
    (rate-sensitive names move more, mechanically), explicitly NOT a tradable
    causal alpha — the language rendered in the README and the frontend caption.

    Parameters
    ----------
    did:
        The DiD interaction result.
    spread:
        The net-of-cost spread test.

    Returns
    -------
    str
        The honest interpretation string.
    """
    direction = "more" if did.coef >= 0.0 else "less"
    tradable = (
        "and the long/short spread survives transaction costs"
        if spread.is_tradable_spread
        else "but the long/short spread does NOT survive transaction costs"
    )
    return (
        f"Rate-sensitive names move {abs(did.coef):.4f} {direction} than controls "
        f"around the surprise (DiD t={did.t_stat:.2f}, clustered by {did.cluster}); "
        f"this is descriptive cross-sectional heterogeneity, {tradable} "
        f"(gross {spread.gross_spread:.4f} -> net {spread.net_spread:.4f} after "
        f"{spread.cost_bps:.0f} bps) — not a tradable causal alpha."
    )
