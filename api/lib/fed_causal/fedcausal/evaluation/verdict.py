"""Pure-function verdict derivation — the honest-NULL gate.

The headline ``fed_effect_is_tradable`` is a PURE FUNCTION of four independent
lines of evidence:

1. **placebo-percentile significance** — the observed CAR sits far in the tail of
   the placebo-date null (the PRIMARY significance source, never a raw t-stat);
2. **HAC-robust CAR** — the mean CAR is significant under a Newey-West HAC
   standard error;
3. **survives multiple testing** — at least one spec survives the
   Benjamini-Hochberg / Romano-Wolf correction across the FULL grid; and
4. **a net-of-cost tradable DiD spread** — the rate-sensitivity heterogeneity
   implies a long/short spread that is still positive AND significant AFTER
   transaction costs.

The verdict reads ``True`` ONLY when ALL FOUR clear. By construction of the honest
deliverable it reads ``False``: the FOMC move is cross-sectional heterogeneity
(rate-sensitive names move more, mechanically), statistically fragile and not
tradable net of costs. The truth table is unit-tested — the verdict is DERIVED,
never narrated.

Importing this module has no side effects.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from fedcausal._constants import DEFAULT_ALPHA
from fedcausal._exceptions import ValidationError
from fedcausal._validation import validate_alpha


@dataclass(frozen=True, slots=True)
class VerdictInputs:
    """The four pieces of evidence the verdict consumes (all already computed).

    Attributes
    ----------
    placebo_pvalue:
        The placebo-date tail probability of the observed CAR (PRIMARY
        significance). Significant when ``< alpha``.
    hac_pvalue:
        The two-sided HAC / Newey-West p-value of the mean CAR. Significant when
        ``< alpha``.
    multiple_testing_survives:
        Whether ANY spec survives the multiple-testing correction over the full
        grid.
    did_net_spread:
        The DiD-implied long/short spread NET of transaction costs.
    did_spread_pvalue:
        The significance of the net-of-cost spread. Tradable when the net spread
        is positive AND ``did_spread_pvalue < alpha``.
    """

    placebo_pvalue: float
    hac_pvalue: float
    multiple_testing_survives: bool
    did_net_spread: float
    did_spread_pvalue: float

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the inputs."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FedVerdict:
    """The derived verdict and the per-condition evidence that produced it.

    Attributes
    ----------
    fed_effect_is_tradable:
        The headline boolean — ``True`` only if ALL four conditions clear.
    placebo_significant:
        Condition 1: placebo-percentile significance.
    hac_robust:
        Condition 2: HAC-robust mean CAR.
    survives_multiple_testing:
        Condition 3: at least one spec survives the full-grid correction.
    tradable_did_spread:
        Condition 4: a net-of-cost positive, significant DiD spread.
    rationale:
        A short, honest sentence explaining the verdict.
    """

    fed_effect_is_tradable: bool
    placebo_significant: bool
    hac_robust: bool
    survives_multiple_testing: bool
    tradable_did_spread: bool
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the verdict."""
        return asdict(self)


def fed_effect_is_tradable(
    inputs: VerdictInputs,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> bool:
    """Return the PURE headline boolean: is the Fed effect a tradable alpha?

    PURE FUNCTION. Returns ``True`` if and only if ALL FOUR conditions hold:

    1. ``placebo_pvalue < alpha`` (placebo-percentile significance), AND
    2. ``hac_pvalue < alpha`` (HAC-robust mean CAR), AND
    3. ``multiple_testing_survives`` is ``True`` (full-grid correction), AND
    4. ``did_net_spread > 0`` AND ``did_spread_pvalue < alpha`` (a net-of-cost
       tradable DiD spread).

    Any single condition failing returns ``False``. This is what keeps the README
    honest: a raw t-stat alone can never flip the verdict.

    Parameters
    ----------
    inputs:
        The four pieces of evidence.
    alpha:
        Significance level applied to every p-value gate.

    Returns
    -------
    bool
        Whether the Fed effect is a placebo-robust, HAC-robust, multiplicity-
        surviving, net-of-cost tradable alpha.

    Raises
    ------
    ValidationError
        If any p-value is outside ``[0, 1]`` or ``alpha`` is out of range.
    """
    return derive_verdict(inputs, alpha=alpha).fed_effect_is_tradable


def _evaluate_conditions(
    inputs: VerdictInputs,
    *,
    alpha: float,
) -> tuple[bool, bool, bool, bool]:
    """Evaluate the four independent verdict conditions (PURE).

    Returns ``(placebo_significant, hac_robust, survives_multiple_testing,
    tradable_did_spread)``. Both p-value gates are strict ``< alpha``; the DiD
    spread is tradable only when it is BOTH positive in the mean AND significant.
    """
    placebo_significant = _validate_pvalue(inputs.placebo_pvalue, name="placebo_pvalue") < alpha
    hac_robust = _validate_pvalue(inputs.hac_pvalue, name="hac_pvalue") < alpha
    survives_multiple_testing = bool(inputs.multiple_testing_survives)
    spread_pvalue = _validate_pvalue(inputs.did_spread_pvalue, name="did_spread_pvalue")
    net_spread = float(inputs.did_net_spread)
    if not math.isfinite(net_spread):
        raise ValidationError(f"did_net_spread must be finite, got {inputs.did_net_spread}.")
    tradable_did_spread = net_spread > 0.0 and spread_pvalue < alpha
    return placebo_significant, hac_robust, survives_multiple_testing, tradable_did_spread


def derive_verdict(
    inputs: VerdictInputs,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> FedVerdict:
    """Derive the full verdict (boolean + per-condition evidence + rationale).

    The PURE core: it evaluates each of the four conditions individually, ANDs
    them into the headline boolean (which :func:`fed_effect_is_tradable` simply
    reads off), and attaches an honest one-line rationale so the API/frontend can
    render the "Tradable Fed effect: NO" badge with its supporting evidence.

    Parameters
    ----------
    inputs:
        The four pieces of evidence.
    alpha:
        Significance level applied to every p-value gate.

    Returns
    -------
    FedVerdict
        The derived verdict with per-condition flags and a rationale.

    Raises
    ------
    ValidationError
        If any p-value is outside ``[0, 1]`` or ``alpha`` is out of range.
    """
    alpha = validate_alpha(alpha)
    placebo_significant, hac_robust, survives_multiple_testing, tradable_did_spread = (
        _evaluate_conditions(inputs, alpha=alpha)
    )
    is_tradable = (
        placebo_significant and hac_robust and survives_multiple_testing and tradable_did_spread
    )
    rationale = _rationale(
        is_tradable=is_tradable,
        placebo_significant=placebo_significant,
        hac_robust=hac_robust,
        survives_multiple_testing=survives_multiple_testing,
        tradable_did_spread=tradable_did_spread,
    )
    return FedVerdict(
        fed_effect_is_tradable=is_tradable,
        placebo_significant=placebo_significant,
        hac_robust=hac_robust,
        survives_multiple_testing=survives_multiple_testing,
        tradable_did_spread=tradable_did_spread,
        rationale=rationale,
    )


#: The four condition flags in verdict order, paired with the honest label naming
#: the line of evidence that fails when the flag is ``False``.
_CONDITION_LABELS: tuple[tuple[str, str], ...] = (
    ("placebo_significant", "placebo-percentile significance"),
    ("hac_robust", "HAC-robust mean CAR"),
    ("survives_multiple_testing", "multiple-testing survival across the full grid"),
    ("tradable_did_spread", "a net-of-cost tradable DiD spread"),
)


def _rationale(
    *,
    is_tradable: bool,
    placebo_significant: bool,
    hac_robust: bool,
    survives_multiple_testing: bool,
    tradable_did_spread: bool,
) -> str:
    """Return a short, honest sentence explaining the derived verdict."""
    if is_tradable:
        return (
            "Tradable: the CAR is placebo-significant, HAC-robust, survives "
            "multiple testing, and the DiD spread clears costs."
        )
    flags = {
        "placebo_significant": placebo_significant,
        "hac_robust": hac_robust,
        "survives_multiple_testing": survives_multiple_testing,
        "tradable_did_spread": tradable_did_spread,
    }
    failed = [label for key, label in _CONDITION_LABELS if not flags[key]]
    return (
        "Not tradable: the FOMC move is cross-sectional heterogeneity, not a "
        f"placebo-robust tradable alpha — failing {', '.join(failed)}."
    )


def _validate_pvalue(value: float, *, name: str) -> float:
    """Validate a probability lies in ``[0, 1]`` (helper for the verdict gates)."""
    out = float(value)
    if not math.isfinite(out) or not 0.0 <= out <= 1.0:
        raise ValidationError(f"{name} must lie in [0, 1], got {value}.")
    return out
