"""Pure-function edge-VERDICT gate (the canonical honest-edge rule).

The headline ``edge`` is a PURE FUNCTION of the inference outputs. It reads
``True`` ("the strategy/model has a robust out-of-sample edge net of costs") ONLY
when ALL of these clear (sourced from ``algo-system:evaluation/verdict.py``, the
only verdict combining all three gates the portfolio standard requires):

1. the Diebold-Mariano test is significant AND signed in the candidate's favour
   (the ``dm_significant`` flag — produced by
   :func:`quantcore.diebold_mariano.dm_favours_a`);
2. the Deflated Sharpe clears the ``1 - alpha`` CONFIDENCE level
   (``deflated_sharpe >= 1 - alpha``, default ``>= 0.95``). The DSR is a
   PROBABILITY in ``[0, 1]`` — the probability the true Sharpe exceeds the
   multiplicity-adjusted benchmark — NOT a Sharpe; a ``> 0`` test would be
   trivially satisfied by any positive Sharpe and the gate would never bind. The
   Bailey-López de Prado significance call is ``>= 1 - alpha``. (DRIFT_REPORT §9:
   the ``> 0`` form survives only in rl-trader's *stale docstring*; every repo's
   *code* correctly gates at 0.95.)
3. where a grid was selected, the Probability of Backtest Overfitting (PBO, via
   CSCV) is below ``0.5`` (``pbo < 0.5``). ``pbo=None`` (no grid) skips this gate.

If ANY applicable gate fails the verdict is ``NO_ROBUST_EDGE`` — the documented,
leakage-free honest-NULL outcome. The verdict is DERIVED from the evidence, never
narrated. Importing this module has no side effects.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from quantcore._exceptions import ValidationError
from quantcore._validation import validate_pvalue


class Verdict(StrEnum):
    """Possible headline verdicts for the honest-edge gate.

    Stable string identifiers, safe to serialize across an API boundary.
    """

    #: The candidate clears the DM-significance, the DSR ``1 - alpha`` confidence,
    #: AND (where a grid was selected) the PBO ``< 0.5`` gates.
    EDGE = "edge"

    #: The candidate is statistically indistinguishable from the baseline after the
    #: Deflated-Sharpe and (optional) PBO corrections — the honest-NULL outcome.
    NO_ROBUST_EDGE = "no_robust_edge"


@dataclass(frozen=True, slots=True)
class VerdictResult:
    """Immutable result of the pure edge-verdict derivation.

    Attributes
    ----------
    verdict:
        The derived :class:`Verdict` enum value.
    edge:
        ``True`` iff every applicable gate cleared. Mirrors
        ``verdict == Verdict.EDGE``.
    dm_significant:
        The DM gate input (significant AND signed in the candidate's favour).
    deflated_sharpe:
        The Deflated Sharpe (honest ``n_effective_trials``) — a probability in
        ``[0, 1]``.
    dsr_threshold:
        The ``1 - alpha`` confidence level the DSR had to clear.
    pbo:
        The Probability of Backtest Overfitting (CSCV) in ``[0, 1]``, or ``None``
        when no grid was selected (the PBO gate is then skipped).
    n_effective_trials:
        The honest multiplicity count used for the DSR.
    """

    verdict: Verdict
    edge: bool
    dm_significant: bool
    deflated_sharpe: float
    dsr_threshold: float
    pbo: float | None
    n_effective_trials: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        out = asdict(self)
        out["verdict"] = self.verdict.value
        return out


def edge_verdict(
    *,
    dm_significant: bool,
    deflated_sharpe: float,
    n_effective_trials: int,
    pbo: float | None = None,
    alpha: float = 0.05,
) -> VerdictResult:
    r"""Derive the headline ``edge`` verdict (pure function).

    Decision rule (truth-table unit-tested): ``edge`` is ``True`` iff ALL
    applicable gates hold:

    1. ``dm_significant`` is ``True`` (the DM test cleared ``alpha`` AND is signed
       in the candidate's favour — typically the output of
       :func:`quantcore.diebold_mariano.dm_favours_a`);
    2. ``deflated_sharpe >= 1 - alpha`` (the ``1 - alpha`` confidence level — the
       DSR is a probability in ``[0, 1]``, so the gate is ``>= 1 - alpha``, e.g.
       ``>= 0.95``, NOT ``> 0``);
    3. if ``pbo`` is not ``None`` (a grid was selected): ``pbo < 0.5``.

    If any applicable gate fails the verdict is :attr:`Verdict.NO_ROBUST_EDGE`.
    This function MUST NOT return :attr:`Verdict.EDGE` while the DM test is
    insignificant, the DSR is below ``1 - alpha``, or (where supplied) the PBO is
    at or above ``0.5`` — regardless of any point estimate.

    Parameters
    ----------
    dm_significant:
        Whether the DM test is significant AND signed in the candidate's favour.
    deflated_sharpe:
        The Deflated Sharpe (honest ``n_effective_trials``) — a probability in
        ``[0, 1]``.
    n_effective_trials:
        The honest multiplicity count used for the DSR (``>= 1``).
    pbo:
        The Probability of Backtest Overfitting (CSCV) in ``[0, 1]``, or ``None``
        when no grid was selected (skips the PBO gate).
    alpha:
        Significance level; the DSR confidence gate is ``>= 1 - alpha``
        (default ``0.05`` => DSR gate at ``0.95``).

    Returns
    -------
    VerdictResult
        The derived verdict and the evidence that produced it.

    Raises
    ------
    ValidationError
        If ``deflated_sharpe`` is non-finite, ``alpha`` is not in ``(0, 1)``,
        ``n_effective_trials < 1``, or ``pbo`` (when supplied) is outside
        ``[0, 1]``.
    """
    if not math.isfinite(deflated_sharpe):
        raise ValidationError(f"deflated_sharpe must be finite, got {deflated_sharpe}.")
    if not 0.0 < alpha < 1.0:
        raise ValidationError(f"alpha must be in (0, 1), got {alpha}.")
    if n_effective_trials < 1:
        raise ValidationError(f"n_effective_trials must be >= 1, got {n_effective_trials}.")
    pbo_checked: float | None = None if pbo is None else validate_pvalue(pbo, name="pbo")

    dsr_threshold = 1.0 - alpha
    # Gate 1: DM significant AND signed in the candidate's favour (caller-supplied).
    dm_ok = bool(dm_significant)
    # Gate 2: the DSR is a probability; it must clear the 1 - alpha confidence
    # level (e.g. 0.95), not merely exceed 0 (which any positive Sharpe satisfies).
    dsr_ok = deflated_sharpe >= dsr_threshold
    # Gate 3: where a grid was selected, the selected config must be more likely
    # than not the genuine best OOS (pbo < 0.5). No grid => gate skipped.
    pbo_ok = True if pbo_checked is None else pbo_checked < 0.5

    edge = dm_ok and dsr_ok and pbo_ok
    verdict = Verdict.EDGE if edge else Verdict.NO_ROBUST_EDGE
    return VerdictResult(
        verdict=verdict,
        edge=edge,
        dm_significant=dm_ok,
        deflated_sharpe=deflated_sharpe,
        dsr_threshold=dsr_threshold,
        pbo=pbo_checked,
        n_effective_trials=n_effective_trials,
    )
