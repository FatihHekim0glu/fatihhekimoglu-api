"""Pure-function verdict derivation: ``system_has_edge``.

The headline verdict is a PURE FUNCTION of the inference outputs. It CANNOT read
``True`` ("the strategy beats buy-and-hold out-of-sample net of costs") unless ALL
THREE lines of evidence agree:

1. the OOS net Sharpe beats buy-and-hold with a Diebold-Mariano-significant margin
   on the per-bar net-return differential (``dm_pvalue < alpha`` AND
   ``dm_statistic > 0`` — a strictly *higher* mean net return);
2. the Deflated Sharpe (with the honest ``n_trials`` = #signals x #param configs)
   clears the ``1 - alpha`` CONFIDENCE level (``deflated_sharpe > 1 - alpha``, e.g.
   ``> 0.95``). The DSR is a PROBABILITY in ``[0, 1]`` — the probability the true
   Sharpe exceeds the multiplicity-adjusted benchmark — NOT a Sharpe; a ``> 0``
   test would be trivially satisfied by any positive Sharpe and the gate would
   never bind. The Bailey-Lopez de Prado significance call is ``> 1 - alpha``;
3. the Probability of Backtest Overfitting (PBO, via CSCV) is below ``0.5``
   (``pbo < 0.5`` — the selected configuration is more likely than not to be the
   genuine best out-of-sample, not an in-sample-overfit artifact).

If ANY of the three fails, the verdict is
:attr:`Verdict.NO_ROBUST_EDGE` — the documented, leakage-free outcome: the OOS
Sharpe is statistically indistinguishable from buy-and-hold after the
Deflated-Sharpe correction and the PBO overfitting check. The verdict is derived
from the evidence, never narrated. The truth table is unit-tested. No profit
claim is possible.

Importing this module has no side effects.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from algosystem._exceptions import ValidationError
from algosystem.evaluation.diebold_mariano import dm_favours_system


class Verdict(StrEnum):
    """Possible headline verdicts for the system-vs-buy-hold comparison.

    The values are stable string identifiers safe to serialize across the API
    boundary and render in the frontend.
    """

    #: The strategy beats buy-and-hold with a DM-significant OOS margin, a DSR
    #: clearing the ``1 - alpha`` confidence level, AND a PBO below ``0.5``.
    SYSTEM_HAS_EDGE = "system_has_edge"

    #: The strategy is not distinguishable from buy-and-hold (DM insignificant,
    #: DSR below the confidence level, or PBO >= 0.5) — the expected, honest-NULL
    #: outcome: the OOS net Sharpe shows no robust edge after costs.
    NO_ROBUST_EDGE = "no_robust_edge"


@dataclass(frozen=True, slots=True)
class VerdictResult:
    """Immutable result of the pure verdict derivation.

    Attributes
    ----------
    verdict:
        The derived :class:`Verdict` enum value.
    system_has_edge:
        ``True`` iff the OOS margin cleared the DM-significance, the
        DSR-confidence, AND the PBO gates. Mirrors
        ``verdict == Verdict.SYSTEM_HAS_EDGE``.
    dm_pvalue:
        The DM p-value of the system net return vs. buy-and-hold that drove the
        verdict.
    deflated_sharpe:
        The Deflated Sharpe (honest #signals x #param-config ``n_trials``) of the
        system's OOS net return — a probability in ``[0, 1]``.
    pbo:
        The Probability of Backtest Overfitting (CSCV), in ``[0, 1]``.
    n_effective_trials:
        The honest multiplicity count used for the DSR (#signals x #param configs).
    """

    verdict: Verdict
    system_has_edge: bool
    dm_pvalue: float
    deflated_sharpe: float
    pbo: float
    n_effective_trials: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        out = asdict(self)
        out["verdict"] = self.verdict.value
        return out


def system_has_edge(
    dm_statistic: float,
    dm_pvalue: float,
    deflated_sharpe: float,
    pbo: float,
    n_effective_trials: int,
    *,
    alpha: float = 0.05,
) -> VerdictResult:
    r"""Derive the headline ``system_has_edge`` verdict (pure function).

    Decision rule (truth-table unit-tested): ``system_has_edge`` is ``True`` iff
    ALL of the following hold for the OOS system net return vs. buy-and-hold:

    1. the Diebold-Mariano test on the per-bar net-return differential is
       significant AND signed in the system's favour (``dm_pvalue < alpha`` AND
       ``dm_statistic > 0`` — a strictly *higher* mean net return);
    2. the Deflated Sharpe (with the honest #signals x #param-config
       ``n_effective_trials``) clears the ``1 - alpha`` CONFIDENCE level
       (``deflated_sharpe > 1 - alpha``). The DSR is a probability in ``[0, 1]``,
       so the gate is ``> 1 - alpha`` (e.g. ``> 0.95``), NOT ``> 0`` — the
       multiplicity deflation must have real teeth;
    3. the Probability of Backtest Overfitting is below one half (``pbo < 0.5``).

    If ANY of the three fails, the verdict is :attr:`Verdict.NO_ROBUST_EDGE` — the
    documented honest-NULL outcome. This function MUST NOT return
    :attr:`Verdict.SYSTEM_HAS_EDGE` while the DM test is insignificant, the DSR is
    at or below the confidence level, OR the PBO is at or above ``0.5``, regardless
    of any point estimate. The verdict is a deterministic consequence of the
    evidence, never a narrative choice. No profit claim.

    Parameters
    ----------
    dm_statistic:
        The DM statistic of the system net return vs. buy-and-hold (positive
        favours the system).
    dm_pvalue:
        The two-sided DM p-value of the system net return vs. buy-and-hold.
    deflated_sharpe:
        The Deflated Sharpe (honest #signals x #param-config ``n_trials``) of the
        system's OOS net return — a probability in ``[0, 1]``.
    pbo:
        The Probability of Backtest Overfitting (CSCV), in ``[0, 1]``.
    n_effective_trials:
        The honest multiplicity count (#signals x #param configs).
    alpha:
        Significance level for the DM test and the DSR confidence gate
        (default ``0.05`` => DSR gate at ``0.95``).

    Returns
    -------
    VerdictResult
        The derived verdict and the evidence that produced it.

    Raises
    ------
    ValidationError
        If ``dm_pvalue`` or ``pbo`` is outside ``[0, 1]``, any input is non-finite,
        or ``n_effective_trials < 1``.
    """
    if not math.isfinite(dm_statistic):
        raise ValidationError(f"dm_statistic must be finite, got {dm_statistic}.")
    if not math.isfinite(dm_pvalue) or not 0.0 <= dm_pvalue <= 1.0:
        raise ValidationError(f"dm_pvalue must be in [0, 1], got {dm_pvalue}.")
    if not math.isfinite(deflated_sharpe):
        raise ValidationError(f"deflated_sharpe must be finite, got {deflated_sharpe}.")
    if not math.isfinite(pbo) or not 0.0 <= pbo <= 1.0:
        raise ValidationError(f"pbo must be in [0, 1], got {pbo}.")
    if n_effective_trials < 1:
        raise ValidationError(f"n_effective_trials must be >= 1, got {n_effective_trials}.")

    # Gate 1+2: the Diebold-Mariano test must be significant AND signed in the
    # system's favour (a strictly higher mean net return than buy-and-hold).
    dm_ok = dm_favours_system(dm_statistic, dm_pvalue, alpha=alpha)
    # Gate 3: the Deflated Sharpe must clear a CONFIDENCE threshold, not merely be
    # positive. The DSR is a probability in [0, 1] (the probability the true Sharpe
    # exceeds the multiplicity-adjusted, #signals x #param-config n_trials
    # benchmark), so a `> 0.0` test would be trivially satisfied by ANY positive
    # Sharpe and the gate would never bind. Require `> 1 - alpha` (e.g. 0.95) — the
    # standard Bailey-Lopez de Prado significance call — so the multiplicity
    # deflation has real teeth.
    dsr_ok = deflated_sharpe > (1.0 - alpha)
    # Gate 4: the Probability of Backtest Overfitting must be below one half — the
    # selected config is more likely than not the genuine best OOS, not overfit.
    pbo_ok = pbo < 0.5

    beats = dm_ok and dsr_ok and pbo_ok
    verdict = Verdict.SYSTEM_HAS_EDGE if beats else Verdict.NO_ROBUST_EDGE
    return VerdictResult(
        verdict=verdict,
        system_has_edge=beats,
        dm_pvalue=dm_pvalue,
        deflated_sharpe=deflated_sharpe,
        pbo=pbo,
        n_effective_trials=n_effective_trials,
    )
