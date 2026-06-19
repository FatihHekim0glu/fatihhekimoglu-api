"""Pure-function verdict derivation: ``rl_beats_baseline``.

The headline verdict is a PURE FUNCTION of the inference outputs. It CANNOT read
``True`` ("the PPO agent beats buy-and-hold out-of-sample net of costs") unless ALL
THREE lines of evidence agree:

1. the MEDIAN-seed OOS net Sharpe beats buy-and-hold with a Diebold-Mariano-
   significant margin on the per-bar net-return differential (``dm_pvalue < alpha``
   AND ``dm_statistic > 0``);
2. the Deflated Sharpe (with the honest seed x HP ``n_trials``) is strictly
   positive (``deflated_sharpe > 0``);
3. the ACROSS-SEED Sharpe LOWER bound is strictly positive (``seed_sharpe_lo > 0``
   — the dispersion does not straddle zero, so the apparent skill is not a seed
   lottery).

If ANY of the three fails, the verdict is
:attr:`Verdict.NO_SIGNIFICANT_DIFFERENCE` — the documented, leakage-free outcome:
the OOS Sharpe is dispersed around (and statistically indistinguishable from) zero
after the Deflated-Sharpe correction. The verdict is derived from the evidence,
never narrated. The truth table is unit-tested. No profit claim is possible.

Importing this module has no side effects.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from rltrader._exceptions import ValidationError
from rltrader.evaluation.diebold_mariano import dm_favours_model


class Verdict(StrEnum):
    """Possible headline verdicts for the RL-vs-buy-hold comparison.

    The values are stable string identifiers safe to serialize across the API
    boundary and render in the frontend.
    """

    #: The PPO agent beats buy-and-hold with a DM-significant median-seed margin,
    #: a positive DSR, AND an across-seed Sharpe lower bound > 0.
    RL_BEATS_BASELINE = "rl_beats_baseline"

    #: The PPO agent is not distinguishable from buy-and-hold (DM insignificant,
    #: DSR <= 0, or the seed-Sharpe lower bound <= 0) — the expected, honest-NULL
    #: outcome: the OOS Sharpe is indistinguishable from zero.
    NO_SIGNIFICANT_DIFFERENCE = "no_significant_difference"


@dataclass(frozen=True, slots=True)
class VerdictResult:
    """Immutable result of the pure verdict derivation.

    Attributes
    ----------
    verdict:
        The derived :class:`Verdict` enum value.
    rl_beats_baseline:
        ``True`` iff the median-seed margin cleared the DM-significance, the
        positive-DSR, AND the positive-seed-lower-bound gates. Mirrors
        ``verdict == Verdict.RL_BEATS_BASELINE``.
    dm_pvalue:
        The DM p-value of the median-seed RL net return vs. buy-and-hold that drove
        the verdict.
    deflated_sharpe:
        The Deflated Sharpe (honest seed x HP ``n_trials``) of the median-seed RL
        net return.
    seed_sharpe_lo:
        The across-seed OOS-Sharpe LOWER bound (the seed-lottery dispersion floor).
    n_effective_trials:
        The honest multiplicity count used for the DSR (#seeds x #HP configs).
    """

    verdict: Verdict
    rl_beats_baseline: bool
    dm_pvalue: float
    deflated_sharpe: float
    seed_sharpe_lo: float
    n_effective_trials: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        out = asdict(self)
        out["verdict"] = self.verdict.value
        return out


def derive_verdict(
    dm_statistic: float,
    dm_pvalue: float,
    deflated_sharpe: float,
    seed_sharpe_lo: float,
    n_effective_trials: int,
    *,
    alpha: float = 0.05,
) -> VerdictResult:
    r"""Derive the headline ``rl_beats_baseline`` verdict (pure function).

    Decision rule (truth-table unit-tested): ``rl_beats_baseline`` is ``True`` iff
    ALL of the following hold for the median-seed RL strategy vs. buy-and-hold:

    1. the Diebold-Mariano test on the per-bar net-return differential is
       significant AND signed in the RL agent's favour (``dm_pvalue < alpha`` AND
       ``dm_statistic > 0`` — a strictly *higher* mean net return);
    2. the Deflated Sharpe (with the honest seed x HP ``n_effective_trials``) is
       strictly positive (``deflated_sharpe > 0``);
    3. the across-seed OOS-Sharpe LOWER bound is strictly positive
       (``seed_sharpe_lo > 0`` — the seed-lottery dispersion does not straddle
       zero).

    If ANY of the three fails, the verdict is
    :attr:`Verdict.NO_SIGNIFICANT_DIFFERENCE` — the documented honest-NULL outcome.
    This function MUST NOT return :attr:`Verdict.RL_BEATS_BASELINE` while the DM
    test is insignificant, the DSR is non-positive, OR the seed lower bound is
    non-positive, regardless of any point estimate. The verdict is a deterministic
    consequence of the evidence, never a narrative choice. No profit claim.

    Parameters
    ----------
    dm_statistic:
        The DM statistic of the median-seed RL net return vs. buy-and-hold
        (positive favours the RL agent).
    dm_pvalue:
        The two-sided DM p-value of the median-seed RL net return vs. buy-and-hold.
    deflated_sharpe:
        The Deflated Sharpe (honest seed x HP ``n_trials``) of the median-seed RL
        net return.
    seed_sharpe_lo:
        The across-seed OOS-Sharpe LOWER bound from the seed lottery.
    n_effective_trials:
        The honest multiplicity count (#seeds x #HP configs).
    alpha:
        Significance level for the DM test (default ``0.05``).

    Returns
    -------
    VerdictResult
        The derived verdict and the evidence that produced it.

    Raises
    ------
    ValidationError
        If ``dm_pvalue`` is outside ``[0, 1]``, any input is non-finite, or
        ``n_effective_trials < 1``.
    """
    if not math.isfinite(dm_statistic):
        raise ValidationError(f"dm_statistic must be finite, got {dm_statistic}.")
    if not math.isfinite(dm_pvalue) or not 0.0 <= dm_pvalue <= 1.0:
        raise ValidationError(f"dm_pvalue must be in [0, 1], got {dm_pvalue}.")
    if not math.isfinite(deflated_sharpe):
        raise ValidationError(f"deflated_sharpe must be finite, got {deflated_sharpe}.")
    if not math.isfinite(seed_sharpe_lo):
        raise ValidationError(f"seed_sharpe_lo must be finite, got {seed_sharpe_lo}.")
    if n_effective_trials < 1:
        raise ValidationError(f"n_effective_trials must be >= 1, got {n_effective_trials}.")

    # Gate 1+2: the Diebold-Mariano test must be significant AND signed in the RL
    # agent's favour (a strictly higher mean net return than buy-and-hold).
    dm_ok = dm_favours_model(dm_statistic, dm_pvalue, alpha=alpha)
    # Gate 3: the Deflated Sharpe must clear a CONFIDENCE threshold, not merely be
    # positive. The DSR is a probability in [0, 1] (the probability the true Sharpe
    # exceeds the multiplicity-adjusted, seed x HP n_trials benchmark), so a
    # `> 0.0` test would be trivially satisfied by ANY positive Sharpe and the gate
    # would never bind. Require `> 1 - alpha` (e.g. 0.95) — the standard
    # Bailey-Lopez de Prado significance call — so the multiplicity deflation has
    # real teeth.
    dsr_ok = deflated_sharpe > (1.0 - alpha)
    # Gate 4: the across-seed Sharpe LOWER bound must clear zero (the dispersion
    # does not straddle zero — the apparent skill is not a seed lottery).
    seed_ok = seed_sharpe_lo > 0.0

    beats = dm_ok and dsr_ok and seed_ok
    verdict = Verdict.RL_BEATS_BASELINE if beats else Verdict.NO_SIGNIFICANT_DIFFERENCE
    return VerdictResult(
        verdict=verdict,
        rl_beats_baseline=beats,
        dm_pvalue=dm_pvalue,
        deflated_sharpe=deflated_sharpe,
        seed_sharpe_lo=seed_sharpe_lo,
        n_effective_trials=n_effective_trials,
    )
