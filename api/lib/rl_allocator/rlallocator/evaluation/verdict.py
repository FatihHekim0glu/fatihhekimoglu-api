"""Pure-function verdict derivation: ``rl_beats_baselines``.

The headline verdict is a PURE FUNCTION of the inference outputs. It CANNOT read
``True`` ("the PPO allocator beats the BEST of {1/N, Markowitz, risk-parity}
out-of-sample net of costs") unless ALL FOUR lines of evidence agree:

1. the MEDIAN-seed OOS net Sharpe beats the BEST baseline with a Diebold-Mariano-
   significant margin on the per-bar net-return differential (``dm_pvalue < alpha``
   AND ``dm_statistic > 0`` — a strictly *higher* mean net return);
2. the Deflated Sharpe (with the honest seed x HP ``n_trials``) clears the
   ``1 - alpha`` CONFIDENCE level (``deflated_sharpe > 1 - alpha``). The DSR is a
   PROBABILITY in ``[0, 1]``, so a ``> 0`` test would never bind — the gate is a
   confidence threshold, the standard Bailey-Lopez de Prado significance call;
3. the ACROSS-SEED Sharpe LOWER bound is strictly positive (``seed_sharpe_lo > 0``
   — the dispersion does not straddle zero, so the apparent skill is not a seed
   lottery);
4. the Probability of Backtest Overfitting is below one-half (``pbo < 0.5`` — the
   in-sample-best configuration is more likely than not the genuine OOS winner).

If ANY of the four fails, the verdict is
:attr:`Verdict.NO_SIGNIFICANT_DIFFERENCE` — the documented, leakage-free outcome:
the OOS Sharpe is dispersed around (and statistically indistinguishable from) the
baselines after the Deflated-Sharpe correction + the PBO check. The verdict is
derived from the evidence, never narrated. The truth table is unit-tested. No profit
claim is possible.

Importing this module has no side effects.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from rlallocator._exceptions import ValidationError
from rlallocator.evaluation.diebold_mariano import dm_favours_model


class Verdict(StrEnum):
    """Possible headline verdicts for the RL-allocator-vs-baselines comparison.

    The values are stable string identifiers safe to serialize across the API
    boundary and render in the frontend.
    """

    #: The PPO allocator beats the best baseline with a DM-significant median-seed
    #: margin, a DSR clearing 1-alpha, an across-seed Sharpe lower bound > 0, AND a
    #: PBO < 0.5.
    RL_BEATS_BASELINES = "rl_beats_baselines"

    #: The PPO allocator is not distinguishable from the best baseline (DM
    #: insignificant, DSR <= 1-alpha, the seed-Sharpe lower bound <= 0, or PBO >=
    #: 0.5) — the expected, honest-NULL outcome: the OOS Sharpe is indistinguishable
    #: from the baselines.
    NO_SIGNIFICANT_DIFFERENCE = "no_significant_difference"


@dataclass(frozen=True, slots=True)
class VerdictResult:
    """Immutable result of the pure verdict derivation.

    Attributes
    ----------
    verdict:
        The derived :class:`Verdict` enum value.
    rl_beats_baselines:
        ``True`` iff the median-seed margin cleared the DM-significance, the
        DSR-confidence, the positive-seed-lower-bound, AND the PBO < 0.5 gates.
        Mirrors ``verdict == Verdict.RL_BEATS_BASELINES``.
    dm_pvalue:
        The DM p-value of the median-seed RL net return vs. the best baseline that
        drove the verdict.
    deflated_sharpe:
        The Deflated Sharpe (honest seed x HP ``n_trials``) of the median-seed RL
        net return.
    seed_sharpe_lo:
        The across-seed OOS-Sharpe LOWER bound (the seed-lottery dispersion floor).
    pbo:
        The Probability of Backtest Overfitting (CSCV) across the configurations.
    n_effective_trials:
        The honest multiplicity count used for the DSR (#seeds x #HP configs).
    """

    verdict: Verdict
    rl_beats_baselines: bool
    dm_pvalue: float
    deflated_sharpe: float
    seed_sharpe_lo: float
    pbo: float
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
    pbo: float,
    n_effective_trials: int,
    *,
    alpha: float = 0.05,
) -> VerdictResult:
    r"""Derive the headline ``rl_beats_baselines`` verdict (pure function).

    Decision rule (truth-table unit-tested): ``rl_beats_baselines`` is ``True`` iff
    ALL of the following hold for the median-seed RL allocator vs. the BEST baseline:

    1. the Diebold-Mariano test on the per-bar net-return differential is
       significant AND signed in the RL allocator's favour (``dm_pvalue < alpha`` AND
       ``dm_statistic > 0`` — a strictly *higher* mean net return);
    2. the Deflated Sharpe (with the honest seed x HP ``n_effective_trials``) clears
       the ``1 - alpha`` CONFIDENCE level (``deflated_sharpe > 1 - alpha``) — the DSR
       is a probability, so the gate is a confidence threshold, NEVER ``> 0``;
    3. the across-seed OOS-Sharpe LOWER bound is strictly positive
       (``seed_sharpe_lo > 0`` — the seed-lottery dispersion does not straddle zero);
    4. the Probability of Backtest Overfitting is below one-half (``pbo < 0.5``).

    If ANY of the four fails, the verdict is
    :attr:`Verdict.NO_SIGNIFICANT_DIFFERENCE` — the documented honest-NULL outcome.
    This function MUST NOT return :attr:`Verdict.RL_BEATS_BASELINES` while the DM
    test is insignificant, the DSR is below the confidence level, the seed lower
    bound is non-positive, OR the PBO is at/above one-half, regardless of any point
    estimate. The verdict is a deterministic consequence of the evidence, never a
    narrative choice. No profit claim.

    Parameters
    ----------
    dm_statistic:
        The DM statistic of the median-seed RL net return vs. the best baseline
        (positive favours the RL allocator).
    dm_pvalue:
        The two-sided DM p-value of the median-seed RL net return vs. the best
        baseline.
    deflated_sharpe:
        The Deflated Sharpe (honest seed x HP ``n_trials``) of the median-seed RL
        net return — a PROBABILITY in ``[0, 1]``.
    seed_sharpe_lo:
        The across-seed OOS-Sharpe LOWER bound from the seed lottery.
    pbo:
        The CSCV Probability of Backtest Overfitting in ``[0, 1]``.
    n_effective_trials:
        The honest multiplicity count (#seeds x #HP configs).
    alpha:
        Significance / one-minus-confidence level (default ``0.05`` => DSR gate at
        ``> 0.95``).

    Returns
    -------
    VerdictResult
        The derived verdict and the evidence that produced it.

    Raises
    ------
    ValidationError
        If ``dm_pvalue`` is outside ``[0, 1]``, ``pbo`` is outside ``[0, 1]``, any
        input is non-finite, or ``n_effective_trials < 1``.
    """
    if not math.isfinite(dm_statistic):
        raise ValidationError(f"dm_statistic must be finite, got {dm_statistic}.")
    if not math.isfinite(dm_pvalue) or not 0.0 <= dm_pvalue <= 1.0:
        raise ValidationError(f"dm_pvalue must be in [0, 1], got {dm_pvalue}.")
    if not math.isfinite(deflated_sharpe):
        raise ValidationError(f"deflated_sharpe must be finite, got {deflated_sharpe}.")
    if not math.isfinite(seed_sharpe_lo):
        raise ValidationError(f"seed_sharpe_lo must be finite, got {seed_sharpe_lo}.")
    if not math.isfinite(pbo) or not 0.0 <= pbo <= 1.0:
        raise ValidationError(f"pbo must be in [0, 1], got {pbo}.")
    if n_effective_trials < 1:
        raise ValidationError(f"n_effective_trials must be >= 1, got {n_effective_trials}.")

    # Gate 1: the Diebold-Mariano test must be significant AND signed in the RL
    # allocator's favour (a strictly higher mean net return than the best baseline).
    dm_ok = dm_favours_model(dm_statistic, dm_pvalue, alpha=alpha)
    # Gate 2: the Deflated Sharpe must clear a CONFIDENCE threshold, not merely be
    # positive. The DSR is a probability in [0, 1] (the probability the true Sharpe
    # exceeds the multiplicity-adjusted, seed x HP n_trials benchmark), so a
    # `> 0.0` test would be trivially satisfied by ANY positive Sharpe and the gate
    # would never bind. Require `> 1 - alpha` (e.g. 0.95) — the standard
    # Bailey-Lopez de Prado significance call — so the multiplicity deflation has
    # real teeth.
    dsr_ok = deflated_sharpe > (1.0 - alpha)
    # Gate 3: the across-seed Sharpe LOWER bound must clear zero (the dispersion
    # does not straddle zero — the apparent skill is not a seed lottery).
    seed_ok = seed_sharpe_lo > 0.0
    # Gate 4: the Probability of Backtest Overfitting must be below one-half (the
    # in-sample-best config is more likely than not the genuine OOS winner).
    pbo_ok = pbo < 0.5

    beats = dm_ok and dsr_ok and seed_ok and pbo_ok
    verdict = Verdict.RL_BEATS_BASELINES if beats else Verdict.NO_SIGNIFICANT_DIFFERENCE
    return VerdictResult(
        verdict=verdict,
        rl_beats_baselines=beats,
        dm_pvalue=dm_pvalue,
        deflated_sharpe=deflated_sharpe,
        seed_sharpe_lo=seed_sharpe_lo,
        pbo=pbo,
        n_effective_trials=n_effective_trials,
    )
