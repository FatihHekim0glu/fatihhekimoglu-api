"""Pure-function verdict derivation: ``gnn_beats_baseline``.

The headline verdict is a PURE FUNCTION of the inference outputs. It CANNOT read
``True`` ("a GNN beats the best per-node ridge / cross-sectional-momentum
baseline") unless a GNN beats that baseline with a Diebold-Mariano-significant
margin on the rank-IC / long-short differential AND a Deflated Sharpe clearing the
``1 - alpha`` confidence threshold (``>= 0.95``) net of costs. This is what keeps the README honest: message passing over the
correlation graph recovers sector/cluster structure (DESCRIPTIVE) but the
documented, leakage-free outcome is that the GCN / GraphSAGE do NOT reliably beat
the baselines on OOS rank-IC or long-short decile spread. The verdict is derived
from the evidence, never narrated. The truth table is unit-tested.

There is NO baseline-free in-sample R²/accuracy anywhere in this layer.

Importing this module has no side effects.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from gnnstocks._exceptions import ValidationError
from gnnstocks.evaluation.diebold_mariano import dm_favours_model


class Verdict(StrEnum):
    """Possible headline verdicts for the GNN-vs-baseline comparison.

    The values are stable string identifiers safe to serialize across the API
    boundary and render in the frontend.
    """

    #: A GNN beats the best baseline with a DM-significant rank-IC / long-short
    #: margin AND a DSR clearing the 1 - alpha confidence threshold (>= 0.95).
    GNN_BEATS_BASELINE = "gnn_beats_baseline"

    #: No GNN is distinguishable from the best baseline (DM insignificant or
    #: DSR < 0.95 (1 - alpha)) — the expected, leakage-free outcome.
    NO_SIGNIFICANT_DIFFERENCE = "no_significant_difference"


@dataclass(frozen=True, slots=True)
class VerdictResult:
    """Immutable result of the pure verdict derivation.

    Attributes
    ----------
    verdict:
        The derived :class:`Verdict` enum value.
    gnn_beats_baseline:
        ``True`` iff a GNN cleared BOTH the DM-significance gate and the DSR
        ``1 - alpha`` confidence gate (``deflated_sharpe >= 0.95``). Mirrors
        ``verdict == Verdict.GNN_BEATS_BASELINE``.
    best_gnn_model:
        Name of the best-performing GNN (highest OOS rank-IC), for reporting.
    best_baseline:
        Name of the best baseline the GNN was tested against.
    dm_pvalue:
        The DM p-value of the best GNN vs. the best baseline that drove the
        verdict.
    deflated_sharpe:
        The Deflated Sharpe (FULL-grid ``n_trials``) of the best GNN's
        long-short net return.
    n_effective_trials:
        The multiplicity count used for the DSR (graph-types x models x HP grid).
    """

    verdict: Verdict
    gnn_beats_baseline: bool
    best_gnn_model: str
    best_baseline: str
    dm_pvalue: float
    deflated_sharpe: float
    n_effective_trials: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        out = asdict(self)
        out["verdict"] = self.verdict.value
        return out


def derive_verdict(
    best_gnn_model: str,
    best_baseline: str,
    dm_statistic: float,
    dm_pvalue: float,
    deflated_sharpe: float,
    n_effective_trials: int,
    *,
    alpha: float = 0.05,
    dsr_threshold: float = 0.95,
) -> VerdictResult:
    r"""Derive the headline ``gnn_beats_baseline`` verdict (pure function).

    Decision rule (truth-table unit-tested): ``gnn_beats_baseline`` is ``True``
    iff ALL of the following hold for the best GNN vs. the best baseline:

    1. the Diebold-Mariano test on the rank-IC / long-short differential is
       significant (``dm_pvalue < alpha``);
    2. the DM statistic is signed in the model's favour (``dm_statistic > 0`` — a
       strictly *higher* mean rank-IC / long-short return than the baseline);
    3. the Deflated Sharpe of the GNN's long-short net return clears the
       ``1 - alpha`` confidence threshold (``deflated_sharpe >= dsr_threshold``,
       default ``>= 0.95``) against the multiplicity-inflated benchmark with
       ``n_effective_trials`` = graph-types x models x HP grid.

    If ANY of the three fails, the verdict is
    :attr:`Verdict.NO_SIGNIFICANT_DIFFERENCE` — the expected, leakage-free
    outcome. This function MUST NOT return :attr:`Verdict.GNN_BEATS_BASELINE`
    while the DM test is insignificant or the DSR fails the ``1 - alpha``
    confidence threshold (``deflated_sharpe < dsr_threshold``), regardless of
    any point estimate. The verdict is a deterministic consequence of the
    evidence, never a narrative choice.

    Parameters
    ----------
    best_gnn_model:
        Name of the best-performing (highest OOS rank-IC) GNN.
    best_baseline:
        Name of the best baseline the GNN was tested against.
    dm_statistic:
        The Diebold-Mariano statistic of the best GNN vs. the best baseline
        (positive favours the GNN on a skill differential).
    dm_pvalue:
        The two-sided DM p-value of the best GNN vs. the best baseline.
    deflated_sharpe:
        The Deflated Sharpe (FULL-grid ``n_trials``) of the best GNN's
        long-short net return.
    n_effective_trials:
        The honest multiplicity count (#graph-types x #models x #HP configs).
    alpha:
        Significance level for the DM test (default ``0.05``).
    dsr_threshold:
        Minimum Deflated Sharpe (a probability in ``[0, 1]``) required to clear
        the overfitting gate. Defaults to ``0.95`` (the portfolio-standard
        ``1 - alpha`` confidence level); the DSR must satisfy
        ``deflated_sharpe >= dsr_threshold``.

    Returns
    -------
    VerdictResult
        The derived verdict and the evidence that produced it.

    Raises
    ------
    ValidationError
        If ``dm_pvalue`` is outside ``[0, 1]`` or ``n_effective_trials < 1``.
    """
    if not math.isfinite(dm_statistic):
        raise ValidationError(f"dm_statistic must be finite, got {dm_statistic}.")
    if not math.isfinite(dm_pvalue) or not 0.0 <= dm_pvalue <= 1.0:
        raise ValidationError(f"dm_pvalue must be in [0, 1], got {dm_pvalue}.")
    if not math.isfinite(deflated_sharpe):
        raise ValidationError(f"deflated_sharpe must be finite, got {deflated_sharpe}.")
    if n_effective_trials < 1:
        raise ValidationError(f"n_effective_trials must be >= 1, got {n_effective_trials}.")

    # Gate 1+2: the Diebold-Mariano test must be significant AND signed in the
    # model's favour (a strictly higher mean rank-IC / long-short return).
    dm_ok = dm_favours_model(dm_statistic, dm_pvalue, alpha=alpha)
    # Gate 3: the Deflated Sharpe (FULL-grid n_trials) must clear the 1 - alpha
    # confidence threshold (default >= 0.95). The DSR is a probability in [0, 1];
    # gating at 0.95 is the portfolio standard, not the vacuous "> 0".
    dsr_ok = deflated_sharpe >= dsr_threshold

    beats = dm_ok and dsr_ok
    verdict = Verdict.GNN_BEATS_BASELINE if beats else Verdict.NO_SIGNIFICANT_DIFFERENCE
    return VerdictResult(
        verdict=verdict,
        gnn_beats_baseline=beats,
        best_gnn_model=best_gnn_model,
        best_baseline=best_baseline,
        dm_pvalue=dm_pvalue,
        deflated_sharpe=deflated_sharpe,
        n_effective_trials=n_effective_trials,
    )
