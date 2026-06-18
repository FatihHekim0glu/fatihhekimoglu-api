"""Pure-function verdict derivation: ``deep_beats_naive``.

The headline verdict is a PURE FUNCTION of the inference outputs. It CANNOT read
``True`` ("a deep model beats the naive random walk") unless a deep model beats
naive with a Diebold-Mariano-significant margin AND a positive Deflated Sharpe.
This is what keeps the README honest on noisy daily returns, where the documented,
literature-consistent outcome is that PatchTST / the interpretable transformer /
the LSTM do NOT reliably beat a naive baseline: the verdict is derived from the
evidence, never narrated. The truth table is unit-tested.

There is NO price-level R² anywhere in this layer.

Importing this module has no side effects.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from mvtsforecast._exceptions import ValidationError
from mvtsforecast.evaluation.diebold_mariano import dm_favours_model


class Verdict(StrEnum):
    """Possible headline verdicts for the deep-vs-naive comparison.

    The values are stable string identifiers safe to serialize across the API
    boundary and render in the frontend.
    """

    #: A deep model beats naive with a DM-significant margin AND a positive DSR.
    DEEP_BEATS_NAIVE = "deep_beats_naive"

    #: No deep model is distinguishable from naive (DM insignificant or DSR <= 0)
    #: — the expected, literature-consistent outcome on noisy daily returns.
    NO_SIGNIFICANT_DIFFERENCE = "no_significant_difference"


@dataclass(frozen=True, slots=True)
class VerdictResult:
    """Immutable result of the pure verdict derivation.

    Attributes
    ----------
    verdict:
        The derived :class:`Verdict` enum value.
    deep_beats_naive:
        ``True`` iff a deep model cleared BOTH the DM-significance and
        positive-DSR gates. Mirrors ``verdict == Verdict.DEEP_BEATS_NAIVE``.
    best_deep_model:
        Name of the best-performing deep model (lowest RMSE), for reporting.
    dm_pvalue:
        The DM p-value of the best deep model vs. naive that drove the verdict.
    deflated_sharpe:
        The Deflated Sharpe (FULL-grid ``n_trials``) of the best deep model.
    n_effective_trials:
        The multiplicity count used for the DSR (architectures x HP configs).
    """

    verdict: Verdict
    deep_beats_naive: bool
    best_deep_model: str
    dm_pvalue: float
    deflated_sharpe: float
    n_effective_trials: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        out = asdict(self)
        out["verdict"] = self.verdict.value
        return out


def derive_verdict(
    best_deep_model: str,
    dm_statistic: float,
    dm_pvalue: float,
    deflated_sharpe: float,
    n_effective_trials: int,
    *,
    alpha: float = 0.05,
) -> VerdictResult:
    r"""Derive the headline ``deep_beats_naive`` verdict (pure function).

    Decision rule (truth-table unit-tested): ``deep_beats_naive`` is ``True`` iff
    ALL of the following hold for the best deep model:

    1. the Diebold-Mariano test is significant (``dm_pvalue < alpha``);
    2. the DM statistic is signed in the model's favour (``dm_statistic < 0`` —
       strictly lower squared-error loss than the naive forecast);
    3. the Deflated Sharpe is strictly positive (``deflated_sharpe > 0`` against
       the multiplicity-inflated benchmark with ``n_effective_trials``).

    If ANY of the three fails, the verdict is
    :attr:`Verdict.NO_SIGNIFICANT_DIFFERENCE` — the expected outcome on noisy
    daily returns. This function MUST NOT return :attr:`Verdict.DEEP_BEATS_NAIVE`
    while the DM test is insignificant or the DSR is non-positive, regardless of
    any point estimate. The verdict is a deterministic consequence of the
    evidence, never a narrative choice.

    Parameters
    ----------
    best_deep_model:
        Name of the best-performing (lowest-RMSE) deep model.
    dm_statistic:
        The Diebold-Mariano statistic of the best deep model vs. naive
        (negative favours the model).
    dm_pvalue:
        The two-sided DM p-value of the best deep model vs. naive.
    deflated_sharpe:
        The Deflated Sharpe (FULL-grid ``n_trials``) of the best deep model.
    n_effective_trials:
        The honest multiplicity count (#architectures x #HP configs).
    alpha:
        Significance level for the DM test (default ``0.05``).

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
    # model's favour (strictly lower squared-error loss than the naive forecast).
    dm_ok = dm_favours_model(dm_statistic, dm_pvalue, alpha=alpha)
    # Gate 3: the Deflated Sharpe (FULL-grid n_trials) must clear zero.
    dsr_ok = deflated_sharpe > 0.0

    beats = dm_ok and dsr_ok
    verdict = Verdict.DEEP_BEATS_NAIVE if beats else Verdict.NO_SIGNIFICANT_DIFFERENCE
    return VerdictResult(
        verdict=verdict,
        deep_beats_naive=beats,
        best_deep_model=best_deep_model,
        dm_pvalue=dm_pvalue,
        deflated_sharpe=deflated_sharpe,
        n_effective_trials=n_effective_trials,
    )
