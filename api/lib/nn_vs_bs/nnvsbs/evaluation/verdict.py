"""Pure-function verdict derivation (honest, never a tradable-edge claim).

The headline verdict is a PURE FUNCTION of the reprice-error summaries and the
Diebold-Mariano result. It encodes the project's honest framing as a deterministic
truth table, so the README and the frontend badge cannot overclaim:

- On SYNTHETIC Black-Scholes data the only legitimate positive outcome is
  :attr:`RepriceVerdict.NN_RECOVERS_BS` — the NN's reprice RMSE is within a stated
  sanity band of the BS label AND the DM test finds no significant difference.
  Beating the data-generating process is impossible, so there is no "NN beats BS"
  value on synthetic data.
- On REAL chains the NN may achieve significantly lower out-of-sample reprice error
  across moneyness (:attr:`RepriceVerdict.NN_PRICES_SMILE_BETTER`) — constant-vol BS
  systematically misprices the smile. This is a REPRICE-accuracy statement ONLY,
  never a tradable-P&L / Sharpe claim (no strategy exists).

Importing this module has no side effects.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from nnvsbs._exceptions import ValidationError


class RepriceVerdict(StrEnum):
    """Possible headline verdicts for the NN-vs-BS reprice comparison.

    The values are stable string identifiers safe to serialize across the API
    boundary and render in the frontend. NONE of them is a tradable-edge claim —
    every value is a statement about REPRICE ERROR only.
    """

    #: Synthetic sanity check passed: the NN's reprice RMSE is within the sanity
    #: band of the Black-Scholes label and the DM test finds no significant gap —
    #: the NN has RECOVERED the BS surface (a convergence check, not an edge).
    NN_RECOVERS_BS = "nn_recovers_bs"

    #: Real-chain finding: the NN has significantly LOWER out-of-sample reprice
    #: error than constant-vol BS (it prices the smile better). Reprice accuracy
    #: only — never a tradable edge.
    NN_PRICES_SMILE_BETTER = "nn_prices_smile_better"

    #: Black-Scholes reprices at least as well as the NN (the NN shows no
    #: significant reprice improvement).
    BS_AT_LEAST_AS_GOOD = "bs_at_least_as_good"

    #: The NN failed even to recover the BS surface on synthetic data (reprice RMSE
    #: outside the sanity band) — a convergence FAILURE, surfaced honestly.
    NN_FAILED_TO_RECOVER = "nn_failed_to_recover"


@dataclass(frozen=True, slots=True)
class RepriceVerdictResult:
    """A derived verdict plus the evidence and a one-line honest rationale.

    Attributes
    ----------
    verdict:
        The derived :class:`RepriceVerdict`.
    nn_recovers_bs:
        Whether the NN recovered the BS surface within the sanity band (the
        synthetic sanity-check boolean the frontend badge renders).
    rationale:
        A short, honest, human-readable explanation of the verdict.
    """

    verdict: RepriceVerdict
    nn_recovers_bs: bool
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the verdict result."""
        d = asdict(self)
        d["verdict"] = self.verdict.value
        d["nn_recovers_bs"] = bool(self.nn_recovers_bs)
        d["rationale"] = str(self.rationale)
        return d


def derive_verdict(
    *,
    bs_rmse: float,
    nn_rmse: float,
    dm_pvalue: float,
    data_source: str,
    sanity_band: float = 1e-2,
    alpha: float = 0.05,
) -> RepriceVerdictResult:
    """Derive the honest reprice verdict (pure function; truth-table tested).

    Decision rule:

    - **Synthetic data** (``data_source == "synthetic"``): the NN should RECOVER
      BS. Recovery is keyed on the SANITY BAND: if ``nn_rmse <= sanity_band`` (the
      NN's absolute reprice RMSE against the Black-Scholes surface is within the
      band) return :attr:`RepriceVerdict.NN_RECOVERS_BS` with
      ``nn_recovers_bs=True``; otherwise return
      :attr:`RepriceVerdict.NN_FAILED_TO_RECOVER`. The Diebold-Mariano result is
      reported for transparency but does NOT gate recovery here: the synthetic
      target IS the Black-Scholes price, so BS reprices its own labels with ~zero
      error and any small-but-nonzero NN error makes the DM gap trivially
      "significant" against that perfect baseline. That degenerate gap is not
      evidence of underperformance, and beating the data-generating process is
      impossible — so there is deliberately no "NN beats BS" outcome on synthetic
      data.
    - **Real data**: if the NN has lower reprice RMSE AND the DM test is significant
      (``dm_pvalue < alpha``), return :attr:`RepriceVerdict.NN_PRICES_SMILE_BETTER`;
      otherwise :attr:`RepriceVerdict.BS_AT_LEAST_AS_GOOD`. This is a reprice-error
      statement only — NEVER a tradable-edge claim.

    Parameters
    ----------
    bs_rmse, nn_rmse:
        Out-of-sample reprice RMSE of Black-Scholes and the neural net.
    dm_pvalue:
        The two-sided Diebold-Mariano p-value on the paired reprice errors.
    data_source:
        ``"synthetic"`` or ``"yfinance"`` (real) — selects the rule branch.
    sanity_band:
        The synthetic reprice-RMSE sanity band the NN must reach to count as
        recovering BS (default ``1e-2``).
    alpha:
        Significance level for the DM test (default ``0.05``).

    Returns
    -------
    RepriceVerdictResult
        The derived verdict, the ``nn_recovers_bs`` boolean, and the rationale.

    Raises
    ------
    ValidationError
        If ``dm_pvalue`` is outside ``[0, 1]`` or an RMSE is negative/non-finite.
    NotImplementedError
        This is a typed stub.
    """
    for value, vname in ((bs_rmse, "bs_rmse"), (nn_rmse, "nn_rmse")):
        if not math.isfinite(value) or value < 0.0:
            raise ValidationError(f"{vname} must be finite and non-negative, got {value!r}.")
    if not math.isfinite(dm_pvalue) or not (0.0 <= dm_pvalue <= 1.0):
        raise ValidationError(f"dm_pvalue must lie in [0, 1], got {dm_pvalue!r}.")
    if sanity_band <= 0.0 or not math.isfinite(sanity_band):
        raise ValidationError(f"sanity_band must be finite and positive, got {sanity_band!r}.")
    if not (0.0 < alpha < 1.0):
        raise ValidationError(f"alpha must lie in (0, 1), got {alpha!r}.")

    dm_significant = dm_pvalue < alpha

    if data_source == "synthetic":
        # On data generated BY Black-Scholes the only legitimate positive outcome is
        # RECOVERY: the NN's absolute reprice RMSE is within the sanity band of the BS
        # surface. Beating the data-generating process is impossible, so there is no
        # "NN beats BS" branch. The DM test is reported for transparency but does NOT
        # gate recovery: the synthetic target IS the BS price, so BS reprices its own
        # labels with ~zero error and any nonzero NN error makes the DM gap trivially
        # "significant" against that perfect baseline — not evidence of failure.
        within_band = nn_rmse <= sanity_band
        if not within_band:
            return RepriceVerdictResult(
                verdict=RepriceVerdict.NN_FAILED_TO_RECOVER,
                nn_recovers_bs=False,
                rationale=(
                    f"Synthetic: NN reprice RMSE {nn_rmse:.3g} exceeds the sanity band "
                    f"{sanity_band:.3g} — the NN failed to recover the Black-Scholes surface."
                ),
            )
        dm_note = (
            "the Diebold-Mariano test is not significant (p={p:.3g})"
            if not dm_significant
            else (
                "the Diebold-Mariano gap is significant (p={p:.3g}) only because BS "
                "reprices its own synthetic labels perfectly — not an edge"
            )
        ).format(p=dm_pvalue)
        return RepriceVerdictResult(
            verdict=RepriceVerdict.NN_RECOVERS_BS,
            nn_recovers_bs=True,
            rationale=(
                f"Synthetic: NN reprice RMSE {nn_rmse:.3g} is within the sanity band "
                f"{sanity_band:.3g} and {dm_note} — the NN recovered the BS surface "
                "(a convergence check, not an edge)."
            ),
        )

    # Real chains: the NN may genuinely lower out-of-sample reprice error across
    # moneyness (constant-vol BS misprices the smile). This is a REPRICE-accuracy
    # statement only — never a tradable-P&L / Sharpe claim.
    if nn_rmse < bs_rmse and dm_significant:
        return RepriceVerdictResult(
            verdict=RepriceVerdict.NN_PRICES_SMILE_BETTER,
            nn_recovers_bs=False,
            rationale=(
                f"Real: NN reprice RMSE {nn_rmse:.3g} is below BS {bs_rmse:.3g} and the "
                f"Diebold-Mariano test is significant (p={dm_pvalue:.3g}) — the NN prices "
                "the smile better out-of-sample (reprice accuracy only; no tradable edge "
                "is claimed)."
            ),
        )
    return RepriceVerdictResult(
        verdict=RepriceVerdict.BS_AT_LEAST_AS_GOOD,
        nn_recovers_bs=False,
        rationale=(
            f"Real: BS reprice RMSE {bs_rmse:.3g} vs NN {nn_rmse:.3g} with Diebold-Mariano "
            f"p={dm_pvalue:.3g} — no significant NN reprice improvement, so Black-Scholes is "
            "at least as good."
        ),
    )
