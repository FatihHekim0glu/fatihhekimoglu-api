"""Pure-function verdict derivation for the GARCH-vs-ML horse race.

The headline outputs ``best_model`` and ``ml_beats_garch`` are PURE FUNCTIONS of
the out-of-sample QLIKE-by-model dict and the Diebold-Mariano / Hansen-SPA
significance — never a narrative choice. ``ml_beats_garch`` is ``True`` ONLY when
an ML model (XGBoost / LSTM) has the strictly lowest OOS QLIKE AND beats the best
GARCH/HAR-RV reference by a margin that is SPA-significant (and DM-significant
pairwise). This is the encoded honest null: on GARCH-generated data the ML arm
will not clear that bar, so the function returns ``ml_beats_garch=False`` — by
construction, not by editorial choice.

The truth table is unit-tested. Importing this module has no side effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from volforecast._exceptions import ValidationError

#: Model labels classified as "ML" (the challengers).
ML_MODELS: frozenset[str] = frozenset({"xgboost", "lstm"})

#: Model labels classified as the GARCH/HAR-RV references (the bars to beat).
REFERENCE_MODELS: frozenset[str] = frozenset({"garch", "egarch", "gjr", "har_rv", "ewma", "rw"})


class BestModelClass(StrEnum):
    """Which family produced the lowest OOS QLIKE.

    Stable string identifiers safe to serialize across the API boundary
    (``BestModelClass.ML.value == "ml"``).
    """

    #: An ML challenger (XGBoost / LSTM) had the lowest OOS QLIKE.
    ML = "ml"
    #: A GARCH/HAR-RV/baseline reference had the lowest OOS QLIKE (the null).
    REFERENCE = "reference"


@dataclass(frozen=True, slots=True)
class Verdict:
    """Immutable, JSON-safe verdict for the horse race.

    Attributes
    ----------
    best_model:
        The label of the model with the strictly lowest OOS QLIKE.
    best_model_class:
        Whether ``best_model`` is an ML challenger or a reference.
    ml_beats_garch:
        ``True`` iff an ML model has the lowest QLIKE AND beats the best
        reference by an SPA- and DM-significant margin. The honest null makes
        this ``False`` on GARCH-generated data.
    spa_pvalue:
        The Hansen-SPA consistent p-value for the composite null (no model beats
        the reference benchmark).
    dm_pvalue_vs_best_reference:
        The pairwise DM p-value of ``best_model`` against the best reference
        (``None`` when ``best_model`` is itself the reference).
    """

    best_model: str
    best_model_class: BestModelClass
    ml_beats_garch: bool
    spa_pvalue: float
    dm_pvalue_vs_best_reference: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this verdict."""
        return {
            "best_model": str(self.best_model),
            "best_model_class": self.best_model_class.value,
            "ml_beats_garch": bool(self.ml_beats_garch),
            "spa_pvalue": float(self.spa_pvalue),
            "dm_pvalue_vs_best_reference": (
                None
                if self.dm_pvalue_vs_best_reference is None
                else float(self.dm_pvalue_vs_best_reference)
            ),
        }


def derive_verdict(
    qlike_by_model: dict[str, float],
    spa_pvalue: float,
    dm_pvalues_vs_best: dict[str, float],
    *,
    alpha: float = 0.05,
) -> Verdict:
    r"""Derive ``best_model`` / ``ml_beats_garch`` from OOS QLIKE + significance.

    Decision rule (truth-table unit-tested):

    1. ``best_model`` is the key of ``qlike_by_model`` with the strictly minimum
       QLIKE (lower is better). Ties are broken deterministically in favour of
       the REFERENCE family (the honest, conservative tie-break).
    2. ``ml_beats_garch`` is ``True`` IFF ALL of:
       (a) ``best_model`` is in :data:`ML_MODELS`;
       (b) the Hansen-SPA p-value clears significance (``spa_pvalue < alpha``),
           i.e. the composite "no model beats the reference" null is rejected;
       (c) the pairwise DM p-value of ``best_model`` vs the best reference is
           significant (``dm_pvalues_vs_best[best_model] < alpha``).
       If any condition fails, ``ml_beats_garch`` is ``False``.

    HONESTY REQUIREMENT: this function MUST NOT report ``ml_beats_garch=True`` on
    the strength of a lower point QLIKE alone — the SPA and DM gates are
    mandatory. On GARCH-generated data those gates do not clear, so the function
    returns ``False`` (the honest null), by construction.

    Parameters
    ----------
    qlike_by_model:
        Mapping ``{model_label: mean_OOS_QLIKE}`` over the full model set
        (references and challengers).
    spa_pvalue:
        The Hansen-SPA consistent p-value (composite null over the set).
    dm_pvalues_vs_best:
        Mapping ``{model_label: DM_pvalue_vs_best_reference}`` for the challengers
        (at least ``best_model`` must be present when it is an ML model).
    alpha:
        Significance level for both the SPA and DM gates (default ``0.05``).

    Returns
    -------
    Verdict
        The derived, frozen verdict.

    Raises
    ------
    ValidationError
        If ``qlike_by_model`` is empty, contains non-finite QLIKE values, any
        p-value is outside ``[0, 1]``, or ``best_model`` is an ML model but its
        DM p-value is missing from ``dm_pvalues_vs_best``.
    """
    if not qlike_by_model:
        raise ValidationError("qlike_by_model must be non-empty.")
    for label, value in qlike_by_model.items():
        if not math.isfinite(value):
            raise ValidationError(f"qlike_by_model[{label!r}] is not finite ({value}).")
    if not 0.0 <= spa_pvalue <= 1.0:
        raise ValidationError(f"spa_pvalue must be in [0, 1], got {spa_pvalue}.")
    for label, pval in dm_pvalues_vs_best.items():
        if not 0.0 <= pval <= 1.0:
            raise ValidationError(f"dm_pvalues_vs_best[{label!r}] must be in [0, 1], got {pval}.")

    # 1) best_model = strict argmin of QLIKE, with ties broken conservatively in
    #    favour of the REFERENCE family (the honest, anti-snooping tie-break).
    min_qlike = min(qlike_by_model.values())
    tied = [m for m, q in qlike_by_model.items() if q == min_qlike]
    reference_tied = [m for m in tied if m in REFERENCE_MODELS]
    # Deterministic tie-break: prefer a reference; otherwise the first inserted.
    best_model = reference_tied[0] if reference_tied else tied[0]

    best_is_ml = best_model in ML_MODELS

    # 2) ml_beats_garch: only if best_model is ML AND both the SPA composite null
    #    is rejected AND the pairwise DM gate vs the best reference is significant.
    dm_pvalue_vs_best_reference: float | None = None
    if best_is_ml:
        if best_model not in dm_pvalues_vs_best:
            raise ValidationError(
                f"best_model {best_model!r} is an ML model but its DM p-value is "
                "missing from dm_pvalues_vs_best."
            )
        dm_pvalue_vs_best_reference = float(dm_pvalues_vs_best[best_model])

    ml_beats_garch = bool(
        best_is_ml
        and spa_pvalue < alpha
        and dm_pvalue_vs_best_reference is not None
        and dm_pvalue_vs_best_reference < alpha
    )

    best_model_class = BestModelClass.ML if best_is_ml else BestModelClass.REFERENCE

    return Verdict(
        best_model=best_model,
        best_model_class=best_model_class,
        ml_beats_garch=ml_beats_garch,
        spa_pvalue=float(spa_pvalue),
        dm_pvalue_vs_best_reference=dm_pvalue_vs_best_reference,
    )
