"""Per-loan reason codes (adverse-action style explanations).

Returns the top contributing features for a single scored loan, with a direction
(raises / lowers default risk) and a signed contribution. The container-safe path
uses **weight-of-evidence / logistic coefficients** (no heavy dependency); SHAP is
available for richer DEV-only analysis but is NEVER imported in the shipped
container (it lives in the ``[dev]`` extra only).

Importing this module has no side effects (SHAP, if used, is imported lazily
inside the dev-only helper).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from lendingclub_default._exceptions import ValidationError

# quantcore-candidate: new code (reason codes); WOE/logit coeffs in-container,
# SHAP dev-only (never in the API image).


@dataclass(frozen=True, slots=True)
class ReasonCode:
    """A single feature's contribution to a loan's predicted default risk.

    Attributes
    ----------
    feature:
        The (human-readable) feature name.
    direction:
        ``"increases"`` if the feature pushes PD up, ``"decreases"`` if down.
    contribution:
        The signed contribution to the score (units depend on the explainer:
        log-odds for the logistic/WOE path).
    """

    feature: str
    direction: str
    contribution: float

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this reason code."""
        out = asdict(self)
        out["contribution"] = float(self.contribution)
        return out


def reason_codes_from_logit(
    coefficients: pd.Series,
    x_row: pd.Series,
    *,
    top_k: int = 5,
) -> list[ReasonCode]:
    """Compute reason codes from logistic coefficients for one loan (in-container).

    The signed contribution of feature ``j`` is ``coef_j * x_j`` (its additive
    push on the log-odds); the top ``top_k`` by absolute contribution are returned.

    Parameters
    ----------
    coefficients:
        The fitted logistic coefficients, indexed by feature name.
    x_row:
        The single loan's engineered feature vector, indexed to match.
    top_k:
        Number of reason codes to return.

    Returns
    -------
    list[ReasonCode]
        The top contributing features with direction and signed contribution.

    Raises
    ------
    ValidationError
        If ``top_k`` is not positive or the inputs share no common features.
    """
    if top_k <= 0:
        raise ValidationError("reason_codes_from_logit: top_k must be positive.")

    coef = pd.Series(coefficients).astype("float64")
    row = pd.Series(x_row).astype("float64")

    common = coef.index.intersection(row.index)
    if len(common) == 0:
        raise ValidationError(
            "reason_codes_from_logit: coefficients and x_row share no common features."
        )

    contributions = coef.reindex(common) * row.reindex(common)
    ordered = contributions.reindex(contributions.abs().sort_values(ascending=False).index)

    codes: list[ReasonCode] = []
    for feature, value in ordered.head(top_k).items():
        contribution = float(value)
        direction = "increases" if contribution >= 0.0 else "decreases"
        codes.append(
            ReasonCode(
                feature=str(feature),
                direction=direction,
                contribution=contribution,
            )
        )
    return codes


def shap_reason_codes(
    booster: object,
    x_row: np.ndarray,
    feature_names: list[str],
    *,
    top_k: int = 5,
) -> list[ReasonCode]:
    """DEV-ONLY: reason codes from SHAP values for the tree booster.

    SHAP is in the ``[dev]`` extra and imported LAZILY here; this function MUST
    NOT be called from the shipped container (which has no SHAP installed).

    Parameters
    ----------
    booster:
        The fitted tree booster.
    x_row:
        The single loan's feature vector.
    feature_names:
        Names aligned to ``x_row``.
    top_k:
        Number of reason codes to return.

    Returns
    -------
    list[ReasonCode]
        Top contributing features by absolute SHAP value.

    Raises
    ------
    ValidationError
        If ``top_k`` is not positive or the SHAP vector and names misalign.
    ImportError
        If SHAP is not installed (it lives in the ``[dev]`` extra only).
    """
    if top_k <= 0:
        raise ValidationError("shap_reason_codes: top_k must be positive.")

    # DEV-ONLY lazy import: SHAP lives in the [dev] extra and is never installed
    # in the shipped container, so this function must not run there. SHAP ships no
    # type stubs, so the import is locally suppressed (it is absent from the API
    # image's mypy surface).
    import shap  # type: ignore[import-untyped]

    explainer = shap.TreeExplainer(booster)
    row = np.asarray(x_row, dtype=np.float64).reshape(1, -1)
    values = np.asarray(explainer.shap_values(row), dtype=np.float64).ravel()

    if values.size != len(feature_names):
        raise ValidationError(
            "shap_reason_codes: SHAP vector and feature_names misaligned "
            f"({values.size} != {len(feature_names)})."
        )

    order = np.argsort(np.abs(values))[::-1][:top_k]
    codes: list[ReasonCode] = []
    for idx in order:
        contribution = float(values[idx])
        direction = "increases" if contribution >= 0.0 else "decreases"
        codes.append(
            ReasonCode(
                feature=str(feature_names[idx]),
                direction=direction,
                contribution=contribution,
            )
        )
    return codes
