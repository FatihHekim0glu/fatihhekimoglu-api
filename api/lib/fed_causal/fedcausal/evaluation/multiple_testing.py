"""Multiple-testing corrections across the FULL specification grid.

The event study is run across a grid of specifications — event windows x expected-
return models x surprise subsets — so the family-wise / false-discovery error must
be controlled honestly across the WHOLE grid, not cherry-picked from the single
most significant cell.

Two corrections are provided:

- **Benjamini-Hochberg (1995)** step-up FDR control; and
- **Romano-Wolf (2005)** stepdown, which controls the family-wise error rate
  using a bootstrap of the joint null and is more powerful than Bonferroni under
  dependence (the spec cells are correlated).

HONEST ``n_tests``: the correction counts EVERY cell actually evaluated. A
spec is only declared significant if it survives the correction across the full
grid; this is an input to the PURE verdict.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fedcausal._constants import DEFAULT_ALPHA
from fedcausal._exceptions import ValidationError
from fedcausal._validation import validate_alpha

__all__ = [
    "MultipleTestingResult",
    "benjamini_hochberg",
    "romano_wolf",
]


@dataclass(frozen=True, slots=True)
class MultipleTestingResult:
    """The outcome of a multiple-testing correction over the spec grid.

    Attributes
    ----------
    method:
        ``"benjamini_hochberg"`` or ``"romano_wolf"``.
    n_tests:
        The honest total number of specifications tested (the full grid size).
    adjusted_pvalues:
        The corrected p-values, one per spec, in input order.
    rejected:
        Boolean mask of specs declared significant after correction.
    any_survives:
        Whether ANY spec survives the correction (an input to the verdict).
    alpha:
        The significance level the correction targeted.
    """

    method: str
    n_tests: int
    adjusted_pvalues: np.ndarray
    rejected: np.ndarray
    any_survives: bool
    alpha: float

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable summary ``dict`` (lists, not arrays)."""
        payload = asdict(self)
        payload["adjusted_pvalues"] = [float(x) for x in self.adjusted_pvalues]
        payload["rejected"] = [bool(x) for x in self.rejected]
        payload["any_survives"] = bool(self.any_survives)
        payload["n_tests"] = int(self.n_tests)
        payload["alpha"] = float(self.alpha)
        return payload


def _validate_pvalues(pvalues: np.ndarray) -> NDArray[np.float64]:
    """Coerce ``pvalues`` to a finite 1-D float array inside ``[0, 1]``."""
    arr = np.asarray(pvalues, dtype=np.float64)
    if arr.ndim != 1:
        raise ValidationError(f"pvalues must be 1-dimensional, got ndim={arr.ndim}.")
    if arr.size == 0:
        raise ValidationError("pvalues must be non-empty.")
    if not bool(np.isfinite(arr).all()):
        raise ValidationError("pvalues must all be finite.")
    if bool((arr < 0.0).any()) or bool((arr > 1.0).any()):
        raise ValidationError("pvalues must lie in [0, 1].")
    return arr


def benjamini_hochberg(
    pvalues: np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> MultipleTestingResult:
    """Benjamini-Hochberg step-up FDR correction over the spec grid.

    The BH adjusted p-values are the standard step-up enforcement: order the raw
    p-values ascending, scale each by ``n / rank``, then take the running minimum
    from the largest rank downward (so adjusted p-values are monotone in the raw
    order) and clip to ``[0, 1]``. A spec is rejected when its adjusted p-value is
    ``<= alpha``, which is equivalent to the classic ``p_(i) <= (i / n) * alpha``
    step-up rule.

    Parameters
    ----------
    pvalues:
        Raw per-spec p-values (one per grid cell).
    alpha:
        Target false-discovery rate.

    Returns
    -------
    MultipleTestingResult
        Adjusted p-values, the rejection mask, and ``any_survives`` with an honest
        ``n_tests`` equal to ``len(pvalues)``.

    Raises
    ------
    ValidationError
        If ``pvalues`` is empty or contains values outside ``[0, 1]``.
    """
    alpha = validate_alpha(alpha)
    raw = _validate_pvalues(pvalues)
    n = raw.size

    order = np.argsort(raw, kind="stable")
    ranks = np.arange(1, n + 1, dtype=np.float64)
    scaled = raw[order] * n / ranks
    # Step-up enforcement: cumulative minimum from the largest rank down.
    adjusted_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    adjusted = np.empty(n, dtype=np.float64)
    adjusted[order] = adjusted_sorted
    rejected = adjusted <= alpha

    return MultipleTestingResult(
        method="benjamini_hochberg",
        n_tests=int(n),
        adjusted_pvalues=adjusted,
        rejected=rejected,
        any_survives=bool(rejected.any()),
        alpha=float(alpha),
    )


def _validate_romano_wolf_inputs(
    statistics: np.ndarray,
    null_distributions: np.ndarray,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Coerce and shape-check the Romano-Wolf observed stats and null draws."""
    stats_arr = np.asarray(statistics, dtype=np.float64)
    null_arr = np.asarray(null_distributions, dtype=np.float64)
    if stats_arr.ndim != 1:
        raise ValidationError(f"statistics must be 1-dimensional, got ndim={stats_arr.ndim}.")
    if stats_arr.size == 0:
        raise ValidationError("statistics must be non-empty.")
    if null_arr.ndim != 2:
        raise ValidationError(
            f"null_distributions must be 2-dimensional (n_specs, n_resamples), "
            f"got ndim={null_arr.ndim}."
        )
    if null_arr.shape[0] != stats_arr.size:
        raise ValidationError(
            f"null_distributions must have one row per spec: expected "
            f"{stats_arr.size} rows, got {null_arr.shape[0]}."
        )
    if null_arr.shape[1] < 1:
        raise ValidationError("null_distributions must have at least one resample column.")
    if not bool(np.isfinite(stats_arr).all()) or not bool(np.isfinite(null_arr).all()):
        raise ValidationError("statistics and null_distributions must be finite.")
    return stats_arr, null_arr


def romano_wolf(
    statistics: np.ndarray,
    null_distributions: np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> MultipleTestingResult:
    """Romano-Wolf (2005) stepdown family-wise-error correction (bootstrap null).

    Uses a bootstrap of the joint null (e.g. the per-spec placebo distributions)
    to step down through the ordered statistics, controlling the family-wise error
    rate under dependence between the (correlated) spec cells.

    The procedure works on TWO-SIDED magnitudes. Each spec's null draws are
    mean-centred so they represent the spec's sampling variation under the joint
    null. Specs are ordered by ``|statistic|`` descending. Starting from the most
    extreme spec, the stepdown p-value is the joint-null tail probability of the
    maximum centred ``|null|`` over the *still-active* specs exceeding the observed
    ``|statistic|``; successive stepdown p-values are enforced monotone
    non-decreasing. A spec is rejected when its stepdown p-value is ``<= alpha``,
    and (as in any stepdown) once a spec fails, every less-extreme spec also fails.

    Parameters
    ----------
    statistics:
        The observed per-spec test statistics (one per grid cell).
    null_distributions:
        A ``(n_specs, n_resamples)`` array of joint-null draws for each spec
        (e.g. the placebo CARs per spec), used to derive the stepdown thresholds.
    alpha:
        Target family-wise error rate.

    Returns
    -------
    MultipleTestingResult
        Adjusted p-values, the rejection mask, and ``any_survives`` with an honest
        ``n_tests`` equal to ``len(statistics)``.

    Raises
    ------
    ValidationError
        If the inputs are misshaped or empty.
    """
    alpha = validate_alpha(alpha)
    stats_arr, null_arr = _validate_romano_wolf_inputs(statistics, null_distributions)
    n_specs = stats_arr.size

    abs_stats = np.abs(stats_arr)
    # Centre each spec's null so the draws describe variation under the joint null.
    centred = np.abs(null_arr - null_arr.mean(axis=1, keepdims=True))

    # Order specs from most to least extreme observed statistic.
    order = np.argsort(abs_stats, kind="stable")[::-1]

    adjusted_in_order = np.empty(n_specs, dtype=np.float64)
    running_max = 0.0
    active = order.copy()
    for step, spec_idx in enumerate(order):
        # The joint-null maximum over the still-active (this and less-extreme) specs.
        active = order[step:]
        max_null = centred[active].max(axis=0)
        # Two-sided stepdown tail probability of |stat| under the joint null.
        tail = float(np.mean(max_null >= abs_stats[spec_idx]))
        # Enforce monotone non-decreasing stepdown p-values.
        running_max = max(running_max, tail)
        adjusted_in_order[step] = min(running_max, 1.0)

    adjusted = np.empty(n_specs, dtype=np.float64)
    adjusted[order] = adjusted_in_order
    rejected = adjusted <= alpha

    return MultipleTestingResult(
        method="romano_wolf",
        n_tests=int(n_specs),
        adjusted_pvalues=adjusted,
        rejected=rejected,
        any_survives=bool(rejected.any()),
        alpha=float(alpha),
    )
