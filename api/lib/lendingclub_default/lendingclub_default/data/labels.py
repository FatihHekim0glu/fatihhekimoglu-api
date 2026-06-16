"""Target-label construction with strict label hygiene.

The binary target is built from ``loan_status``:

- positive (1): a *resolved, defaulted* loan — ``loan_status`` in
  :data:`lendingclub_default._constants.DEFAULT_STATUSES`
  (``Charged Off`` / ``Default``);
- negative (0): a *resolved, repaid* loan — ``loan_status`` in
  :data:`lendingclub_default._constants.PAID_STATUSES` (``Fully Paid``).

In-progress loans (``Current``, ``Late ...``, ``In Grace Period``, ``Issued``)
have no known outcome and are **excluded** from the labelled set — including them
would silently bias the base rate and the calibration. This exclusion is the
documented label-hygiene rule and is unit-tested.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from lendingclub_default._constants import DEFAULT_STATUSES, PAID_STATUSES
from lendingclub_default._exceptions import ValidationError

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

# quantcore-candidate: new code (label construction); status vocab in _constants.

#: Resolved statuses mapped to the positive class (1 = default).
_POSITIVE: frozenset[str] = frozenset(DEFAULT_STATUSES)
#: Resolved statuses mapped to the negative class (0 = fully paid).
_NEGATIVE: frozenset[str] = frozenset(PAID_STATUSES)


@dataclass(frozen=True, slots=True)
class LabelResult:
    """Immutable result of building labels from a raw panel.

    Attributes
    ----------
    panel:
        The panel restricted to resolved loans (in-progress rows dropped),
        index-aligned with ``y``.
    y:
        The binary target Series (1 = default, 0 = fully paid), same index as
        ``panel``.
    base_rate:
        The marginal default rate over the resolved set (``y.mean()``).
    n_excluded:
        Count of in-progress rows dropped for label hygiene.
    """

    panel: pd.DataFrame
    y: pd.Series
    base_rate: float
    n_excluded: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable summary (omits the heavy frames)."""
        out = {k: v for k, v in asdict(self).items() if k not in {"panel", "y"}}
        out["n_resolved"] = int(self.panel.shape[0]) if self.panel is not None else 0
        return out


def build_labels(df: pd.DataFrame, *, status_col: str = "loan_status") -> LabelResult:
    """Build the binary default target, excluding in-progress loans.

    Parameters
    ----------
    df:
        A panel carrying a ``loan_status`` column.
    status_col:
        Name of the status column (default ``"loan_status"``).

    Returns
    -------
    LabelResult
        The resolved-only panel, its aligned binary target, the base rate, and
        the number of excluded in-progress rows.

    Raises
    ------
    ValidationError
        If ``status_col`` is absent, or no rows resolve to a usable label.
    """
    if status_col not in df.columns:
        raise ValidationError(f"build_labels: status column {status_col!r} not found in panel.")

    status = df[status_col].astype("string")
    is_pos = status.isin(_POSITIVE)
    is_neg = status.isin(_NEGATIVE)
    resolved_mask = (is_pos | is_neg).to_numpy()

    n_total = int(df.shape[0])
    n_excluded = int(n_total - resolved_mask.sum())

    panel = df.loc[resolved_mask].copy()
    if panel.shape[0] == 0:
        raise ValidationError(
            "build_labels: no rows resolve to a usable label "
            f"(all {n_total} row(s) are in-progress or have unknown status)."
        )

    y = is_pos.loc[resolved_mask].astype("int64")
    y.name = "default"
    base_rate = float(y.mean())

    return LabelResult(
        panel=panel,
        y=y,
        base_rate=base_rate,
        n_excluded=n_excluded,
        meta={"n_total": n_total, "n_positive": int(y.sum()), "status_col": status_col},
    )
