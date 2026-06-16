"""Project-wide constants.

Single source of truth for numerical tolerances, the canonical label/status
vocabularies, and default modelling knobs so that no magic value is duplicated
across modules. Importing this module has no side effects.
"""

from __future__ import annotations

from typing import Final

# quantcore-candidate: mirrors risk-metrics:src/riskmetrics/_constants.py

#: Small positive floor used to clip probabilities away from ``{0, 1}`` before
#: taking a logit/log-loss, and to guard divisions. Chosen well above float64
#: round-off but far below any economically meaningful probability.
EPS: Final[float] = 1e-12

#: ``loan_status`` values that resolve to a positive label (1 = default).
DEFAULT_STATUSES: Final[tuple[str, ...]] = (
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
)

#: ``loan_status`` values that resolve to a negative label (0 = fully paid).
PAID_STATUSES: Final[tuple[str, ...]] = (
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
)

#: ``loan_status`` values for in-progress loans whose outcome is not yet known.
#: These rows are EXCLUDED from the labelled training/eval set (label hygiene).
IN_PROGRESS_STATUSES: Final[tuple[str, ...]] = (
    "Current",
    "Late (16-30 days)",
    "Late (31-120 days)",
    "In Grace Period",
    "Issued",
)

#: The two LendingClub loan terms, as the raw string values seen in the CSV.
VALID_TERMS: Final[tuple[str, ...]] = ("36 months", "60 months")

#: The seven LendingClub credit grades, best to worst.
VALID_GRADES: Final[tuple[str, ...]] = ("A", "B", "C", "D", "E", "F", "G")

#: Number of risk buckets used when reporting a calibrated ``risk_decile``.
N_RISK_DECILES: Final[int] = 10
