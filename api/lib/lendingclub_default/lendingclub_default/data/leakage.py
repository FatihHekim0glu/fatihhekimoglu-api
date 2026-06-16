"""Post-funding leakage allowlist and the canonical ``drop_leakage`` filter.

Post-funding leakage is THE trap of LendingClub default modelling. The accepted-
loans CSV ships dozens of columns that are only populated *after* a loan is
funded — payments received, recoveries, charge-off bookkeeping, hardship and
settlement records. Every one of these encodes (directly or indirectly) the loan
outcome we are trying to predict, so a model that sees them scores a fraudulent
~0.99 AUC. :data:`LEAKAGE_COLS` is the frozen, reviewed allowlist of every such
column from the LendingClub data dictionary; :func:`drop_leakage` removes any of
them that are present so only origination-time features reach the model.

This module is pure reference data plus one pure function. Importing it has no
side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from lendingclub_default._exceptions import LeakageError

if TYPE_CHECKING:
    import pandas as pd

# quantcore-candidate: new code (leakage allowlist); reference = LendingClub
# "LCDataDictionary" post-funding fields.

#: The canonical, frozen allowlist of post-funding (outcome-leaking) columns.
#: Sourced from the LendingClub data dictionary. Anything here is NOT known at
#: application/origination time and MUST be dropped before feature engineering.
#: Kept as a ``frozenset`` so it cannot be mutated at runtime; membership checks
#: are O(1). Column matching is case-insensitive in :func:`drop_leakage`.
LEAKAGE_COLS: Final[frozenset[str]] = frozenset(
    {
        # --- payments received (cash flows realised after funding) ---------- #
        "total_pymnt",
        "total_pymnt_inv",
        "total_rec_prncp",
        "total_rec_int",
        "total_rec_late_fee",
        "last_pymnt_d",
        "last_pymnt_amnt",
        "next_pymnt_d",
        "out_prncp",
        "out_prncp_inv",
        "pymnt_plan",
        # --- recoveries & charge-off bookkeeping ---------------------------- #
        "recoveries",
        "collection_recovery_fee",
        "collections_12_mths_ex_med",
        "chargeoff_within_12_mths",
        # --- borrower-status fields LendingClub REFRESHES post-funding ------- #
        # (current, not application-time, snapshots — they track the delinquency
        # the outcome is derived from, so they leak on the real `accepted.csv`
        # path even though they look like benign bureau features)
        "acc_now_delinq",
        "tot_coll_amt",
        "delinq_amnt",
        # --- funded amounts (only known once the loan is actually funded) ---- #
        "funded_amnt",
        "funded_amnt_inv",
        # --- the outcome label and its derivatives -------------------------- #
        "loan_status",
        "last_fico_range_high",
        "last_fico_range_low",
        "last_credit_pull_d",
        # --- hardship program fields (post-funding distress signals) -------- #
        "hardship_flag",
        "hardship_type",
        "hardship_reason",
        "hardship_status",
        "deferral_term",
        "hardship_amount",
        "hardship_start_date",
        "hardship_end_date",
        "payment_plan_start_date",
        "hardship_length",
        "hardship_dpd",
        "hardship_loan_status",
        "orig_projected_additional_accrued_interest",
        "hardship_payoff_balance_amount",
        "hardship_last_payment_amount",
        # --- debt settlement fields (post-funding) -------------------------- #
        "debt_settlement_flag",
        "debt_settlement_flag_date",
        "settlement_status",
        "settlement_date",
        "settlement_amount",
        "settlement_percentage",
        "settlement_term",
    }
)


def drop_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with every known post-funding column removed.

    Matching is case-insensitive: a column is dropped if its lower-cased name is
    in :data:`LEAKAGE_COLS`. The input is never mutated. Columns not present are
    silently ignored (the same allowlist is applied to the synthetic panel, a
    perturbed-schema fixture, and the real Kaggle CSV, which expose overlapping
    but not identical subsets of these fields).

    Parameters
    ----------
    df:
        A loan-application panel that may contain post-funding columns.

    Returns
    -------
    pandas.DataFrame
        A copy of ``df`` containing only non-leakage (origination-time) columns,
        with original column order preserved.
    """
    to_drop = [col for col in df.columns if str(col).lower() in LEAKAGE_COLS]
    # ``df.drop`` returns a copy (the input is never mutated) and preserves the
    # relative order of the surviving columns.
    return df.drop(columns=to_drop)


def assert_no_leakage(df: pd.DataFrame) -> None:
    """Raise if any :data:`LEAKAGE_COLS` member survives in ``df``.

    The hard backstop behind the leakage property test: call it on the final
    feature matrix to guarantee no outcome-leaking column reached the model.

    Parameters
    ----------
    df:
        The (post-preprocessing) feature matrix to audit.

    Raises
    ------
    LeakageError
        If one or more leakage columns are present.
    """
    survivors = sorted(str(col) for col in df.columns if str(col).lower() in LEAKAGE_COLS)
    if survivors:
        raise LeakageError(
            "post-funding leakage column(s) survived preprocessing: "
            f"{survivors}. These encode the loan outcome and must be dropped "
            "before the model sees them."
        )
