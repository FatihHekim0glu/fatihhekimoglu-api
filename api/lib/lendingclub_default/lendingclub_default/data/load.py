"""Data loading: real Kaggle CSV when given, else the synthetic generator.

:func:`load_panel` is the single entry point used by training and the CLI. When a
path to the real LendingClub accepted-loans CSV is supplied it reads and dtype-
coerces it; when omitted it falls back to :func:`lendingclub_default.data.synthetic.generate_synthetic_panel`.
The same downstream pipeline (leakage drop -> labels -> temporal split ->
calibration) runs on either source, so the synthetic path is a faithful stand-in.

Importing this module has no side effects (no file is read at import time).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

from lendingclub_default._exceptions import ValidationError

if TYPE_CHECKING:
    from lendingclub_default.data.synthetic import SyntheticConfig

# quantcore-candidate: new code (LC CSV loader); dtype coercion via _validation.

#: The application-time columns the loader guarantees to surface (origination
#: features the model is allowed to use). Used both to validate a real CSV and to
#: document the contract the synthetic generator must satisfy.
APPLICATION_COLUMNS: Final[tuple[str, ...]] = (
    "loan_amnt",
    "term",
    "int_rate",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "dti",
    "fico_range_low",
    "fico_range_high",
    "revol_util",
    "open_acc",
    "pub_rec",
    "purpose",
    "addr_state",
    "installment",
    "verification_status",
    "issue_d",
)


def load_panel(
    data_path: Path | str | None = None,
    *,
    config: SyntheticConfig | None = None,
) -> pd.DataFrame:
    """Load a loan-application panel from a CSV path, or generate one synthetically.

    Parameters
    ----------
    data_path:
        Path to the real LendingClub accepted-loans CSV. If ``None``, the
        synthetic generator is used instead (the reproducible default).
    config:
        Synthetic-generator configuration, used only when ``data_path is None``.

    Returns
    -------
    pandas.DataFrame
        The loaded (or generated) panel with at least :data:`APPLICATION_COLUMNS`
        plus ``loan_status``; a real CSV additionally carries post-funding columns
        that are dropped downstream by
        :func:`lendingclub_default.data.leakage.drop_leakage`.

    Raises
    ------
    ValidationError
        If a supplied CSV is missing required application-time columns.
    """
    if data_path is None:
        # LAZY import: keep the synthetic generator out of the import graph until
        # it is actually needed (import purity).
        from lendingclub_default.data.synthetic import generate_synthetic_panel

        return generate_synthetic_panel(config)

    path = Path(data_path)
    if not path.exists():
        raise ValidationError(f"load_panel: data file does not exist: {path}.")

    raw = pd.read_csv(path, low_memory=False)
    missing = [col for col in APPLICATION_COLUMNS if col not in raw.columns]
    if missing:
        raise ValidationError(
            f"load_panel: CSV {path.name} is missing required application-time "
            f"column(s): {missing}."
        )
    return coerce_dtypes(raw)


def _parse_percent(series: pd.Series) -> pd.Series:
    """Coerce a percentage column (``"13.5%"`` or ``13.5``) to a float64 Series."""
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        cleaned = series.astype("string").str.replace("%", "", regex=False).str.strip()
        return pd.to_numeric(cleaned, errors="coerce").astype("float64")
    return series.astype("float64")


def _parse_emp_length(series: pd.Series) -> pd.Series:
    """Coerce ``emp_length`` (``"< 1 year"``, ``"10+ years"``, ``"3 years"``) to years.

    ``"< 1 year"`` -> 0, ``"10+ years"`` -> 10, ``"n years"`` -> n; already-numeric
    input passes through. Unparseable values become NaN (imputed downstream).
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("float64")
    text = series.astype("string").str.strip()
    text = text.str.replace("< 1 year", "0", regex=False)
    text = text.str.replace("10+ years", "10", regex=False)
    extracted = text.str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(extracted, errors="coerce").astype("float64")


def coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce raw LC columns to model-ready dtypes (a copy; input untouched).

    Parses percentage strings (``int_rate``, ``revol_util`` such as ``"13.5%"``)
    to floats, ``term`` to its canonical ``"36 months"``/``"60 months"`` string,
    ``emp_length`` to a numeric year count, and ``issue_d`` to a sortable
    period/datetime used by the temporal split.

    Parameters
    ----------
    df:
        The raw panel as read from CSV (or produced synthetically).

    Returns
    -------
    pandas.DataFrame
        A copy with coerced dtypes.
    """
    out = df.copy()

    # Percentage strings -> float (only when the column is present).
    for col in ("int_rate", "revol_util"):
        if col in out.columns:
            out[col] = _parse_percent(out[col])

    # term -> canonical "36 months" / "60 months" string.
    if "term" in out.columns:
        term = out["term"].astype("string").str.strip()
        digits = term.str.extract(r"(\d+)", expand=False)
        out["term"] = np.where(digits == "60", "60 months", "36 months").astype(object)

    # emp_length -> numeric year count.
    if "emp_length" in out.columns:
        out["emp_length"] = _parse_emp_length(out["emp_length"])

    # Numeric application-time columns -> float64 (coerce stray strings to NaN).
    for col in (
        "loan_amnt",
        "annual_inc",
        "dti",
        "fico_range_low",
        "fico_range_high",
        "open_acc",
        "pub_rec",
        "installment",
        "funded_amnt",
    ):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    return out
