"""Non-circular feature construction and the leakage / split guards.

Builds the neural-net feature matrix from the canonical option panel and enforces
the project's two correctness invariants:

1. **No target-in-features.** The label is the option PRICE. Implied volatility is
   a one-to-one Black-Scholes inversion of that price, so any IV-derived column is
   the label in disguise and must NEVER reach the model. The allowed features are
   moneyness ``K/S``, time-to-expiry ``T``, the rate ``r``, the dividend yield
   ``q``, and a *realized/historical* volatility (estimated from the underlying's
   own past returns, NOT from option prices). :func:`assert_no_leakage` is a hard
   guard called before fitting.

2. **No future-in-train.** Real chains are split by QUOTE DATE with an embargo
   (a group split), never randomly — adjacent strikes/expiries on the same day are
   near-duplicates, so a random split would leak. :func:`quote_date_group_split`
   returns index masks whose test fold is strictly in the future of its train fold.

The ``StandardScaler`` that standardizes these features is fit on the TRAIN fold
only; that lives in the model Pipeline (:mod:`nnvsbs.models.mlp`), not here.

``numpy`` is a hard dependency; ``pandas`` (the ``[data]`` extra) is imported
lazily. Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import NDArray

from nnvsbs._exceptions import InsufficientDataError, LeakageError, ValidationError
from nnvsbs._typing import FloatArray

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

#: A 1-D array of POSITIONAL row indices (the leakage-free split's output dtype).
#: The brief's stub typed the split as ``FloatArray``, but index arrays must be
#: integer to index rows, so the split returns this ``np.intp`` array instead.
IntpArray: TypeAlias = NDArray[np.intp]

#: The ordered, allowed NN feature columns. NOTE: ``sigma`` here is a
#: *realized/historical* volatility from the underlying's own returns — it is NOT
#: the option-implied vol and is NOT a function of the target price.
FEATURE_COLUMNS: tuple[str, ...] = (
    "moneyness",  # K / S
    "T",  # time-to-expiry, years
    "r",  # risk-free rate
    "q",  # dividend yield
    "realized_vol",  # historical vol of the underlying (NOT implied vol)
)

#: Columns that MUST NOT appear in the feature matrix when the label is price.
#: Any of these reaching the model is a leak (IV inverts the target price).
FORBIDDEN_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {"iv", "implied_vol", "implied_volatility", "sigma_implied", "price"}
)


def build_features(
    chain: pd.DataFrame,
    *,
    realized_vol: float | None = None,
) -> pd.DataFrame:
    """Build the non-circular NN feature matrix from an option panel.

    Derives moneyness ``K/S`` and carries ``T``, ``r``, ``q`` and a realized vol
    of the underlying into the ordered :data:`FEATURE_COLUMNS`. On synthetic data
    the realized vol defaults to the chain's constant ``sigma`` (it is the
    underlying's vol, not an option inversion); on real data it is estimated
    upstream from the underlying's return history and passed in.

    Parameters
    ----------
    chain:
        The canonical option panel (:data:`nnvsbs.data.synthetic.CHAIN_COLUMNS`).
    realized_vol:
        Optional realized/historical volatility of the underlying. ``None`` =>
        derive it from the chain's own (constant) ``sigma`` column on synthetic
        data.

    Returns
    -------
    pandas.DataFrame
        A frame with exactly :data:`FEATURE_COLUMNS`, one row per contract, in the
        same row order as ``chain``.

    Raises
    ------
    ValidationError
        If required input columns are missing or contain non-finite values.
    """
    import pandas as pd

    if not isinstance(chain, pd.DataFrame):
        raise ValidationError("build_features expects a pandas DataFrame.")

    required = ("S", "K", "T", "r", "q")
    missing = [col for col in required if col not in chain.columns]
    if missing:
        raise ValidationError(f"chain is missing required column(s): {missing}.")
    if chain.shape[0] == 0:
        raise ValidationError("chain must have at least one row.")

    spot = chain["S"].to_numpy(dtype="float64")
    strike = chain["K"].to_numpy(dtype="float64")
    expiry = chain["T"].to_numpy(dtype="float64")
    rate = chain["r"].to_numpy(dtype="float64")
    div = chain["q"].to_numpy(dtype="float64")

    if bool((spot <= 0.0).any()):
        raise ValidationError("S (spot) must be strictly positive.")
    for arr, name in ((spot, "S"), (strike, "K"), (expiry, "T"), (rate, "r"), (div, "q")):
        if not bool(np.isfinite(arr).all()):
            raise ValidationError(f"{name} contains non-finite values.")

    # Realized vol: the UNDERLYING's own volatility — never an option inversion.
    # On real chains it is passed in (estimated from the underlying's return
    # history); on synthetic data it defaults to the chain's constant ``sigma``.
    if realized_vol is None:
        if "sigma" not in chain.columns:
            raise ValidationError(
                "realized_vol is None and the chain has no 'sigma' column to derive it from."
            )
        realized = chain["sigma"].to_numpy(dtype="float64")
        if not bool(np.isfinite(realized).all()):
            raise ValidationError("sigma contains non-finite values.")
    else:
        if not np.isfinite(realized_vol):
            raise ValidationError(f"realized_vol must be finite, got {realized_vol}.")
        realized = np.full(spot.shape[0], float(realized_vol), dtype="float64")

    features = pd.DataFrame(
        {
            "moneyness": strike / spot,
            "T": expiry,
            "r": rate,
            "q": div,
            "realized_vol": realized,
        },
        index=chain.index,
    )[list(FEATURE_COLUMNS)]
    return features


def assert_no_leakage(features: pd.DataFrame) -> None:
    """Assert the feature matrix contains no target-derived (IV/price) column.

    The hard guard for invariant (1): raises if any column name in ``features``
    matches :data:`FORBIDDEN_FEATURE_COLUMNS` (case-insensitively) — i.e. an
    implied-vol or raw-price column has leaked into the model inputs. Called
    immediately before fitting / serving.

    Parameters
    ----------
    features:
        The candidate NN feature matrix.

    Raises
    ------
    LeakageError
        If a forbidden (target-derived) column is present.
    """
    present = {str(col).lower() for col in features.columns}
    leaked = sorted(present & FORBIDDEN_FEATURE_COLUMNS)
    if leaked:
        raise LeakageError(
            "target-derived column(s) leaked into the NN feature matrix: "
            f"{leaked}. Implied vol / price invert the label and must never be features."
        )


def quote_date_group_split(
    quote_dates: pd.Series,
    *,
    train_frac: float = 0.7,
    embargo: int = 1,
) -> tuple[IntpArray, IntpArray]:
    """Split contract indices by QUOTE DATE with an embargo (no future-in-train).

    Implements invariant (2): sorts the unique quote dates, assigns the earliest
    ``train_frac`` of *dates* to train and the rest to test, drops ``embargo``
    quote dates straddling the boundary, and returns the corresponding row-index
    masks. The test fold is therefore strictly later than the train fold — never a
    random split, since same-day strikes/expiries are near-duplicates.

    Parameters
    ----------
    quote_dates:
        The per-contract ``quote_date`` column (one entry per row of the chain).
    train_frac:
        Fraction of *unique quote dates* assigned to train (default ``0.7``).
    embargo:
        Number of quote dates to drop at the train/test boundary (default ``1``).

    Returns
    -------
    tuple[IntpArray, IntpArray]
        ``(train_idx, test_idx)`` POSITIONAL integer index arrays into the chain's
        rows (dtype ``np.intp``), with
        ``max(quote_date[train_idx]) < min(quote_date[test_idx])``.

    Raises
    ------
    ValidationError
        If ``train_frac`` is not in ``(0, 1)`` or ``embargo`` is negative.
    InsufficientDataError
        If there are too few unique quote dates to form both folds after the
        embargo.
    """
    import pandas as pd

    if not 0.0 < train_frac < 1.0:
        raise ValidationError(f"train_frac must be in (0, 1), got {train_frac}.")
    if embargo < 0:
        raise ValidationError(f"embargo must be non-negative, got {embargo}.")

    dates = pd.Series(quote_dates).reset_index(drop=True)
    if dates.empty:
        raise InsufficientDataError("quote_dates must be non-empty.")

    unique_dates = np.sort(np.asarray(pd.unique(dates)))
    n_dates = int(unique_dates.shape[0])
    # Need at least one train date, ``embargo`` boundary dates, and one test date.
    if n_dates < embargo + 2:
        raise InsufficientDataError(
            f"need at least {embargo + 2} unique quote dates to form train/test folds "
            f"with embargo={embargo}, got {n_dates}."
        )

    # Earliest ``train_frac`` of DATES (not rows) form the train fold; the rest are
    # candidate test dates. Clamp so both folds keep at least one date.
    n_train_dates = int(np.floor(train_frac * n_dates))
    n_train_dates = max(1, min(n_train_dates, n_dates - embargo - 1))

    train_dates = unique_dates[:n_train_dates]
    # Drop ``embargo`` dates straddling the boundary; the remainder is the test fold.
    test_dates = unique_dates[n_train_dates + embargo :]
    if test_dates.shape[0] == 0:
        raise InsufficientDataError(
            "embargo consumed every candidate test date; reduce embargo or train_frac."
        )

    positions = np.arange(dates.shape[0], dtype=np.intp)
    date_values = dates.to_numpy()
    train_mask = np.isin(date_values, train_dates)
    test_mask = np.isin(date_values, test_dates)

    train_idx = positions[train_mask]
    test_idx = positions[test_mask]
    if train_idx.shape[0] == 0 or test_idx.shape[0] == 0:
        raise InsufficientDataError("quote-date split produced an empty train or test fold.")

    return train_idx, test_idx


def _realized_vol_from_returns(returns: FloatArray, *, periods_per_year: int = 252) -> float:
    """Annualized realized volatility from a series of underlying log-returns.

    A helper for the real-chain path: ``std(returns) * sqrt(periods_per_year)``.
    This is a function of the UNDERLYING's history only — never of option prices —
    so it is leakage-safe as a feature.

    Parameters
    ----------
    returns:
        The underlying's periodic (log-)returns.
    periods_per_year:
        Annualization factor (default ``252`` trading days).

    Returns
    -------
    float
        The annualized realized volatility.

    Raises
    ------
    ValidationError
        If ``returns`` is empty/non-finite or ``periods_per_year`` is non-positive.
    """
    arr = np.asarray(returns, dtype="float64").ravel()
    if arr.size == 0:
        raise ValidationError("returns must be non-empty.")
    if not bool(np.isfinite(arr).all()):
        raise ValidationError("returns contains non-finite values.")
    if periods_per_year <= 0:
        raise ValidationError(f"periods_per_year must be positive, got {periods_per_year}.")
    # Sample std (ddof=1) when we have >1 observation; ddof=0 for a single point.
    ddof = 1 if arr.size > 1 else 0
    return float(np.std(arr, ddof=ddof) * np.sqrt(periods_per_year))
