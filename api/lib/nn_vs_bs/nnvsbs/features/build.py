"""Non-circular feature construction and the leakage / split guards.

Builds the neural-net feature matrix from the canonical option panel and enforces
the project's two correctness invariants:

1. **No target-in-features.** The label is the option PRICE. Implied volatility is
   a one-to-one Black-Scholes inversion of that price, so any IV-derived column is
   the label in disguise and must NEVER reach the model. The allowed features are
   moneyness ``K/S``, time-to-expiry ``T``, the rate ``r``, the dividend yield
   ``q``, and a *realized/historical* volatility (estimated from the underlying's
   own past returns, NOT from option prices). The per-contract chain ``sigma`` (the
   option implied volatility — the SMILE) is the label in disguise and is NEVER a
   feature: it is an analysis OUTPUT only. :func:`assert_no_leakage` is a hard
   POSITIVE-ALLOWLIST guard called before fitting — it rejects any column not in
   :data:`FEATURE_COLUMNS`, so a leak cannot pass simply by being renamed.

2. **No future-in-train.** Real chains are split by QUOTE DATE with an embargo
   (a group split), never randomly — adjacent strikes/expiries on the same day are
   near-duplicates, so a random split would leak. :func:`quote_date_group_split`
   returns index masks whose test fold is strictly in the future of its train fold.

The realized volatility is ALWAYS a function of the underlying's own return
history (synthetic: a seeded geometric-Brownian-motion path at the chain's
constant level; real: the underlying's historical price returns), NEVER the
per-contract option implied vol. :func:`realized_vol_for_chain` is the single
entry point both the train and serve paths call to obtain that leak-free scalar.

The ``StandardScaler`` that standardizes these features is fit on the TRAIN fold
only; that lives in the model Pipeline (:mod:`nnvsbs.models.mlp`), not here.

``numpy`` is a hard dependency; ``pandas`` (the ``[data]`` extra) is imported
lazily. Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import NDArray

from nnvsbs._constants import PERIODS_PER_YEAR
from nnvsbs._exceptions import InsufficientDataError, LeakageError, ValidationError
from nnvsbs._rng import make_rng
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

#: The POSITIVE allowlist enforced by :func:`assert_no_leakage`. Only these column
#: names may appear in the NN feature matrix. A name-only *blocklist* is theatre —
#: an implied-vol column relabelled ``realized_vol`` would sail straight through it —
#: so the guard instead REJECTS any column that is not in this allowlist. (Identical
#: to :data:`FEATURE_COLUMNS`; kept as a separate frozenset to make the guard's
#: membership-test intent explicit and O(1).)
ALLOWED_FEATURE_COLUMNS: frozenset[str] = frozenset(FEATURE_COLUMNS)

#: A small, explicitly-named set of target-derived columns the guard calls out by
#: name in its error message when one slips in (purely for a clearer diagnostic —
#: the allowlist above is what actually enforces the invariant). ``sigma`` is the
#: per-contract option implied vol (the smile), which inverts the target price.
KNOWN_LEAKY_COLUMNS: frozenset[str] = frozenset(
    {"iv", "implied_vol", "implied_volatility", "sigma", "sigma_implied", "price"}
)


def build_features(
    chain: pd.DataFrame,
    *,
    realized_vol: float | None = None,
) -> pd.DataFrame:
    """Build the non-circular NN feature matrix from an option panel.

    Derives moneyness ``K/S`` and carries ``T``, ``r``, ``q`` and a realized vol
    of the UNDERLYING into the ordered :data:`FEATURE_COLUMNS`. The realized vol is
    NEVER the per-contract option implied vol (the chain ``sigma`` / smile, which is
    a Black-Scholes inversion of the target price): it is a single scalar estimated
    from the underlying's own return history. The caller (train / serve) computes it
    via :func:`realized_vol_for_chain` and passes it in.

    Parameters
    ----------
    chain:
        The canonical option panel (:data:`nnvsbs.data.synthetic.CHAIN_COLUMNS`).
    realized_vol:
        Realized/historical volatility of the UNDERLYING, as a single non-negative
        scalar applied to every row. ``None`` => derive it from a seeded GBM path at
        the chain's constant volatility level (the synthetic underlying), via
        :func:`realized_vol_for_chain`. The per-contract chain ``sigma`` (option IV)
        is NEVER used here — that would leak the target price.

    Returns
    -------
    pandas.DataFrame
        A frame with exactly :data:`FEATURE_COLUMNS`, one row per contract, in the
        same row order as ``chain``.

    Raises
    ------
    ValidationError
        If required input columns are missing or contain non-finite values, or if a
        supplied ``realized_vol`` is non-finite / negative.
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

    # Realized vol: the UNDERLYING's own volatility — a single scalar, never the
    # per-contract option implied vol (the chain ``sigma`` smile inverts the target
    # price). When the caller does not supply it we derive it from a seeded GBM path
    # at the chain's constant level (the synthetic underlying) — NOT from ``sigma``.
    rv = realized_vol_for_chain(chain) if realized_vol is None else float(realized_vol)
    if not np.isfinite(rv):
        raise ValidationError(f"realized_vol must be finite, got {rv}.")
    if rv < 0.0:
        raise ValidationError(f"realized_vol must be non-negative, got {rv}.")
    realized = np.full(spot.shape[0], rv, dtype="float64")

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
    """Assert the feature matrix contains ONLY the allowed non-circular columns.

    The hard guard for invariant (1), implemented as a POSITIVE ALLOWLIST rather
    than a name-only blocklist. A blocklist is theatre: an implied-vol column
    relabelled ``realized_vol`` (or any arbitrary extra column) would pass it. This
    guard instead REJECTS any column whose name is not in
    :data:`ALLOWED_FEATURE_COLUMNS`, so a leak cannot slip through by renaming, and
    it also requires every allowed column to be present (no silent drop). Called
    immediately before fitting / serving.

    Parameters
    ----------
    features:
        The candidate NN feature matrix.

    Raises
    ------
    LeakageError
        If any column is not in the allowlist (a possible target leak), or a
        required allowed column is missing.
    """
    present = [str(col) for col in features.columns]
    present_set = set(present)

    extras = sorted(present_set - ALLOWED_FEATURE_COLUMNS)
    if extras:
        # Name the ones we recognize as classic target leaks for a clearer message,
        # but reject on the allowlist regardless of name (renamed IV included).
        flagged = sorted(c for c in extras if c.lower() in KNOWN_LEAKY_COLUMNS)
        suffix = (
            f" (recognized target-derived column(s): {flagged})"
            if flagged
            else " (any non-allowlisted column may be a renamed target leak)"
        )
        raise LeakageError(
            "non-allowlisted column(s) reached the NN feature matrix: "
            f"{extras}. Only {sorted(ALLOWED_FEATURE_COLUMNS)} are permitted; implied "
            f"vol / price invert the label and must never be features{suffix}."
        )

    missing = sorted(ALLOWED_FEATURE_COLUMNS - present_set)
    if missing:
        raise LeakageError(
            f"feature matrix is missing required allowlisted column(s): {missing}. "
            f"The NN input must be exactly {sorted(ALLOWED_FEATURE_COLUMNS)}."
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


def realized_vol_for_chain(
    chain: pd.DataFrame,
    *,
    underlying_returns: FloatArray | None = None,
    periods_per_year: int = PERIODS_PER_YEAR,
    n_days: int = 252,
    seed: int = 7,
) -> float:
    """Return a single leak-free realized volatility of the UNDERLYING for a chain.

    This is the one entry point the train / serve paths use to obtain the
    ``realized_vol`` feature. It is ALWAYS a function of the underlying's own return
    history, NEVER of the per-contract option implied vol (the chain ``sigma`` smile,
    which inverts the target price):

    * If ``underlying_returns`` is supplied (the real-chain path, from the
      underlying's historical price series), the realized vol is estimated directly
      from those returns.
    * Otherwise (the synthetic path) a deterministic geometric-Brownian-motion
      return path is simulated at the chain's CONSTANT volatility *level* and the
      realized vol is estimated from that simulated underlying path. The level is
      read from the chain's constant ``sigma`` only as the GBM *parameter* of the
      synthetic underlying's dynamics — it is **not** fed per-contract and it is not
      an inversion of any option price (on synthetic data every contract shares one
      true vol, so this matches the data-generating process while staying a property
      of the underlying, not the option). The result is a single scalar with
      sampling noise, so the NN must RECOVER Black-Scholes from the geometry
      (moneyness, T, r, q) plus this noisy vol — never invert a per-contract IV.

    Parameters
    ----------
    chain:
        The canonical option panel (used only for its constant ``sigma`` *level* on
        the synthetic path; the per-contract values are never used as features).
    underlying_returns:
        Optional underlying periodic returns (the real-chain path). When given, the
        synthetic GBM simulation is bypassed entirely.
    periods_per_year:
        Annualization factor (default :data:`nnvsbs._constants.PERIODS_PER_YEAR`).
    n_days:
        Length of the simulated synthetic underlying return path (default ``252``).
    seed:
        Master seed for the synthetic underlying simulation (deterministic).

    Returns
    -------
    float
        The annualized realized volatility of the underlying (a single scalar).

    Raises
    ------
    ValidationError
        If neither real returns nor a usable constant ``sigma`` level is available,
        or an input is non-finite.
    """
    if underlying_returns is not None:
        return _realized_vol_from_underlying_returns(
            np.asarray(underlying_returns, dtype="float64"),
            periods_per_year=periods_per_year,
        )

    if "sigma" not in chain.columns:
        raise ValidationError(
            "cannot derive realized_vol: no underlying_returns supplied and the chain "
            "has no constant 'sigma' level to simulate the synthetic underlying from."
        )
    level = np.asarray(chain["sigma"].to_numpy(dtype="float64"))
    if level.size == 0 or not bool(np.isfinite(level).all()):
        raise ValidationError("chain 'sigma' level is empty or non-finite.")
    vol_level = float(np.median(level))
    if vol_level <= 0.0:
        raise ValidationError(f"synthetic vol level must be positive, got {vol_level}.")

    returns = _simulate_underlying_returns(
        vol_level, n_days=n_days, periods_per_year=periods_per_year, seed=seed
    )
    return _realized_vol_from_underlying_returns(returns, periods_per_year=periods_per_year)


def _simulate_underlying_returns(
    vol_level: float,
    *,
    n_days: int,
    periods_per_year: int,
    seed: int,
) -> FloatArray:
    """Simulate a seeded daily GBM return path for the synthetic underlying.

    The synthetic chain has no real price history, so the underlying's realized vol
    is estimated from a deterministic geometric-Brownian-motion path whose per-period
    volatility is ``vol_level / sqrt(periods_per_year)``. This keeps ``realized_vol``
    a property of the (simulated) UNDERLYING — never an inversion of any option
    price — while matching the constant-vol data-generating process up to sampling
    noise.

    Parameters
    ----------
    vol_level:
        The annualized volatility level of the synthetic underlying (its GBM sigma).
    n_days:
        Number of daily returns to simulate.
    periods_per_year:
        Annualization factor used to scale the per-period vol.
    seed:
        Deterministic seed for the path.

    Returns
    -------
    FloatArray
        A ``(n_days,)`` array of simulated daily log-returns.
    """
    if n_days <= 0:
        raise ValidationError(f"n_days must be positive, got {n_days}.")
    if periods_per_year <= 0:
        raise ValidationError(f"periods_per_year must be positive, got {periods_per_year}.")
    rng = make_rng(seed)
    daily_vol = vol_level / np.sqrt(periods_per_year)
    # Zero-drift GBM log-returns; only the dispersion (vol) matters for realized vol.
    returns: FloatArray = rng.normal(0.0, daily_vol, size=int(n_days))
    return returns


def _realized_vol_from_underlying_returns(
    returns: FloatArray, *, periods_per_year: int = PERIODS_PER_YEAR
) -> float:
    """Annualized realized volatility from a series of underlying returns.

    The leak-safe estimator both paths share: ``std(returns) * sqrt(periods_per_year)``.
    This is a function of the UNDERLYING's history only — never of option prices —
    so it is safe to use as a feature.

    Parameters
    ----------
    returns:
        The underlying's periodic (log-)returns.
    periods_per_year:
        Annualization factor (default :data:`nnvsbs._constants.PERIODS_PER_YEAR`).

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


#: Backwards-compatible alias: the old private estimator name still used by tests.
_realized_vol_from_returns = _realized_vol_from_underlying_returns
