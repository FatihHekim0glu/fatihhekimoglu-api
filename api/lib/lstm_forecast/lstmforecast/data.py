"""Data generation and loading: synthetic random-walk prices + a CSV loader.

The DEFAULT and SHIPPED data path is a seeded *geometric random walk* — there is
no API key and no real market data here, by design. On a true random walk the
next-day log-return is unpredictable, so a properly-validated LSTM CANNOT beat a
persistence baseline; the honest NULL holds BY CONSTRUCTION, which is exactly
what makes the leakage-guard integration test meaningful (if the LSTM ever
"beats" persistence on random-walk data, leakage has re-entered).

The optional ``trend_plus_noise`` and ``pure_noise`` generators back property
tests. A thin :func:`load_prices` loader reads a real ``date,close`` CSV for the
``train --data <csv>`` retraining path; pandas/pyarrow live behind the ``data``
extra and are imported lazily.

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from lstmforecast._exceptions import ValidationError
from lstmforecast._rng import make_rng
from lstmforecast._validation import ensure_series

if TYPE_CHECKING:
    from pathlib import Path

#: Where a price series ultimately came from. Returned alongside data so callers
#: (and the API ``data_source`` field) can report provenance honestly.
DataSource = Literal["synthetic", "csv"]

# quantcore-candidate: mirrors hrp:data.py synthetic GBM fallback (here the
# random walk is the PRIMARY, shipped path — not a fallback).


def _business_index(n_obs: int, start: str = "2015-01-01") -> pd.DatetimeIndex:
    """Return an ``n_obs``-length business-day (Mon-Fri) index from ``start``."""
    return pd.date_range(start=start, periods=n_obs, freq="B")


def random_walk_prices(
    n_obs: int = 2000,
    *,
    seed: int = 7,
    s0: float = 100.0,
    sigma: float = 0.01,
    start: str = "2015-01-01",
) -> pd.Series:
    r"""Generate a seeded geometric random-walk price series (the shipped path).

    The log-price follows a driftless random walk
    :math:`\ln P_t = \ln P_{t-1} + \varepsilon_t`, with
    :math:`\varepsilon_t \sim \mathcal{N}(0, \sigma^2)` i.i.d. — so the next-day
    log-return :math:`r_{t+1}` is pure white noise and is UNPREDICTABLE from the
    past by construction. This is the data on which the honest NULL must hold: a
    leakage-free LSTM cannot beat persistence here.

    Parameters
    ----------
    n_obs:
        Number of price observations (``>= 2``).
    seed:
        Master RNG seed (drawn via :func:`lstmforecast._rng.make_rng`, never the
        numpy global state).
    s0:
        Strictly-positive starting price level.
    sigma:
        Per-step log-return standard deviation (``> 0``).
    start:
        First date of the business-day index.

    Returns
    -------
    pandas.Series
        A strictly-positive, time-indexed price series of length ``n_obs``.

    Raises
    ------
    ValidationError
        If ``n_obs < 2``, ``s0 <= 0``, or ``sigma <= 0``.
    """
    if n_obs < 2:
        raise ValidationError(f"random_walk_prices: n_obs must be >= 2, got {n_obs}.")
    if s0 <= 0.0:
        raise ValidationError(f"random_walk_prices: s0 must be > 0, got {s0}.")
    if sigma <= 0.0:
        raise ValidationError(f"random_walk_prices: sigma must be > 0, got {sigma}.")

    gen = make_rng(seed)
    shocks = gen.normal(loc=0.0, scale=sigma, size=n_obs)
    shocks[0] = 0.0  # anchor the first observation at s0
    log_prices = np.log(s0) + np.cumsum(shocks)
    prices = np.exp(log_prices)
    return pd.Series(prices, index=_business_index(n_obs, start), name="close", dtype="float64")


def trend_plus_noise_prices(
    n_obs: int = 2000,
    *,
    seed: int = 7,
    s0: float = 100.0,
    mu: float = 0.0003,
    sigma: float = 0.01,
    start: str = "2015-01-01",
) -> pd.Series:
    r"""Generate a deterministic-drift-plus-noise price series (property fixture).

    The log-price adds a constant per-step drift to the random walk
    (:math:`\ln P_t = \ln P_{t-1} + \mu + \varepsilon_t`). The drift makes the
    PRICE LEVEL trend (which is precisely why a price-level R² looks deceptively
    high — the debunked trap), yet the next-day RETURN is still
    ``mu + white noise`` and so remains near-unpredictable in return space.

    Parameters
    ----------
    n_obs:
        Number of price observations (``>= 2``).
    seed:
        Master RNG seed.
    s0:
        Strictly-positive starting price level.
    mu:
        Constant per-step log-return drift.
    sigma:
        Per-step idiosyncratic log-return standard deviation (``> 0``).
    start:
        First date of the business-day index.

    Returns
    -------
    pandas.Series
        A strictly-positive, time-indexed, upward-trending price series.

    Raises
    ------
    ValidationError
        If ``n_obs < 2``, ``s0 <= 0``, or ``sigma <= 0``.
    """
    if n_obs < 2:
        raise ValidationError(f"trend_plus_noise_prices: n_obs must be >= 2, got {n_obs}.")
    if s0 <= 0.0:
        raise ValidationError(f"trend_plus_noise_prices: s0 must be > 0, got {s0}.")
    if sigma <= 0.0:
        raise ValidationError(f"trend_plus_noise_prices: sigma must be > 0, got {sigma}.")

    gen = make_rng(seed)
    shocks = gen.normal(loc=0.0, scale=sigma, size=n_obs)
    # Add a constant per-step drift to every increment, then anchor the first
    # observation at s0 (no drift/shock applied to the starting bar).
    increments = mu + shocks
    increments[0] = 0.0
    log_prices = np.log(s0) + np.cumsum(increments)
    prices = np.exp(log_prices)
    return pd.Series(prices, index=_business_index(n_obs, start), name="close", dtype="float64")


def pure_noise_returns(
    n_obs: int = 2000,
    *,
    seed: int = 7,
    sigma: float = 0.01,
    start: str = "2015-01-01",
) -> pd.Series:
    """Generate an i.i.d. zero-mean Gaussian RETURN series (property fixture).

    The pure-noise null on which persistence (``r_hat = 0``) is provably the
    error-minimizing point forecast, used by the "persistence is the floor"
    property test.

    Parameters
    ----------
    n_obs:
        Number of return observations (``>= 1``).
    seed:
        Master RNG seed.
    sigma:
        Return standard deviation (``> 0``).
    start:
        First date of the business-day index.

    Returns
    -------
    pandas.Series
        A zero-mean i.i.d. Gaussian return series.

    Raises
    ------
    ValidationError
        If ``n_obs < 1`` or ``sigma <= 0``.
    """
    if n_obs < 1:
        raise ValidationError(f"pure_noise_returns: n_obs must be >= 1, got {n_obs}.")
    if sigma <= 0.0:
        raise ValidationError(f"pure_noise_returns: sigma must be > 0, got {sigma}.")

    gen = make_rng(seed)
    returns = gen.normal(loc=0.0, scale=sigma, size=n_obs)
    return pd.Series(returns, index=_business_index(n_obs, start), name="return", dtype="float64")


def to_log_returns(prices: pd.Series) -> pd.Series:
    r"""Convert a price series to next-step log-returns ``r_t = ln(P_t / P_{t-1})``.

    NO-LOOKAHEAD REQUIREMENT: returns are computed via ``np.log(prices).diff()``
    on the raw observed prices (NEVER forward-filled first — ffill-then-diff
    manufactures spurious zero returns across gaps and leaks information). The
    leading NaN row is dropped.

    Parameters
    ----------
    prices:
        A strictly-positive, time-indexed price series.

    Returns
    -------
    pandas.Series
        The log-return series with the leading NaN removed.

    Raises
    ------
    ValidationError
        If ``prices`` is empty or contains a non-positive value.
    """
    series = ensure_series(prices, name="prices", allow_nan=True)
    if bool((series <= 0.0).any()):
        raise ValidationError("to_log_returns: prices must be strictly positive.")

    # NO-LOOKAHEAD REQUIREMENT: difference the log of the raw observed prices.
    # Differencing on observed values never forward-fills before differencing, so
    # it does not manufacture spurious zero returns across gaps. The leading NaN
    # row produced by ``.diff()`` is dropped.
    log_prices = np.log(series.to_numpy(dtype="float64"))
    diffs = np.diff(log_prices)
    return pd.Series(diffs, index=series.index[1:], name="return", dtype="float64")


def load_prices(path: str | Path) -> tuple[pd.Series, DataSource]:
    """Load a real ``date,close`` CSV into a price series (the retrain path).

    Backs ``train --data <csv>``: reads a two-column (or ``date``-indexed) CSV of
    daily closes, parses the date index, sorts ascending, and validates the
    prices are strictly positive. pandas/pyarrow live behind the ``data`` extra
    and are imported lazily; importing this module never imports them.

    Parameters
    ----------
    path:
        Filesystem path to a CSV with a ``date`` column and a ``close`` column.

    Returns
    -------
    tuple[pandas.Series, DataSource]
        The loaded price series and the literal ``"csv"`` provenance tag.

    Raises
    ------
    ValidationError
        If the file is missing required columns, is empty, or has non-positive
        or non-monotonic-in-time prices.
    """
    # ``os.fspath`` accepts both ``str`` and ``pathlib.Path`` without importing
    # ``pathlib`` at module scope; the read itself uses the already-imported
    # pandas (the heavy parquet/pyarrow cache layer is not needed for a plain CSV).
    import os

    csv_path = os.fspath(path)
    if not os.path.isfile(csv_path):
        raise ValidationError(f"load_prices: file not found: {csv_path!r}.")

    try:
        frame = pd.read_csv(csv_path)
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        raise ValidationError(f"load_prices: could not parse CSV {csv_path!r}: {exc}.") from exc

    # Resolve the date and close columns case-insensitively.
    lower = {str(col).strip().lower(): col for col in frame.columns}
    if "close" not in lower:
        raise ValidationError(
            f"load_prices: CSV must have a 'close' column, found {list(frame.columns)}."
        )
    close_col = lower["close"]

    if "date" in lower:
        date_index = pd.to_datetime(frame[lower["date"]], errors="raise")
    else:
        # No explicit date column: treat the first column as the date index.
        date_index = pd.to_datetime(frame.iloc[:, 0], errors="raise")

    series = pd.Series(
        frame[close_col].to_numpy(dtype="float64"),
        index=pd.DatetimeIndex(date_index),
        name="close",
        dtype="float64",
    )

    if series.empty:
        raise ValidationError(f"load_prices: CSV {csv_path!r} has no rows.")
    # Sort ascending in time so downstream ``.shift(1)`` / windowing is valid.
    series = series.sort_index()
    if bool(series.isna().any()):
        raise ValidationError(f"load_prices: CSV {csv_path!r} contains missing close values.")
    if bool((series <= 0.0).any()):
        raise ValidationError(f"load_prices: CSV {csv_path!r} has non-positive close prices.")
    if series.index.has_duplicates:
        raise ValidationError(f"load_prices: CSV {csv_path!r} has duplicate dates.")

    return series, "csv"
