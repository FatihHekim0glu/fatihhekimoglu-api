"""Data loading for the PIT tone-spread panel: prices and forward returns.

The panel study needs point-in-time issuer prices to label each filing's forward
return. The default offline path is a deterministic synthetic panel; the real
data path uses the existing Polygon EOD provider (lazy ``httpx``) behind the
``data`` extra. ``compute_returns`` differences prices with ``fill_method=None``
(no forward-fill, no look-ahead) exactly as the HRP infra does.

Importing this module has no side effects.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

import pandas as pd

from edgar_nlp._exceptions import ValidationError
from edgar_nlp._typing import PricesLike
from edgar_nlp._validation import ensure_dataframe

# quantcore-candidate: mirrors hrp-portfolio:src/hrp/data.py (price fetch +
# pct_change(fill_method=None) returns), trimmed to the panel-study needs.

#: Where a price/return panel came from. Returned alongside data so callers (and
#: the API ``data_source`` field) can report provenance.
DataSource = Literal["polygon", "synthetic", "cache"]


def get_prices(
    tickers: list[str],
    start: date,
    end: date,
    *,
    source_pref: Literal["polygon", "synthetic", "auto"] = "auto",
    use_cache: bool = True,
) -> tuple[pd.DataFrame, DataSource]:
    """Fetch a wide panel of adjusted close prices with graceful fallback.

    Resolution order (``source_pref="auto"``): the real Polygon EOD provider
    (when a key is configured) and, on any failure, a deterministic synthetic
    panel so the library is usable offline and in CI.

    LAZY IMPORT: ``polygon``/``httpx`` (the ``data`` extra) are imported inside
    this function, never at module import time.

    Parameters
    ----------
    tickers:
        The issuer symbols to fetch.
    start, end:
        Inclusive date range (e.g. spanning all filing acceptance dates plus the
        forward-return horizon).
    source_pref:
        Preferred source. ``"auto"`` tries Polygon then synthetic; ``"polygon"``
        requires Polygon; ``"synthetic"`` forces the seeded synthetic panel.
    use_cache:
        Whether to read/write the parquet/diskcache cache.

    Returns
    -------
    tuple[pandas.DataFrame, DataSource]
        The price panel (rows = date, columns = ticker) and its provenance.

    Raises
    ------
    ValidationError
        If ``tickers`` is empty or ``end <= start``.
    """
    symbols = list(tickers)
    if len(symbols) == 0:
        raise ValidationError("get_prices: tickers must be non-empty.")
    if end <= start:
        raise ValidationError(f"get_prices: end ({end}) must be after start ({start}).")

    _ = use_cache  # cache wiring is intentionally a no-op in the lean offline build

    if source_pref == "synthetic":
        return _synthetic_prices(symbols, start, end), "synthetic"

    if source_pref in {"polygon", "auto"}:
        try:
            # LAZY import: keep the ``data`` extra off this module's import path.
            from edgar_nlp.data_providers.polygon import PolygonProvider

            frame = PolygonProvider().fetch(symbols, start, end)
            if frame.empty:  # pragma: no cover - defensive: provider returns rows
                raise ValidationError("get_prices: Polygon returned an empty panel.")
            return frame.astype("float64"), "polygon"
        except Exception:
            if source_pref == "polygon":
                raise
            # ``auto`` degrades to the deterministic synthetic panel (offline/CI).

    return _synthetic_prices(symbols, start, end), "synthetic"


def _synthetic_prices(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """Build a deterministic synthetic adjusted-close panel (offline fallback).

    Generates a seeded geometric-random-walk price path per ticker on business
    days between ``start`` and ``end``. The seed is derived from the date range so
    the same request always yields a byte-identical panel; no network is touched.
    """
    from edgar_nlp._rng import make_rng

    index = pd.bdate_range(start=start, end=end)
    if len(index) == 0:  # pragma: no cover - defensive: end > start guarantees rows
        index = pd.DatetimeIndex([pd.Timestamp(start)])

    seed = (start.toordinal() * 1_000_003 + end.toordinal()) & 0x7FFF_FFFF
    gen = make_rng(seed)

    columns: dict[str, pd.Series] = {}
    n = len(index)
    for ticker in tickers:
        # Small daily drift + idiosyncratic noise → a positive, smoothly varying
        # price path. The cumulative product keeps prices strictly positive so the
        # downstream pct-change / forward-return is well defined.
        shocks = gen.standard_normal(n) * 0.01 + 0.0002
        path = 100.0 * pd.Series(shocks, index=index).add(1.0).cumprod()
        columns[ticker] = path.astype("float64")

    frame = pd.DataFrame(columns)
    return frame.reindex(columns=tickers).astype("float64")


def compute_returns(prices: PricesLike) -> pd.DataFrame:
    r"""Convert a price panel to simple returns.

    NO-LOOKAHEAD REQUIREMENT: returns are computed with
    ``prices.pct_change(fill_method=None)`` — prices are NEVER forward-filled
    before differencing, because ffill-then-diff manufactures spurious zero
    returns across gaps and leaks information. The first (all-NaN) row is dropped.

    Parameters
    ----------
    prices:
        A wide panel of prices (rows = date, columns = issuer).

    Returns
    -------
    pandas.DataFrame
        Simple returns with the leading NaN row removed.

    Raises
    ------
    ValidationError
        If ``prices`` is malformed.
    """
    frame = ensure_dataframe(prices, name="prices", allow_nan=True)

    # NO-LOOKAHEAD REQUIREMENT: never forward-fill prices before differencing.
    returns = frame.pct_change(fill_method=None)
    returns = returns.iloc[1:]
    return returns.astype("float64")


def forward_returns(
    prices: PricesLike,
    *,
    horizon: int = 21,
) -> pd.DataFrame:
    r"""Compute non-overlapping ``horizon``-day forward simple returns.

    For each date ``t`` and issuer, returns ``price[t + horizon] / price[t] - 1``
    — the label earned strictly AFTER ``t``. This is the forward-return panel a
    filing's tone score is joined to (keyed on its acceptance date).

    NO-LOOKAHEAD REQUIREMENT: the label at ``t`` uses only prices at and after
    ``t``; it is the realized future return, never aligned to past prices.

    Parameters
    ----------
    prices:
        A wide panel of prices (rows = date, columns = issuer).
    horizon:
        Forward horizon in trading days (``> 0``).

    Returns
    -------
    pandas.DataFrame
        The forward-return panel aligned to the decision date ``t``.

    Raises
    ------
    ValidationError
        If ``horizon <= 0`` or ``prices`` is malformed.
    """
    if horizon <= 0:
        raise ValidationError(f"forward_returns: horizon must be > 0, got {horizon}.")

    frame = ensure_dataframe(prices, name="prices", allow_nan=True)

    # NO-LOOKAHEAD: the label at row ``t`` is ``price[t + horizon] / price[t] - 1``
    # — built by dividing the panel shifted BACKWARD by ``horizon`` (future prices)
    # by the current price. The label therefore uses only prices at and after ``t``
    # and is aligned to the decision date ``t``. The trailing ``horizon`` rows have
    # no realized future price and are dropped.
    future = frame.shift(-horizon)
    fwd = (future / frame) - 1.0
    fwd = fwd.iloc[: frame.shape[0] - horizon] if frame.shape[0] > horizon else fwd.iloc[0:0]
    return fwd.astype("float64")
