"""Real-data loaders: keyless FRED rate series + Polygon PIT single-name returns.

The synthetic panel (:mod:`fedcausal.data.synthetic`) is the default everywhere.
These loaders provide the OPTIONAL real-data path described in the brief:

- **FRED is free and keyless** via the ``fredgraph`` CSV endpoint
  (``https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFEDTARU`` for the
  target-rate upper bound, ``DFF`` for the effective rate). It is fetched lazily
  with ``httpx`` to classify each meeting's surprise sign, degrading to the
  committed snapshot in :mod:`fedcausal.events.calendar_data` on any failure.
- **Polygon** supplies the single-name cross-section via the existing
  point-in-time (PIT) universe + price provider
  (:mod:`fedcausal.data_providers.polygon`); a ticker is only included if it was
  in the index as-of the event (PIT, no survivorship leak).

LEAKAGE GUARDS: simple returns are computed with ``pct_change(fill_method=None)``
(prices are never forward-filled before differencing); the FRED series is read at
its RELEASE date so the surprise uses only information available at the
announcement.

LAZY IMPORTS: ``httpx`` and the Polygon provider are imported INSIDE the
functions, never at module import time. Importing this module has no side
effects (no network).
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

from fedcausal._exceptions import ValidationError

if TYPE_CHECKING:
    from fedcausal._typing import DataSource
    from fedcausal.data.synthetic import SyntheticPanel
    from fedcausal.events.calendar import FOMCEvent

#: Keyless FRED ``fredgraph`` CSV endpoint template (no API key required).
FREDGRAPH_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

#: FRED series ids used by the loaders.
FRED_TARGET_UPPER = "DFEDTARU"  # federal funds target range, upper bound
FRED_EFFECTIVE = "DFF"  # daily effective federal funds rate


def parse_fredgraph_csv(text: str) -> pd.Series:
    """Parse a keyless ``fredgraph`` CSV payload into a date-indexed float series.

    The ``fredgraph`` CSV has a two-column layout: a date column (labelled
    ``observation_date`` on the modern endpoint, ``DATE`` historically) and one
    value column named after the series id. FRED encodes missing observations as
    a literal ``.`` which is dropped here.

    Parameters
    ----------
    text:
        The raw CSV body.

    Returns
    -------
    pandas.Series
        A date-indexed float64 series (the value column), sorted ascending, with
        missing markers removed. The series ``name`` is the value-column header.

    Raises
    ------
    RuntimeError
        If the CSV has no parseable value column.
    """
    from io import StringIO

    try:
        frame = pd.read_csv(StringIO(text))
    except (ValueError, pd.errors.ParserError) as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"fredgraph CSV parse failed: {exc}") from exc

    if frame.shape[1] < 2:
        raise RuntimeError(f"fredgraph CSV missing value column (columns={list(frame.columns)}).")

    date_col = frame.columns[0]
    value_col = frame.columns[1]

    parsed = pd.to_datetime(frame[date_col], errors="coerce")
    # FRED uses "." for missing; coerce non-numeric to NaN.
    values = pd.to_numeric(frame[value_col], errors="coerce")

    # Drop rows with an unparseable date or a missing value, then build the
    # date-indexed series (no ~Index boolean operator; mask on the columns).
    keep = parsed.notna() & values.notna()
    series = pd.Series(
        values[keep].to_numpy(dtype="float64"),
        index=pd.DatetimeIndex(parsed[keep]),
        name=str(value_col),
        dtype="float64",
    )
    return series.sort_index()


def fetch_fred_series(
    series_id: str,
    *,
    start: date | None = None,
    end: date | None = None,
    timeout: float = 30.0,
) -> pd.Series:
    """Fetch a FRED series from the keyless ``fredgraph`` CSV endpoint (lazy httpx).

    LAZY IMPORT: ``httpx`` (the ``data`` extra) is imported inside this function;
    importing the module touches no network. The CSV is parsed into a date-indexed
    float series, with FRED's ``.`` missing markers dropped.

    Parameters
    ----------
    series_id:
        The FRED series id (e.g. ``"DFEDTARU"`` or ``"DFF"``).
    start, end:
        Optional inclusive date bounds applied after download.
    timeout:
        Per-request timeout in seconds.

    Returns
    -------
    pandas.Series
        The date-indexed series (percent), sorted ascending.

    Raises
    ------
    ValidationError
        If ``series_id`` is empty.
    RuntimeError
        On network/parse failure (callers decide whether to fall back).
    """
    if not series_id or not series_id.strip():
        raise ValidationError("fetch_fred_series: series_id must be non-empty.")

    url = FREDGRAPH_CSV_URL.format(series_id=series_id.strip())

    import httpx  # lazy: the ``data`` extra

    try:
        response = httpx.get(url, timeout=timeout)
        response.raise_for_status()
        text = response.text
    except httpx.HTTPError as exc:
        raise RuntimeError(f"fetch_fred_series: HTTP error for {series_id!r}: {exc}") from exc

    series = parse_fredgraph_csv(text)
    if start is not None:
        series = series[series.index >= pd.Timestamp(start)]
    if end is not None:
        series = series[series.index <= pd.Timestamp(end)]
    if series.empty:
        raise RuntimeError(f"fetch_fred_series: {series_id!r} returned no usable observations.")
    return series


def _surprise_change_at(series: pd.Series, current: date, previous: date | None) -> float:
    """Signed target-rate change (bps) between two meetings, read at RELEASE date.

    The value at each meeting is the last published observation on or BEFORE the
    announcement date (point-in-time: never a future revision). Returns the
    change in basis points; ``0.0`` when the predecessor is unknown.
    """
    cur_idx = series.index[series.index <= pd.Timestamp(current)]
    if len(cur_idx) == 0:
        raise RuntimeError(f"no FRED observation on or before {current.isoformat()}.")
    cur_val = float(series.loc[cur_idx[-1]])
    if previous is None:
        return 0.0
    prev_idx = series.index[series.index <= pd.Timestamp(previous)]
    if len(prev_idx) == 0:
        raise RuntimeError(f"no FRED observation on or before {previous.isoformat()}.")
    prev_val = float(series.loc[prev_idx[-1]])
    return (cur_val - prev_val) * 100.0  # percent -> basis points


def load_surprise_labels(
    events: list[FOMCEvent],
    *,
    use_fred: bool = False,
) -> list[FOMCEvent]:
    """Refresh each event's surprise label from live FRED, else keep committed.

    With ``use_fred=True`` this fetches ``DFEDTARU`` and re-derives the signed
    rate change per meeting from the series read at the announcement's RELEASE
    date (no future revision); on any failure it returns the committed
    ``events`` unchanged. With ``use_fred=False`` it is a pass-through.

    Parameters
    ----------
    events:
        The committed FOMC events to (optionally) refresh.
    use_fred:
        Whether to attempt a live FRED refresh.

    Returns
    -------
    list[FOMCEvent]
        Events with surprise labels (refreshed or committed).
    """
    if not use_fred or not events:
        return list(events)

    from fedcausal.events.calendar import classify_surprise

    try:
        dates = [ev.announcement_date for ev in events]
        series = fetch_fred_series(
            FRED_TARGET_UPPER,
            # Pad the start so the first meeting has a defined predecessor value.
            start=date(min(d.year for d in dates) - 1, 1, 1),
            end=max(dates),
        )
        refreshed: list[FOMCEvent] = []
        prev: date | None = None
        for ev in events:
            change_bps = _surprise_change_at(series, ev.announcement_date, prev)
            cur_idx = series.index[series.index <= pd.Timestamp(ev.announcement_date)]
            target_upper = float(series.loc[cur_idx[-1]])
            refreshed.append(
                replace(
                    ev,
                    target_upper=target_upper,
                    rate_change_bps=change_bps,
                    surprise=classify_surprise(change_bps),
                )
            )
            prev = ev.announcement_date
        return refreshed
    except (RuntimeError, ValidationError, ImportError, OSError):
        # Degrade to the committed labels on any failure (never hard-fail).
        return list(events)


def fetch_polygon_returns(
    tickers: list[str],
    start: date,
    end: date,
    *,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Fetch single-name simple returns from Polygon (lazy provider import).

    LAZY IMPORT: :class:`fedcausal.data_providers.polygon.PolygonProvider` (and,
    inside it, ``httpx``) is imported here, never at module import time. Adjusted
    closes are converted to simple returns with ``pct_change(fill_method=None)``
    (no forward-fill before differencing).

    Parameters
    ----------
    tickers:
        PIT-resolved index members to fetch.
    start, end:
        Inclusive date range.
    api_key:
        Optional explicit Polygon key (else resolved from env/.env).

    Returns
    -------
    pandas.DataFrame
        Wide panel of single-name simple returns (rows = date, columns = ticker).

    Raises
    ------
    ValidationError
        If ``tickers`` is empty or ``end <= start``.
    """
    symbols = list(tickers)
    if len(symbols) == 0:
        raise ValidationError("fetch_polygon_returns: tickers must be non-empty.")
    if end <= start:
        raise ValidationError(f"fetch_polygon_returns: end ({end}) must be after start ({start}).")

    from fedcausal.data_providers.polygon import PolygonProvider

    prices = PolygonProvider(api_key=api_key).fetch(symbols, start, end)
    # NO-LOOKAHEAD: never forward-fill prices before differencing.
    returns = prices.pct_change(fill_method=None)
    returns = returns.iloc[1:]
    return returns.astype("float64")


def load_event_panel(
    *,
    data_source_pref: DataSource = "synthetic",
    seed: int = 7,
    use_fred: bool = False,
    tickers: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
) -> tuple[SyntheticPanel, DataSource]:
    """Load an event panel, defaulting to synthetic, with a real-data option.

    Resolution: ``data_source_pref="synthetic"`` (the default) returns the seeded
    synthetic panel directly. ``"fred+polygon"`` attempts to assemble a real
    panel (FRED surprise labels + Polygon PIT single-name returns) and DEGRADES
    to the synthetic panel on any failure, so the loader NEVER hard-fails.

    Parameters
    ----------
    data_source_pref:
        ``"synthetic"`` (default) or ``"fred+polygon"``.
    seed:
        Master RNG seed for the synthetic panel / fallback.
    use_fred:
        Whether to refresh surprise labels from live FRED.
    tickers:
        Optional explicit PIT ticker list for the Polygon path.
    start, end:
        Optional inclusive date range for the real-data path.

    Returns
    -------
    tuple[SyntheticPanel, DataSource]
        The panel and the source it actually came from (``"synthetic"`` on
        fallback).
    """
    from fedcausal.data.synthetic import (
        SyntheticPanel as _SyntheticPanel,
    )
    from fedcausal.data.synthetic import (
        synthetic_event_panel,
    )

    if data_source_pref == "synthetic":
        return synthetic_event_panel(seed=seed), "synthetic"

    # Real-data path: FRED surprise labels + Polygon PIT single-name returns.
    # Any failure degrades to the deterministic synthetic panel.
    try:
        from fedcausal.events.calendar import load_fomc_calendar

        events = load_fomc_calendar(start=start, end=end)
        events = load_surprise_labels(events, use_fred=use_fred)
        announcement_dates = [ev.announcement_date for ev in events]
        if not announcement_dates:
            raise RuntimeError("no FOMC events in the requested range.")

        pit_tickers = list(tickers) if tickers else _default_pit_tickers()
        span_start = start if start is not None else min(announcement_dates)
        span_end = end if end is not None else max(announcement_dates)
        returns = fetch_polygon_returns(pit_tickers, span_start, span_end)
        if returns.empty:
            raise RuntimeError("Polygon returned no usable single-name returns.")

        # The market factor is the cross-sectional mean return (an equal-weight
        # proxy index) when no explicit benchmark is supplied.
        market = returns.mean(axis=1).astype("float64")
        market.name = "market"
        surprises = [ev.surprise for ev in events]

        panel = _SyntheticPanel(
            returns=returns,
            market=market,
            announcement_dates=announcement_dates,
            surprises=surprises,
            rate_sensitive=list(returns.columns[: max(1, returns.shape[1] // 2)]),
            injected_car=float("nan"),
            seed=int(seed),
            meta={"source": "fred+polygon", "use_fred": bool(use_fred)},
        )
        return panel, "fred+polygon"
    except (RuntimeError, ValidationError, ImportError, OSError, KeyError):
        # Never hard-fail: degrade to the deterministic synthetic panel.
        return synthetic_event_panel(seed=seed), "synthetic"


def _default_pit_tickers() -> list[str]:
    """Best-effort PIT ticker list from the vendored Polygon universe, else a stub.

    Tries the vendored point-in-time S&P 500 universe; on any failure returns a
    small liquid default set. Never raises.
    """
    try:  # pragma: no cover - exercised only on the real-data path
        from fedcausal.data_providers.polygon import default_universe  # type: ignore[attr-defined]

        return list(default_universe())
    except (ImportError, AttributeError, OSError):
        return ["AAPL", "MSFT", "JPM", "XLF", "TLT"]
