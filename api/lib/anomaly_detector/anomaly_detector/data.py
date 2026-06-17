"""Data layer: synthetic anomaly-injected generator + EOD price loader.

Two responsibilities, both import-pure (heavy deps are imported lazily inside
functions):

1. :func:`generate_injected_series` — a deterministic, seeded base low-vol
   return series with VOLATILITY BURSTS and JUMPS injected at KNOWN indices, so
   the detectors have a recoverable core to be tested against (with no network).
   The known injection indices are returned alongside the series so the
   regression suite can assert recovery without any ground-truth leakage into
   the detectors themselves.

2. :func:`load_prices` — fetch real daily EOD closes for a single ticker via the
   existing Polygon provider (reused from the HRP infra), degrading to the
   deterministic synthetic path on any upstream failure, and reporting which
   source was used (``"polygon"`` | ``"synthetic"``). Results are best-effort
   cached to a parquet file (the ``data`` extra) when caching is requested; the
   cache is a no-op when ``pyarrow`` is unavailable.

NO-LOOKAHEAD: returns are differenced with ``pct_change(fill_method=None)`` via
:func:`anomaly_detector.data.compute_returns` (re-exported from the shared
return helper); prices are never forward-filled before differencing.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from anomaly_detector._exceptions import ValidationError
from anomaly_detector._rng import make_rng

if TYPE_CHECKING:
    from pathlib import Path

    from anomaly_detector._typing import PricesLike

#: Where a price/return series ultimately came from (for the API ``data_source``).
DataSource = Literal["polygon", "synthetic"]

#: Default fixed liquid-ETF universe the deployed tool fits on (single-ticker per
#: request; the set documents the survivorship justification in the README).
DEFAULT_TICKER: str = "SPY"

#: Initial price level the synthetic price path is cumulated from.
_BASE_PRICE: float = 100.0

#: Length (in days) of each injected volatility-burst window.
_BURST_LEN: int = 5


@dataclass(frozen=True, slots=True)
class InjectedSeries:
    """A synthetic return series with anomalies injected at KNOWN indices.

    Attributes
    ----------
    returns:
        The synthetic per-day return series (calm base + injected stress),
        indexed by date.
    prices:
        The corresponding price level series (cumulated from ``returns``),
        indexed by the same dates.
    vol_burst_idx:
        The positional indices (into ``returns``) where volatility bursts were
        injected.
    jump_idx:
        The positional indices where discrete jumps were injected.
    meta:
        Free-form JSON-serializable provenance (seed, base vol, burst/jump sizes).
    """

    returns: pd.Series
    prices: pd.Series
    vol_burst_idx: tuple[int, ...]
    jump_idx: tuple[int, ...]
    meta: dict[str, Any] = field(default_factory=dict)

    def known_anomaly_idx(self) -> tuple[int, ...]:
        """Return the sorted union of injected vol-burst and jump indices.

        A volatility burst spans :data:`_BURST_LEN` consecutive days starting at
        each recorded burst index, so the full window is expanded here; jump
        indices are single days. The union is de-duplicated and ascending so the
        regression suite can compare detector flags against it without any
        ground-truth leaking into the detectors.

        Returns
        -------
        tuple[int, ...]
            The known anomalous positional indices, ascending and de-duplicated.
        """
        n_obs = int(self.returns.shape[0])
        idx: set[int] = set()
        for start_i in self.vol_burst_idx:
            for offset in range(_BURST_LEN):
                pos = start_i + offset
                if 0 <= pos < n_obs:
                    idx.add(pos)
        for j in self.jump_idx:
            if 0 <= j < n_obs:
                idx.add(j)
        return tuple(sorted(idx))

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this series.

        Timestamps are rendered as ISO date strings so the result crosses the
        API boundary cleanly.

        Returns
        -------
        dict[str, Any]
            The returns, prices, injection indices, and provenance metadata.
        """
        dates = [ts.isoformat() for ts in pd.DatetimeIndex(self.returns.index)]
        return {
            "dates": dates,
            "returns": [float(x) for x in self.returns.to_numpy(dtype="float64")],
            "prices": [float(x) for x in self.prices.to_numpy(dtype="float64")],
            "vol_burst_idx": list(self.vol_burst_idx),
            "jump_idx": list(self.jump_idx),
            "known_anomaly_idx": list(self.known_anomaly_idx()),
            "meta": dict(self.meta),
        }


def _business_index(n_obs: int, start: date) -> pd.DatetimeIndex:
    """Return an ``n_obs``-long business-day (Mon-Fri) index starting at ``start``."""
    return pd.date_range(start=start, periods=n_obs, freq="B")


def generate_injected_series(
    *,
    n_obs: int = 1000,
    seed: int = 7,
    base_vol: float = 0.008,
    n_vol_bursts: int = 6,
    n_jumps: int = 6,
    burst_vol_mult: float = 5.0,
    jump_size: float = 0.06,
    start: date = date(2015, 1, 1),
) -> InjectedSeries:
    """Generate a deterministic return series with anomalies at KNOWN indices.

    Builds a calm Gaussian base return series (``base_vol`` per-day std), then
    injects ``n_vol_bursts`` short volatility-burst windows and ``n_jumps``
    discrete jumps at indices drawn from a seeded
    :func:`anomaly_detector._rng.make_rng` generator. The injection indices are
    recorded on the returned :class:`InjectedSeries` so tests can assert detector
    recovery WITHOUT feeding any label into the detectors.

    The injections are placed in the back half of the series (well-separated and
    deterministic), so a causal train/OOS split can keep them out-of-sample.

    Parameters
    ----------
    n_obs:
        Number of trading days to generate.
    seed:
        Master RNG seed (deterministic; same seed -> byte-identical series).
    base_vol:
        Per-day standard deviation of the calm base regime.
    n_vol_bursts:
        Number of volatility-burst windows to inject.
    n_jumps:
        Number of discrete jumps to inject.
    burst_vol_mult:
        Multiplier applied to ``base_vol`` inside a burst window.
    jump_size:
        Absolute return size of an injected jump (sign is randomized).
    start:
        First date of the generated business-day index.

    Returns
    -------
    InjectedSeries
        The synthetic series with its known injection indices.

    Raises
    ------
    ValidationError
        If ``n_obs`` is too small for the requested injections, or any size
        parameter is non-positive.
    """
    if n_obs <= 0:
        raise ValidationError(f"generate_injected_series: n_obs must be positive, got {n_obs}.")
    if base_vol <= 0.0:
        raise ValidationError(
            f"generate_injected_series: base_vol must be positive, got {base_vol}."
        )
    if n_vol_bursts < 0 or n_jumps < 0:
        raise ValidationError(
            "generate_injected_series: n_vol_bursts and n_jumps must be non-negative "
            f"(got {n_vol_bursts}, {n_jumps})."
        )
    if burst_vol_mult <= 0.0:
        raise ValidationError(
            f"generate_injected_series: burst_vol_mult must be positive, got {burst_vol_mult}."
        )
    if jump_size <= 0.0:
        raise ValidationError(
            f"generate_injected_series: jump_size must be positive, got {jump_size}."
        )

    # The injections live in the back half; require enough room for them plus a
    # full burst window at the tail without spilling past the series end.
    need = max(n_vol_bursts, n_jumps) + _BURST_LEN
    if n_obs < 2 * need:
        raise ValidationError(
            f"generate_injected_series: n_obs={n_obs} is too small for "
            f"{n_vol_bursts} bursts and {n_jumps} jumps (need at least {2 * need})."
        )

    gen = make_rng(seed)
    returns = gen.standard_normal(n_obs) * base_vol

    # Deterministic, well-separated injection indices in the back half. Bursts
    # end at least one burst-length before the series end; jumps are offset so
    # they do not coincide exactly with burst starts.
    hi_burst = n_obs - _BURST_LEN - 1
    lo = n_obs // 2 + _BURST_LEN
    vol_burst_idx: tuple[int, ...] = (
        tuple(int(i) for i in np.unique(np.linspace(lo, hi_burst, n_vol_bursts).astype(int)))
        if n_vol_bursts > 0
        else ()
    )
    jump_offset = 3
    hi_jump = n_obs - 1 - jump_offset
    lo_jump = n_obs // 2
    jump_idx: tuple[int, ...] = (
        tuple(
            int(i)
            for i in np.unique(np.linspace(lo_jump, hi_jump, n_jumps).astype(int) + jump_offset)
            if int(i) < n_obs
        )
        if n_jumps > 0
        else ()
    )

    # Volatility bursts: a short window of inflated-variance returns.
    for start_i in vol_burst_idx:
        end_i = min(start_i + _BURST_LEN, n_obs)
        returns[start_i:end_i] = gen.standard_normal(end_i - start_i) * (base_vol * burst_vol_mult)

    # Discrete jumps: a single large signed return.
    for j in jump_idx:
        if 0 <= j < n_obs:
            sign = 1.0 if gen.random() < 0.5 else -1.0
            returns[j] = sign * jump_size

    index = _business_index(n_obs, start)
    ret_series = pd.Series(returns, index=index, name="return")
    price_levels = _BASE_PRICE * np.cumprod(1.0 + returns)
    prices = pd.Series(price_levels, index=index, name="price")

    meta: dict[str, Any] = {
        "seed": int(seed),
        "n_obs": int(n_obs),
        "base_vol": float(base_vol),
        "n_vol_bursts": int(n_vol_bursts),
        "n_jumps": int(n_jumps),
        "burst_vol_mult": float(burst_vol_mult),
        "jump_size": float(jump_size),
        "burst_len": int(_BURST_LEN),
        "start": start.isoformat(),
    }
    return InjectedSeries(
        returns=ret_series,
        prices=prices,
        vol_burst_idx=vol_burst_idx,
        jump_idx=jump_idx,
        meta=meta,
    )


def _synthetic_prices(ticker: str, start: date, end: date, seed: int) -> pd.Series:
    """Deterministic synthetic close series for one ticker over ``[start, end]``.

    Reuses :func:`generate_injected_series` so the offline path shares the exact
    same injected-anomaly structure the detectors are validated against. The
    series is clipped to the requested date range; the seed is mixed with the
    ticker so different symbols yield different (but reproducible) paths.
    """
    index = _business_index_inclusive(start, end)
    n_obs = len(index)
    if n_obs == 0:
        return pd.Series(dtype="float64", name=ticker)

    # Mix the ticker into the seed deterministically (masked to 31 bits).
    mixed = (seed * 1_000_003 + (hash(ticker) & 0x7FFFFFFF)) & 0x7FFFFFFF
    injected = generate_injected_series(n_obs=n_obs, seed=mixed, start=start)
    closes = injected.prices.to_numpy(dtype="float64")
    return pd.Series(closes, index=index, name=ticker)


def _business_index_inclusive(start: date, end: date) -> pd.DatetimeIndex:
    """Inclusive business-day (Mon-Fri) index spanning ``[start, end]``."""
    return pd.date_range(start=start, end=end, freq="B")


def _fetch_polygon_close(ticker: str, start: date, end: date) -> pd.Series:
    """Fetch one ticker's daily adjusted close from Polygon (lazy import). May raise.

    LAZY IMPORT: :class:`anomaly_detector.data_providers.polygon.PolygonProvider`
    (and, inside it, ``httpx``) are imported here, never at module import time.
    """
    from anomaly_detector.data_providers.polygon import PolygonProvider

    frame = PolygonProvider().fetch([ticker], start, end)
    if frame.empty or bool(frame.isna().all(axis=None)):
        raise ValueError(f"Polygon returned no usable price data for {ticker}.")
    series = frame[ticker].astype("float64")
    series.name = ticker
    return series.dropna()


def _cache_path(ticker: str, start: date, end: date) -> Path:
    """Deterministic parquet cache path for a (ticker, start, end) request."""
    import tempfile
    from pathlib import Path

    from anomaly_detector._manifest import config_hash

    key = config_hash({"ticker": ticker, "start": start.isoformat(), "end": end.isoformat()})
    cache_dir = Path(tempfile.gettempdir()) / "anomaly_detector_cache"
    return cache_dir / f"prices_{ticker}_{key}.parquet"


def _read_cache(path: Path) -> pd.Series | None:
    """Best-effort read of a cached close series; ``None`` on any failure.

    LAZY: ``pyarrow`` (the ``data`` extra) is only touched here. A missing cache
    file, an absent ``pyarrow``, or a corrupt file all degrade to ``None`` so the
    caller falls through to a live fetch.
    """
    if not path.is_file():
        return None
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return None
    if frame.shape[1] == 0:
        return None
    series = frame.iloc[:, 0].astype("float64")
    series.index = pd.to_datetime(series.index)
    return series


def _write_cache(path: Path, series: pd.Series) -> None:
    """Best-effort write of a close series to parquet; silently no-ops on failure.

    LAZY: ``pyarrow`` is only required here; if it (or the filesystem) is
    unavailable the write is skipped so the live result is still returned.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        series.to_frame(name=series.name or "close").to_parquet(path)
    except Exception:
        return


def load_prices(
    ticker: str,
    start: date,
    end: date,
    *,
    source_pref: Literal["auto", "polygon", "synthetic"] = "auto",
    seed: int = 7,
    use_cache: bool = True,
) -> tuple[pd.Series, DataSource]:
    """Load a single-ticker daily close series, degrading to synthetic.

    With ``source_pref="polygon"`` (or ``"auto"``) the real Polygon EOD provider
    is tried first; on ANY failure — and always for ``"synthetic"`` — a
    deterministic synthetic price series (from :func:`generate_injected_series`)
    is returned. The second element reports which path was taken so the API can
    surface a ``data_source`` badge.

    A successful Polygon fetch is best-effort cached to a parquet file (the
    ``data`` extra); a warm cache short-circuits the network. The cache is a
    no-op when ``pyarrow`` is unavailable, so the offline/CI path never depends
    on it.

    LAZY IMPORT: the Polygon provider (and ``httpx``), and ``pyarrow`` for the
    cache, are imported inside this function, never at module import time.

    Parameters
    ----------
    ticker:
        The asset symbol (e.g. ``"SPY"``).
    start, end:
        Inclusive date range.
    source_pref:
        ``"auto"``/``"polygon"`` try Polygon then fall back; ``"synthetic"``
        forces the offline path.
    seed:
        Seed for the synthetic fallback (deterministic).
    use_cache:
        Whether to read/write the parquet cache for live fetches.

    Returns
    -------
    tuple[pandas.Series, DataSource]
        The daily close series and the source it came from.

    Raises
    ------
    ValidationError
        If ``ticker`` is empty or ``end <= start``.
    """
    symbol = ticker.strip()
    if not symbol:
        raise ValidationError("load_prices: ticker must be a non-empty string.")
    if end <= start:
        raise ValidationError(f"load_prices: end ({end}) must be after start ({start}).")

    if source_pref == "synthetic":
        return _synthetic_prices(symbol, start, end, seed), "synthetic"

    # "auto" / "polygon": try a warm cache, then a live Polygon fetch, then
    # degrade to the deterministic synthetic path on any failure.
    if use_cache:
        path = _cache_path(symbol, start, end)
        cached = _read_cache(path)
        if cached is not None and not cached.empty:
            return cached.astype("float64"), "polygon"

    try:
        series = _fetch_polygon_close(symbol, start, end)
    except Exception:
        return _synthetic_prices(symbol, start, end, seed), "synthetic"

    if series.empty:
        return _synthetic_prices(symbol, start, end, seed), "synthetic"

    if use_cache:
        _write_cache(_cache_path(symbol, start, end), series)
    return series.astype("float64"), "polygon"


def compute_returns(prices: PricesLike) -> pd.Series:
    """Convert a price series to simple returns with no lookahead.

    NO-LOOKAHEAD REQUIREMENT: differenced with ``pct_change(fill_method=None)``
    — prices are NEVER forward-filled before differencing (ffill-then-diff
    manufactures spurious zero returns across gaps and leaks information). The
    leading NaN row is dropped.

    Parameters
    ----------
    prices:
        A price level series (or single-column price panel) indexed by date.

    Returns
    -------
    pandas.Series
        Simple returns with the leading NaN row removed.

    Raises
    ------
    ValidationError
        If ``prices`` is malformed.
    """
    if isinstance(prices, pd.DataFrame):
        if prices.shape[1] != 1:
            raise ValidationError(
                f"compute_returns: a price DataFrame must have exactly one column, "
                f"got {prices.shape[1]}."
            )
        series = prices.iloc[:, 0]
    elif isinstance(prices, pd.Series):
        series = prices
    elif isinstance(prices, np.ndarray):
        if prices.ndim != 1:
            raise ValidationError(
                f"compute_returns: a price array must be 1-dimensional, got ndim={prices.ndim}."
            )
        series = pd.Series(prices)
    else:
        raise ValidationError(
            "compute_returns: prices must be a pandas Series/DataFrame or a 1-D ndarray."
        )

    series = series.astype("float64")
    if series.empty:
        raise ValidationError("compute_returns: prices must be non-empty.")

    # NO-LOOKAHEAD: never forward-fill prices before differencing.
    returns = series.pct_change(fill_method=None)
    returns = returns.iloc[1:]
    returns.name = "return"
    return returns.astype("float64")
