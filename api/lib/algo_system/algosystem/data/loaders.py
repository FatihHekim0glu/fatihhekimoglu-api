"""Real-data loaders: Polygon PIT single-asset OHLC bars (CLI path) + synthetic default.

The default, deployed data path is the seeded synthetic OHLC bar process
(:mod:`algosystem.data.synthetic`) — no API keys, no survivorship questions, and
the honest NULL holds by construction. This module is the OFFLINE CLI path for
real data:

- :func:`load_single_asset_bars` fetches a single ticker's point-in-time daily
  bars via the vendored Polygon provider over a date span, computes per-bar close
  returns with ``pct_change(fill_method=None)`` (no forward-fill across gaps), and
  tags the provenance (``"polygon"`` / ``"synthetic"``). On any failure (no key,
  no network, ``data`` extra absent) it falls back to the deterministic synthetic
  path so the loader is usable offline and in CI.

Heavy data dependencies (httpx via the vendored Polygon provider, pyarrow,
diskcache) live behind the ``data`` extra and are imported LAZILY inside these
functions, so importing this module pulls in nothing heavy and has no side effects.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from algosystem._exceptions import AlgoSystemError, ValidationError
from algosystem.data import DataSource, compute_returns

#: The synthetic process kinds routed by :func:`synthetic_default_bars`.
SYNTHETIC_KINDS: frozenset[str] = frozenset(
    {"gbm_regime", "learnable_trend", "regime_trend", "pure_noise"}
)

#: Canonical OHLC column order for the bar panel these loaders emit.
_OHLC_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")


def _bars_from_close_panel(closes: pd.Series) -> pd.DataFrame:
    """Assemble a single-asset OHLC bar panel from a Point-in-Time close series.

    The vendored Polygon provider returns adjusted CLOSES only; there is no
    intrabar quote. We build a no-lookahead, leakage-free OHLC frame with the
    gapless-open convention used everywhere in this package — the open of bar
    ``t`` is the prior bar's close (the next-bar fill price the execution engine
    reads) — and a degenerate intrabar envelope where ``high`` / ``low`` bracket
    the open and close so the OHLC invariants hold without inventing any intrabar
    information the data does not contain.
    """
    close = closes.astype("float64")
    open_ = close.shift(1)
    open_.iloc[0] = close.iloc[0]  # first open == first close (gapless anchor).
    high = pd.concat([open_, close], axis=1).max(axis=1)
    low = pd.concat([open_, close], axis=1).min(axis=1)
    frame = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
        index=close.index,
    )
    return frame.reindex(columns=list(_OHLC_COLUMNS)).astype("float64")


def load_single_asset_bars(
    ticker: str,
    *,
    start: date,
    end: date,
    data_source_pref: str = "synthetic",
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.Series, DataSource]:
    """Load a single asset's PIT OHLC bars + per-bar close returns and tag the provenance.

    With ``data_source_pref="polygon"`` the vendored Polygon provider is tried
    first; ``"synthetic"`` (default) and any provider failure fall straight through
    to the deterministic synthetic GBM-regime bars so the loader is usable offline
    and in CI. Returns are computed with ``pct_change(fill_method=None)`` (no
    forward-fill across gaps).

    LAZY IMPORTS: the vendored Polygon provider (and ``httpx`` inside it — the
    ``data`` extra) are imported inside this function, so importing this module is
    cheap and side-effect-free.

    Parameters
    ----------
    ticker:
        The single asset symbol to fetch (e.g. ``"SPY"``).
    start, end:
        Inclusive date span.
    data_source_pref:
        ``"polygon"`` tries the real PIT provider first; ``"synthetic"`` (default)
        and ``"auto"`` resolve to the synthetic path.
    seed:
        Master RNG seed for the synthetic fallback path.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.Series, DataSource]
        The OHLC bar panel, the per-bar close-return series, and the resolved
        source label.

    Raises
    ------
    ValidationError
        If ``ticker`` is empty or ``end <= start``.
    """
    if not ticker or not ticker.strip():
        raise ValidationError("load_single_asset_bars: ticker must be a non-empty symbol.")
    if end <= start:
        raise ValidationError(
            f"load_single_asset_bars: end ({end}) must be after start ({start})."
        )

    if data_source_pref == "polygon":
        try:
            # LAZY: pulls in httpx (the ``data`` extra) only on the real-data path.
            from algosystem.data_providers.polygon import PolygonProvider

            provider = PolygonProvider()
            panel = provider.fetch([ticker], start, end)
            closes = panel[ticker].astype("float64")
            bars = _bars_from_close_panel(closes)
            returns = compute_returns(bars["close"])
            return bars, returns, "polygon"
        except (AlgoSystemError, OSError, RuntimeError, ValueError, KeyError, ImportError):
            # No key / no network / ``data`` extra absent / empty payload: fall
            # through to the deterministic synthetic path so CALLERS stay offline-safe.
            pass

    # ``synthetic`` / ``auto`` / any Polygon failure: deterministic synthetic bars.
    span_obs = max(2, _business_days_between(start, end))
    return synthetic_default_bars(n_obs=span_obs, seed=seed, kind="gbm_regime")


def _business_days_between(start: date, end: date) -> int:
    """Count business days in ``[start, end]`` for the synthetic-fallback length."""
    index = pd.bdate_range(start=pd.Timestamp(start), end=pd.Timestamp(end))
    return len(index)


def synthetic_default_bars(
    *,
    n_obs: int = 2000,
    seed: int = 7,
    kind: str = "gbm_regime",
) -> tuple[pd.DataFrame, pd.Series, DataSource]:
    """Build the deployed-default synthetic OHLC bars + close returns (torch-free, no network).

    Routes to the requested synthetic process (``"gbm_regime"`` = the honest-null
    default, ``"learnable_trend"`` = the monotonic-uptrend long/flat sanity fixture,
    ``"regime_trend"`` = the directional regime-trend the long/short pipeline beats
    buy-and-hold on, ``"pure_noise"`` = the strict null) and returns the OHLC bars,
    per-bar close returns, and the ``"synthetic"`` provenance label. The deployed
    request path uses this — it never needs a key or network.

    Parameters
    ----------
    n_obs:
        Number of bars to generate.
    seed:
        Master RNG seed.
    kind:
        One of ``{"gbm_regime", "learnable_trend", "regime_trend", "pure_noise"}``.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.Series, DataSource]
        The OHLC bars, per-bar close returns, and ``"synthetic"``.

    Raises
    ------
    ValidationError
        If ``kind`` is unknown or ``n_obs < 2``.
    """
    if kind not in SYNTHETIC_KINDS:
        raise ValidationError(
            f"synthetic_default_bars: unknown kind {kind!r}; "
            f"expected one of {sorted(SYNTHETIC_KINDS)}."
        )
    if n_obs < 2:
        raise ValidationError(f"synthetic_default_bars: n_obs must be >= 2, got {n_obs}.")

    # Import the generators here (cheap numpy/pandas only; no network, no heavy deps).
    from algosystem.data.synthetic import (
        gbm_regime_bars,
        learnable_trend_bars,
        pure_noise_bars,
        regime_trend_bars,
    )

    if kind == "gbm_regime":
        path = gbm_regime_bars(n_obs=n_obs, seed=seed)
    elif kind == "learnable_trend":
        path = learnable_trend_bars(n_obs=n_obs, seed=seed)
    elif kind == "regime_trend":
        path = regime_trend_bars(n_obs=n_obs, seed=seed)
    else:  # "pure_noise"
        path = pure_noise_bars(n_obs=n_obs, seed=seed)

    bars = path.bars
    returns = compute_returns(bars["close"])
    return bars, returns, "synthetic"
