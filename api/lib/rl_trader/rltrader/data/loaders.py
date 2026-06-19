"""Real-data loaders: Polygon PIT single-asset bars (CLI path) + synthetic default.

The default, deployed data path is the seeded synthetic price process
(:mod:`rltrader.data.synthetic`) — no API keys, no survivorship questions, and the
honest NULL holds by construction. This module is the OFFLINE CLI path for real
data:

- :func:`load_single_asset_bars` fetches a single ticker's point-in-time daily
  bars via the vendored Polygon provider over a date span, computes per-bar
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

from rltrader._exceptions import ValidationError
from rltrader.data import DataSource, compute_returns
from rltrader.data.synthetic import (
    PricePath,
    gbm_regime_path,
    pure_noise_path,
    pure_trend_path,
)

#: The synthetic process kinds routed by :func:`synthetic_default_path`.
_SYNTHETIC_KINDS: frozenset[str] = frozenset({"gbm_regime", "pure_trend", "pure_noise"})


def load_single_asset_bars(
    ticker: str,
    *,
    start: date,
    end: date,
    data_source_pref: str = "synthetic",
    seed: int = 7,
) -> tuple[pd.Series, pd.Series, DataSource]:
    """Load a single asset's PIT price path + per-bar returns and tag the provenance.

    With ``data_source_pref="polygon"`` the vendored Polygon provider is tried
    first; ``"synthetic"`` (default) and any provider failure fall straight through
    to the deterministic synthetic GBM-regime path so the loader is usable offline
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
    tuple[pandas.Series, pandas.Series, DataSource]
        The price path, the per-bar return series, and the resolved source label.

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
        fetched = _try_polygon_bars(ticker, start=start, end=end)
        if fetched is not None:
            prices = fetched
            returns = compute_returns(prices)
            return prices, returns, "polygon"
        # Provider failed (no key / no network / data extra absent): fall through.

    # Synthetic fallback / default: deterministic, offline, CI-safe.
    n_obs = max(2, _bdays_between(start, end))
    return synthetic_default_path(n_obs=n_obs, seed=seed, kind="gbm_regime")


def _bdays_between(start: date, end: date) -> int:
    """Return the number of business days spanning ``[start, end]`` (inclusive-ish).

    Sizes the synthetic fallback path to the requested calendar span so the loader
    returns a path of a sensible length for the asked-for window.
    """
    span = pd.bdate_range(start=pd.Timestamp(start), end=pd.Timestamp(end))
    return int(span.size)


def _try_polygon_bars(ticker: str, *, start: date, end: date) -> pd.Series | None:
    """Best-effort fetch of one ticker's PIT closes via the vendored Polygon provider.

    LAZY IMPORT: the provider (and ``httpx`` inside it — the ``data`` extra) is
    imported here, not at module load. Any failure — missing key, no network, the
    ``data`` extra absent, or an empty payload — returns ``None`` so the caller can
    fall through to the deterministic synthetic path.
    """
    try:
        from rltrader.data_providers.polygon import PolygonProvider

        provider = PolygonProvider()
        panel = provider.fetch([ticker], start, end)
    except Exception:  # any provider failure (no key/network/extra) -> synthetic fallback.
        return None

    if panel.empty or ticker not in panel.columns:
        return None
    series = panel[ticker].astype("float64")
    series = series[series.notna()]
    if series.empty:
        return None
    return series


def synthetic_default_path(
    *,
    n_obs: int = 2000,
    seed: int = 7,
    kind: str = "gbm_regime",
) -> tuple[pd.Series, pd.Series, DataSource]:
    """Build the deployed-default synthetic price path + returns (torch-free, no network).

    Routes to the requested synthetic process (``"gbm_regime"`` = the honest-null
    default, ``"pure_trend"`` = the sanity fixture, ``"pure_noise"`` = the strict
    null) and returns the price path, per-bar returns, and the ``"synthetic"``
    provenance label. The deployed request path uses this — it never needs a key or
    network.

    Parameters
    ----------
    n_obs:
        Number of bars to generate.
    seed:
        Master RNG seed.
    kind:
        One of ``{"gbm_regime", "pure_trend", "pure_noise"}``.

    Returns
    -------
    tuple[pandas.Series, pandas.Series, DataSource]
        The price path, per-bar returns, and ``"synthetic"``.

    Raises
    ------
    ValidationError
        If ``kind`` is unknown or ``n_obs < 2``.

    """
    if kind not in _SYNTHETIC_KINDS:
        raise ValidationError(
            f"synthetic_default_path: unknown kind {kind!r}; "
            f"expected one of {sorted(_SYNTHETIC_KINDS)}."
        )
    if n_obs < 2:
        raise ValidationError(f"synthetic_default_path: n_obs must be >= 2, got {n_obs}.")

    path: PricePath
    if kind == "gbm_regime":
        path = gbm_regime_path(n_obs=n_obs, seed=seed)
    elif kind == "pure_trend":
        path = pure_trend_path(n_obs=n_obs, seed=seed)
    else:  # "pure_noise"
        path = pure_noise_path(n_obs=n_obs, seed=seed)

    prices = path.prices
    returns = compute_returns(prices)
    return prices, returns, "synthetic"
