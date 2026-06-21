"""Multi-asset data loaders: synthetic default + the optional offline PIT readers.

The default, deployed data path is the seeded synthetic multi-asset panel
(:mod:`rlallocator.data.synthetic`) — no API keys, no survivorship questions, and the
honest NULL holds by construction. This module is the OFFLINE CLI path for real
cross-asset data:

- :func:`load_multi_asset_panel` fetches a basket of tickers' point-in-time daily
  bars via the vendored Polygon provider (or an EODHD reader behind an optional key),
  computes per-bar returns with ``pct_change(fill_method=None)`` (no forward-fill
  across gaps), and tags the provenance. On any failure (no key, no network, the
  ``data`` extra absent) it falls back to the deterministic synthetic panel so the
  loader is usable offline and in CI.

Heavy data dependencies (httpx via the vendored Polygon provider, pyarrow, diskcache)
live behind the ``data`` extra and are imported LAZILY inside these functions, so
importing this module pulls in nothing heavy and has no side effects.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import pandas as pd

from rlallocator._exceptions import ValidationError
from rlallocator.data import DataSource, compute_returns
from rlallocator.data.synthetic import (
    ReturnPanelData,
    factor_regime_panel,
    learnable_edge_panel,
    pure_noise_panel,
)

#: The synthetic panel kinds routed by :func:`synthetic_default_panel`.
_SYNTHETIC_KINDS: frozenset[str] = frozenset({"factor_regime", "learnable_edge", "pure_noise"})


def synthetic_default_panel(
    *,
    n_obs: int = 2000,
    n_assets: int = 6,
    seed: int = 7,
    kind: str = "factor_regime",
) -> tuple[pd.DataFrame, pd.DataFrame, DataSource]:
    """Build the deployed-default synthetic price + return panels (torch-free, no network).

    Routes to the requested synthetic panel (``"factor_regime"`` = the honest-null
    default, ``"learnable_edge"`` = the sanity fixture, ``"pure_noise"`` = the strict
    null) and returns a synthetic PRICE panel (cumulative product of ``1 + returns``,
    anchored at 100), the per-bar return panel, and the ``"synthetic"`` provenance
    label. The deployed request path uses this — it never needs a key or network.

    Parameters
    ----------
    n_obs:
        Number of bars to generate.
    n_assets:
        Number of assets in the basket.
    seed:
        Master RNG seed.
    kind:
        One of ``{"factor_regime", "learnable_edge", "pure_noise"}``.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, DataSource]
        The price panel, per-bar return panel, and ``"synthetic"``.

    Raises
    ------
    ValidationError
        If ``kind`` is unknown or a dimension is below its minimum.
    """
    if kind not in _SYNTHETIC_KINDS:
        raise ValidationError(
            f"synthetic_default_panel: unknown kind {kind!r}; "
            f"expected one of {sorted(_SYNTHETIC_KINDS)}."
        )

    panel: ReturnPanelData
    if kind == "factor_regime":
        panel = factor_regime_panel(n_obs=n_obs, n_assets=n_assets, seed=seed)
    elif kind == "learnable_edge":
        panel = learnable_edge_panel(n_obs=n_obs, n_assets=n_assets, seed=seed)
    else:  # "pure_noise"
        panel = pure_noise_panel(n_obs=n_obs, n_assets=n_assets, seed=seed)

    returns = panel.returns
    # A synthetic price panel anchored at 100, consistent with the returns (the API
    # never needs the levels, but the loader contract returns both).
    prices = 100.0 * (1.0 + returns).cumprod()
    return prices.astype("float64"), returns.astype("float64"), "synthetic"


def load_multi_asset_panel(
    tickers: Sequence[str],
    *,
    start: date,
    end: date,
    data_source_pref: str = "synthetic",
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, DataSource]:
    """Load a multi-asset PIT price + return panel and tag the provenance.

    With ``data_source_pref="polygon"`` the vendored Polygon provider is tried first;
    ``"eodhd"`` tries the EODHD reader (behind an optional key); ``"synthetic"``
    (default) and any provider failure fall straight through to the deterministic
    synthetic factor-regime panel so the loader is usable offline and in CI. Returns
    are computed with ``pct_change(fill_method=None)`` (no forward-fill across gaps).

    LAZY IMPORTS: the vendored providers (and ``httpx`` inside them — the ``data``
    extra) are imported inside this function, so importing this module is cheap and
    side-effect-free.

    Parameters
    ----------
    tickers:
        The basket of asset symbols to fetch (e.g. ``["SPY", "TLT", "GLD", "LQD"]``).
    start, end:
        Inclusive date span.
    data_source_pref:
        ``"polygon"`` / ``"eodhd"`` try the real PIT providers first; ``"synthetic"``
        (default) and ``"auto"`` resolve to the synthetic panel.
    seed:
        Master RNG seed for the synthetic fallback panel.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, DataSource]
        The price panel, the per-bar return panel, and the resolved source label.

    Raises
    ------
    ValidationError
        If ``tickers`` is empty / duplicated or ``end <= start``.
    """
    symbols = list(tickers)
    if len(symbols) < 2:
        raise ValidationError("load_multi_asset_panel: need at least two tickers.")
    if len(set(symbols)) != len(symbols):
        raise ValidationError("load_multi_asset_panel: tickers must not contain duplicates.")
    if end <= start:
        raise ValidationError(f"load_multi_asset_panel: end ({end}) must be after start ({start}).")

    if data_source_pref in {"polygon", "eodhd"}:
        fetched = _try_provider_panel(symbols, start=start, end=end, provider=data_source_pref)
        if fetched is not None:
            prices = fetched
            returns = compute_returns(prices)
            source: DataSource = "polygon" if data_source_pref == "polygon" else "eodhd"
            return prices, returns.dropna(how="any"), source
        # Provider failed (no key / no network / data extra absent): fall through.

    # Synthetic fallback / default: deterministic, offline, CI-safe.
    n_obs = max(2, _bdays_between(start, end))
    return synthetic_default_panel(
        n_obs=n_obs, n_assets=len(symbols), seed=seed, kind="factor_regime"
    )


def _bdays_between(start: date, end: date) -> int:
    """Return the number of business days spanning ``[start, end]`` (inclusive-ish)."""
    span = pd.bdate_range(start=pd.Timestamp(start), end=pd.Timestamp(end))
    return int(span.size)


def _try_provider_panel(
    tickers: list[str],
    *,
    start: date,
    end: date,
    provider: str,
) -> pd.DataFrame | None:
    """Best-effort fetch of a basket's PIT closes via the requested provider.

    LAZY IMPORT: the provider (and ``httpx`` inside it — the ``data`` extra) is
    imported here, not at module load. Any failure — missing key, no network, the
    ``data`` extra absent, or an empty payload — returns ``None`` so the caller can
    fall through to the deterministic synthetic panel. ``provider="polygon"`` uses the
    vendored Polygon aggregates API; ``provider="eodhd"`` is reserved for an EODHD
    reader behind an optional key (not yet wired — returns ``None``).
    """
    if provider == "eodhd":
        # The EODHD reader is the optional offline path behind a paid key; the
        # deployed tool never requires it, so the scaffold returns None (synthetic).
        return None
    try:
        from rlallocator.data_providers.polygon import PolygonProvider

        prov = PolygonProvider()
        panel = prov.fetch(tickers, start, end)
    except Exception:  # any provider failure (no key/network/extra) -> synthetic fallback.
        return None

    if panel.empty or not all(t in panel.columns for t in tickers):
        return None
    frame = panel[tickers].astype("float64")
    frame = frame.dropna(how="any")
    return frame if not frame.empty else None
