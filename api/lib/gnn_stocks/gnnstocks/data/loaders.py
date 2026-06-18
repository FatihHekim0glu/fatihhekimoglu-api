"""Real-data loaders: Polygon point-in-time universe + per-node features (CLI path).

The default, deployed data path is the seeded synthetic block-factor generator
(:mod:`gnnstocks.data.synthetic`) — no API keys, no survivorship questions, and the
honest NULL holds by construction. This module is the OFFLINE CLI path for real
data, and the HEADLINE leakage guard of the whole tool: the point-in-time universe.

- :func:`pit_universe_as_of` filters a candidate index-membership set to the
  issuers that were ACTUALLY index members AS-OF a fold's date AND actively traded
  that day — never a static current-membership node set (the documented
  eigen-portfolios / Stock-Price-Forecast survivorship failure). A property test
  asserts this drops future-only constituents.
- :func:`load_pit_panel` builds the as-of universe across a date span via the
  vendored Polygon provider, assembles the wide returns panel, and tags the
  provenance (``"polygon"`` / ``"synthetic"``).
- :func:`node_features` derives the per-node return/vol/momentum FEATURE matrix
  from a TRAIN window only (the forward-return label NEVER enters the features).

Heavy data dependencies (httpx via the vendored Polygon provider, pyarrow,
diskcache) live behind the ``data`` extra and are imported LAZILY inside these
functions, so importing this module pulls in nothing heavy and has no side effects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

from gnnstocks._constants import EPS
from gnnstocks._exceptions import InsufficientDataError, ValidationError

# quantcore-candidate: PIT membership mirrors
# fatihhekimoglu-api:lib/polygon/sp500_universe.py (as-of grouped-daily intersection).

#: Where a returns panel ultimately came from. Returned alongside the data so
#: callers (and the API ``data_source`` field) can report provenance.
DataSource = Literal["polygon", "synthetic", "cache"]


def pit_universe_as_of(
    candidate_members: Sequence[str],
    active_tickers: Sequence[str],
    as_of: date,
) -> list[str]:
    """Return the point-in-time universe AS-OF ``as_of`` (the headline survivorship guard).

    A ticker enters the universe for a fold dated ``as_of`` ONLY if it is in the
    candidate index-membership set AND it actively traded on ``as_of`` (i.e. it
    appears in ``active_tickers``, the as-of grouped-daily snapshot). This drops:

    - future-only constituents that had no trading row as-of the date (e.g. a
      company that IPO'd after the fold date), and
    - candidates that were not actually trading that day.

    A static current-membership node set is NEVER used (the documented
    survivorship failure). A property test asserts that perturbing the future
    cannot inject a future-only constituent into a past fold's universe.

    Parameters
    ----------
    candidate_members:
        The candidate index-membership set (e.g. the modern constituents list).
    active_tickers:
        The tickers with an actual grouped-daily trading row on ``as_of``.
    as_of:
        The fold's as-of date (used for provenance / error messages).

    Returns
    -------
    list[str]
        The sorted point-in-time universe for ``as_of`` (the intersection).

    Raises
    ------
    ValidationError
        If ``candidate_members`` is empty.
    """
    if len(candidate_members) == 0:
        raise ValidationError(
            f"pit_universe_as_of: candidate_members must be non-empty (as_of={as_of.isoformat()})."
        )
    # The as-of intersection: a candidate enters the universe ONLY if it actually
    # traded on ``as_of`` (i.e. it appears in the grouped-daily snapshot). This
    # mirrors the vendored SP500UniverseBuilder.get_membership_as_of intersection
    # and structurally cannot admit a future-only constituent: a ticker that had
    # no trading row as-of the date is absent from ``active_tickers`` and dropped.
    active = set(active_tickers)
    return sorted({t for t in candidate_members if t in active})


def load_pit_panel(
    candidate_members: Sequence[str],
    *,
    start: date,
    end: date,
    rebalance: str = "ME",
    data_source_pref: Literal["polygon", "auto"] = "auto",
) -> tuple[pd.DataFrame, dict[date, list[str]], DataSource]:
    """Load a point-in-time returns panel + the per-rebalance as-of universe.

    Builds the as-of universe at each rebalance date in ``[start, end]`` via the
    vendored Polygon provider (:func:`pit_universe_as_of` per date), fetches the
    union of members across the span, computes returns with
    ``pct_change(fill_method=None)``, and tags the provenance. On any failure
    (no key, no network, ``data`` extra absent) it falls back to a deterministic
    synthetic panel so the loader is usable offline and in CI.

    LAZY IMPORTS: the vendored Polygon provider (and ``httpx``) are imported inside
    this function — the ``data`` extra — so importing this module is cheap and
    side-effect-free.

    Parameters
    ----------
    candidate_members:
        The candidate index-membership set.
    start, end:
        Inclusive date span.
    rebalance:
        Pandas offset alias for the rebalance cadence (default ``"ME"`` month-end).
    data_source_pref:
        ``"polygon"`` tries the real PIT provider first; ``"auto"`` falls straight
        through to synthetic when the provider is unavailable.

    Returns
    -------
    tuple[pandas.DataFrame, dict[date, list[str]], DataSource]
        The wide returns panel, the per-rebalance as-of universe map, and the
        resolved data-source label.

    Raises
    ------
    ValidationError
        If ``candidate_members`` is empty or ``end <= start``.
    """
    if len(candidate_members) == 0:
        raise ValidationError("load_pit_panel: candidate_members must be non-empty.")
    if end <= start:
        raise ValidationError(f"load_pit_panel: end ({end}) must be after start ({start}).")

    rebalance_dates = _rebalance_dates(start, end, rebalance)

    if data_source_pref == "polygon":
        try:
            return _load_polygon_pit(
                candidate_members, start=start, end=end, rebalance_dates=rebalance_dates
            )
        except Exception:  # any provider failure falls through to synthetic.
            pass

    return _synthetic_pit_panel(
        candidate_members, start=start, end=end, rebalance_dates=rebalance_dates
    )


def node_features(
    train_returns: pd.DataFrame,
    *,
    momentum_lookback: int = 60,
    vol_lookback: int = 20,
) -> pd.DataFrame:
    """Derive the per-node return/vol/momentum FEATURE matrix from a TRAIN window.

    Builds a ``(n_nodes, n_features)`` cross-sectional feature matrix from the
    TRAIN window only: a short-horizon mean return, a rolling volatility, and a
    momentum (trailing cumulative return). The forward-return LABEL is NEVER one
    of these features (anti-leakage; a property test asserts no-target-in-features),
    and the cross-sectional standardization uses only the contemporaneous
    cross-section (allowed), never the time-series future.

    Parameters
    ----------
    train_returns:
        The TRAIN-window wide returns panel (rows = time, columns = node).
    momentum_lookback:
        Trailing window for the momentum feature.
    vol_lookback:
        Trailing window for the volatility feature.

    Returns
    -------
    pandas.DataFrame
        The ``(n_nodes, n_features)`` feature matrix indexed by node.

    Raises
    ------
    ValidationError
        If ``train_returns`` is empty or a lookback exceeds the window length.
    """
    if train_returns.shape[0] == 0 or train_returns.shape[1] == 0:
        raise ValidationError("node_features: train_returns must be non-empty.")
    if momentum_lookback < 1 or vol_lookback < 1:
        raise ValidationError(
            "node_features: momentum_lookback and vol_lookback must be >= 1 "
            f"(got momentum_lookback={momentum_lookback}, vol_lookback={vol_lookback})."
        )
    n_obs = int(train_returns.shape[0])
    if momentum_lookback > n_obs or vol_lookback > n_obs:
        raise InsufficientDataError(
            f"node_features: train window has {n_obs} observation(s) but momentum_lookback="
            f"{momentum_lookback} / vol_lookback={vol_lookback} require more."
        )

    frame = train_returns.astype("float64")
    nodes = pd.Index([str(c) for c in frame.columns], name="node")

    # All three features are derived from the TRAIN window ONLY; the forward-return
    # label is NEVER among them (anti-leakage — a property test asserts this).
    # 1) Short-horizon mean return over the volatility window (a level signal).
    mean_return = frame.iloc[-vol_lookback:].mean(axis=0).to_numpy(dtype="float64")
    # 2) Rolling volatility: cross-sectional std over the same trailing window.
    volatility = frame.iloc[-vol_lookback:].std(axis=0, ddof=1).to_numpy(dtype="float64")
    # 3) Momentum: trailing cumulative (compounded) return over the momentum window.
    momentum_window = frame.iloc[-momentum_lookback:]
    momentum = (np.prod(1.0 + momentum_window.to_numpy(dtype="float64"), axis=0) - 1.0).astype(
        "float64"
    )

    raw = np.column_stack([mean_return, volatility, momentum])
    # Cross-sectional standardization at THIS rebalance (uses only the
    # contemporaneous cross-section across nodes — allowed; never the time-series
    # future). Guard the std with EPS so a degenerate column stays finite/zero.
    centered = raw - raw.mean(axis=0, keepdims=True)
    std = raw.std(axis=0, ddof=0, keepdims=True)
    std = np.where(std < EPS, 1.0, std)
    standardized = centered / std

    return pd.DataFrame(
        standardized,
        index=nodes,
        columns=pd.Index(["mean_return", "volatility", "momentum"], name="feature"),
        dtype="float64",
    )


def forward_returns(panel: pd.DataFrame, *, horizon: int = 1) -> pd.DataFrame:
    """Return the per-node ``horizon``-step-ahead forward returns (LABELS ONLY).

    The forward return at time ``t`` for node ``i`` is the cumulative return over
    ``[t+1, t+horizon]``; it is the prediction TARGET and NEVER enters the node
    feature matrix. Computed with ``pct_change(fill_method=None)`` semantics so no
    synthetic zero-returns are injected across gaps.

    Parameters
    ----------
    panel:
        The wide returns panel (rows = time, columns = node).
    horizon:
        The forward-return horizon (1 or 5).

    Returns
    -------
    pandas.DataFrame
        The forward-return label panel, aligned and trailing-trimmed by horizon.

    Raises
    ------
    ValidationError
        If ``panel`` is empty or ``horizon < 1``.
    """
    if panel.shape[0] == 0 or panel.shape[1] == 0:
        raise ValidationError("forward_returns: panel must be non-empty.")
    if horizon < 1:
        raise ValidationError(f"forward_returns: horizon must be >= 1, got {horizon}.")

    returns = panel.astype("float64")
    # Compound the next ``horizon`` single-period returns into a forward return,
    # then shift it back to label time ``t`` (so row ``t`` carries the return over
    # ``[t+1, t+horizon]``). NaNs from ``pct_change(fill_method=None)``-style gaps
    # propagate honestly rather than being forward-filled into spurious zeros.
    gross = 1.0 + returns
    forward = gross.rolling(window=horizon).apply(np.prod, raw=True) - 1.0
    forward = forward.shift(-horizon)
    # The trailing ``horizon`` rows have no realized future and are dropped.
    return forward.iloc[: len(forward) - horizon]


def synthetic_universe_map(
    n_nodes: int,
    rebalance_dates: Sequence[date],
    *,
    candidate_members: Mapping[date, Sequence[str]] | None = None,
) -> dict[date, list[str]]:
    """Build a fixed-node-set as-of universe map for the synthetic default.

    The synthetic default has a FIXED node set (it is not survivorship-sensitive),
    so the as-of universe is the same node list at every rebalance date — but the
    loader path still flows through the same as-of API surface so the PIT contract
    is uniform across synthetic and real data.

    Parameters
    ----------
    n_nodes:
        The fixed synthetic node count.
    rebalance_dates:
        The rebalance dates to key the universe map by.
    candidate_members:
        Optional per-date override (unused for the fixed synthetic set; present
        for API parity with the real PIT path).

    Returns
    -------
    dict[date, list[str]]
        The per-date universe map (identical node list at every date).

    Raises
    ------
    ValidationError
        If ``n_nodes < 1`` or ``rebalance_dates`` is empty.
    """
    if n_nodes < 1:
        raise ValidationError(f"synthetic_universe_map: n_nodes must be >= 1, got {n_nodes}.")
    if len(rebalance_dates) == 0:
        raise ValidationError("synthetic_universe_map: rebalance_dates must be non-empty.")

    from gnnstocks.data.synthetic import _node_names

    universe_map: dict[date, list[str]] = {}
    for rebalance_date in rebalance_dates:
        members = (
            list(candidate_members[rebalance_date])
            if candidate_members is not None and rebalance_date in candidate_members
            else _node_names(n_nodes)
        )
        universe_map[rebalance_date] = members
    return universe_map


# --------------------------------------------------------------------------- #
# Private helpers: rebalance grid + the Polygon-PIT loader + synthetic fallback #
# --------------------------------------------------------------------------- #
def _rebalance_dates(start: date, end: date, rebalance: str) -> list[date]:
    """Return the rebalance anchor dates in ``[start, end]`` (pandas offset alias).

    Mirrors the vendored ``SP500UniverseBuilder.get_membership_window`` cadence: a
    pandas ``date_range`` on ``rebalance`` (default month-end). When the span is too
    short to contain a single anchor, the endpoints are used so callers always get a
    non-empty grid.
    """
    anchors = pd.date_range(start=start, end=end, freq=rebalance)
    if len(anchors) == 0:
        anchors = pd.DatetimeIndex([pd.Timestamp(start), pd.Timestamp(end)]).unique()
    return [ts.date() for ts in anchors]


def _load_polygon_pit(
    candidate_members: Sequence[str],
    *,
    start: date,
    end: date,
    rebalance_dates: Sequence[date],
) -> tuple[pd.DataFrame, dict[date, list[str]], DataSource]:
    """Fetch the real Polygon-PIT panel + as-of universe map. May raise (caller falls back).

    LAZY IMPORT: the vendored Polygon provider (and ``httpx`` inside it — the
    ``data`` extra) is imported here, never at module import time. For each
    rebalance date the as-of universe is the intersection of ``candidate_members``
    with the tickers actively trading that day (:func:`pit_universe_as_of`); the
    union of those universes is fetched and differenced into returns with
    ``pct_change(fill_method=None)`` (no forward-fill across gaps).
    """
    from gnnstocks.data_providers.polygon import PolygonProvider

    provider = PolygonProvider()
    candidates = list(candidate_members)

    universe_map: dict[date, list[str]] = {}
    for as_of in rebalance_dates:
        active = _active_tickers_as_of(provider, candidates, as_of)
        universe_map[as_of] = pit_universe_as_of(candidates, active, as_of)

    members_union = sorted({t for members in universe_map.values() for t in members})
    if not members_union:
        raise InsufficientDataError(
            "load_pit_panel: the point-in-time universe is empty across all rebalance dates."
        )

    prices = provider.fetch(members_union, start, end)
    if prices.empty or bool(prices.isna().all(axis=None)):
        raise InsufficientDataError("load_pit_panel: Polygon returned no usable price data.")

    returns = prices.pct_change(fill_method=None).iloc[1:]
    return returns.astype("float64"), universe_map, "polygon"


def _active_tickers_as_of(
    provider: object, candidates: Sequence[str], as_of: date
) -> list[str]:
    """Return the candidate tickers that have a real price bar on ``as_of``.

    Probes the provider for a one-row window around ``as_of``; a candidate counts as
    active only if it actually returned a bar (the as-of grouped-daily intersection).
    Any provider/parse failure for a ticker treats it as inactive that day. This is
    the seam that makes the as-of filter bite: a future-only constituent simply has
    no bar and is dropped.
    """
    import datetime as _dt

    probe_end = as_of + _dt.timedelta(days=1)
    active: list[str] = []
    for ticker in candidates:
        try:
            frame = provider.fetch([ticker], as_of, probe_end)  # type: ignore[attr-defined]
        except Exception:  # a missing/failed bar means "inactive that day".
            continue
        if not frame.empty and not bool(frame.isna().all(axis=None)):
            active.append(ticker)
    return active


def _synthetic_pit_panel(
    candidate_members: Sequence[str],
    *,
    start: date,
    end: date,
    rebalance_dates: Sequence[date],
) -> tuple[pd.DataFrame, dict[date, list[str]], DataSource]:
    """Deterministic synthetic-PIT fallback over the same date span (offline/CI path).

    Builds a seeded block-factor returns panel (the honest-null DGP) whose node
    columns are the candidate tickers, reindexed onto the business days of
    ``[start, end]``, and a fixed-node-set as-of universe map (the synthetic set is
    not survivorship-sensitive). Flows through the SAME as-of API surface so the PIT
    contract is uniform across synthetic and real data. Seeded off the candidate set
    + span so the same request always yields a byte-identical panel.
    """
    from gnnstocks.data.synthetic import block_factor_panel

    nodes = list(candidate_members)
    n_nodes = len(nodes)
    index = pd.bdate_range(start=start, end=end)
    n_obs = max(2, len(index))

    n_blocks = max(1, min(5, n_nodes))
    # Deterministic seed from the request (candidate set + span), masked to 31 bits.
    seed = hash((tuple(nodes), start.isoformat(), end.isoformat())) & 0x7FFFFFFF
    panel = block_factor_panel(
        n_nodes=n_nodes,
        n_blocks=n_blocks,
        n_obs=n_obs,
        seed=seed,
        start=start.isoformat(),
    )
    returns = panel.returns.copy()
    returns.columns = pd.Index(nodes)
    # Align onto the requested business-day calendar (trimming any generated tail).
    if len(index) > 0:
        returns = returns.iloc[: len(index)]
        returns.index = index[: len(returns)]

    universe_map = synthetic_universe_map(
        n_nodes, list(rebalance_dates), candidate_members=None
    )
    # The synthetic universe uses the candidate tickers as the fixed node set.
    universe_map = {d: nodes for d in universe_map}
    return returns.astype("float64"), universe_map, "synthetic"
