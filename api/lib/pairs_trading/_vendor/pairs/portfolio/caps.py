"""Hard caps applied to pair weight vectors.

The :func:`apply_caps` projection is the last step before weights are sent to
execution. It enforces:

1. A hard cap on the number of simultaneously held pairs.
2. A per-sector gross-exposure ceiling.
3. A per-asset notional ceiling that accounts for both legs of every pair.

Each binding constraint emits a :class:`~pairs.portfolio.results.CapEvent`
record so that the audit log can be replayed downstream.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from pairs._exceptions import InputError
from pairs.portfolio.results import CapEvent

__all__ = ["apply_caps"]


def _validate(
    weights: pd.Series,
    sector_map: Mapping[str, str],
    asset_legs_map: Mapping[str, Sequence[str]],
) -> None:
    if not isinstance(weights, pd.Series):
        msg = "weights must be a pandas Series"
        raise InputError(msg)
    if weights.isna().any():
        msg = "weights must not contain NaN"
        raise InputError(msg)
    for pid in weights.index:
        if pid not in sector_map:
            msg = f"sector_map missing entry for pair {pid!r}"
            raise InputError(msg)
        if pid not in asset_legs_map:
            msg = f"asset_legs_map missing entry for pair {pid!r}"
            raise InputError(msg)


def apply_caps(
    weights: pd.Series,
    *,
    max_pairs: int = 15,
    max_sector_gross: float = 0.30,
    max_asset_notional: float = 0.10,
    sector_map: Mapping[str, str],
    asset_legs_map: Mapping[str, Sequence[str]],
    asof: pd.Timestamp | None = None,
) -> tuple[pd.Series, list[CapEvent]]:
    """Project ``weights`` so that all cap constraints are satisfied.

    The projection runs in four ordered phases. Each phase logs a
    :class:`CapEvent` per pair whose weight is altered by that phase.

    1. Rank by absolute weight and keep only the top ``max_pairs``.
    2. Iteratively scale per-asset notional. A pair that loads two assets has
       its weight bounded by ``max_asset_notional`` divided by the asset's
       cumulative notional contribution from other pairs.
    3. Iteratively scale per-sector gross exposure to ``max_sector_gross``.
    4. Renormalise the weight vector so that ``sum(|w|) <= 1`` (only scaling
       down; the renormalisation is a no-op when the vector already satisfies
       the budget).

    Parameters
    ----------
    weights : pandas.Series
        Pair-level weight vector. Indexed by pair id.
    max_pairs : int
        Maximum number of pairs to hold simultaneously.
    max_sector_gross : float
        Maximum gross exposure to a single sector, expressed as a fraction of
        the total weight budget (``1.0``).
    max_asset_notional : float
        Maximum per-asset notional, also as a fraction of the budget.
    sector_map : mapping
        ``{pair_id: sector_label}``.
    asset_legs_map : mapping
        ``{pair_id: (asset_a, asset_b)}``.
    asof : pandas.Timestamp, optional
        Timestamp stamped on each emitted :class:`CapEvent`. Defaults to
        ``pd.Timestamp("1970-01-01")`` when ``None`` is passed.

    Returns
    -------
    (pandas.Series, list of CapEvent)
        The projected weights (indexed identically to the input) and the
        ordered audit log of cap events.
    """
    _validate(weights, sector_map, asset_legs_map)
    if int(max_pairs) <= 0:
        msg = f"max_pairs must be positive, got {max_pairs!r}"
        raise InputError(msg)
    if not (0.0 < float(max_sector_gross) <= 1.0):
        msg = f"max_sector_gross must lie in (0, 1], got {max_sector_gross!r}"
        raise InputError(msg)
    if not (0.0 < float(max_asset_notional) <= 1.0):
        msg = f"max_asset_notional must lie in (0, 1], got {max_asset_notional!r}"
        raise InputError(msg)

    stamp = asof if asof is not None else pd.Timestamp("1970-01-01")
    events: list[CapEvent] = []
    out = weights.astype(float).copy()

    # Phase 1: keep top max_pairs by absolute weight.
    abs_sorted = out.abs().sort_values(ascending=False)
    keep = abs_sorted.head(int(max_pairs)).index
    drop = out.index.difference(keep)
    for pid in drop:
        prev = float(out.loc[pid])
        if prev != 0.0:
            out.loc[pid] = 0.0
            events.append(
                CapEvent(
                    asof=stamp,
                    kind="max_pairs",
                    pair_id=str(pid),
                    pre_weight=prev,
                    post_weight=0.0,
                    detail={"max_pairs": int(max_pairs)},
                )
            )

    # Phase 2: per-asset notional cap (iterative until stable).
    for _ in range(50):
        asset_load: dict[str, float] = {}
        for pid, w in out.items():
            if w == 0.0:
                continue
            for asset in asset_legs_map[pid]:
                asset_load[asset] = asset_load.get(asset, 0.0) + abs(float(w))
        binding = {a: load for a, load in asset_load.items() if load > max_asset_notional}
        if not binding:
            break
        scaled_any = False
        for asset, load in binding.items():
            scale = max_asset_notional / load
            for pid in out.index:
                if asset in asset_legs_map[pid] and out.loc[pid] != 0.0:
                    prev = float(out.loc[pid])
                    new = prev * scale
                    if not np.isclose(prev, new):
                        out.loc[pid] = new
                        events.append(
                            CapEvent(
                                asof=stamp,
                                kind="asset_notional",
                                pair_id=str(pid),
                                pre_weight=prev,
                                post_weight=new,
                                detail={"asset": asset, "scale": float(scale)},
                            )
                        )
                        scaled_any = True
        if not scaled_any:
            break

    # Phase 3: per-sector gross cap (iterative).
    for _ in range(50):
        sector_load: dict[str, float] = {}
        for pid, w in out.items():
            if w == 0.0:
                continue
            sector = sector_map[pid]
            sector_load[sector] = sector_load.get(sector, 0.0) + abs(float(w))
        binding = {s: load for s, load in sector_load.items() if load > max_sector_gross}
        if not binding:
            break
        for sector, load in binding.items():
            scale = max_sector_gross / load
            for pid in out.index:
                if sector_map[pid] == sector and out.loc[pid] != 0.0:
                    prev = float(out.loc[pid])
                    new = prev * scale
                    out.loc[pid] = new
                    events.append(
                        CapEvent(
                            asof=stamp,
                            kind="sector_gross",
                            pair_id=str(pid),
                            pre_weight=prev,
                            post_weight=new,
                            detail={"sector": sector, "scale": float(scale)},
                        )
                    )

    # Phase 4: enforce overall budget |w| <= 1.
    gross = float(out.abs().sum())
    if gross > 1.0:
        out = out / gross
    return out, events
