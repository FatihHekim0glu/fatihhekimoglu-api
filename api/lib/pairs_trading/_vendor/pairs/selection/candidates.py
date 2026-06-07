"""Candidate pair generation.

Three modes are supported:

* ``"curated"`` -- load a hand-built pair universe (default
  ``"curated_25_v1"``) via :func:`pairs.data.load_pair_universe` and return
  one :class:`Candidate` per entry.
* ``"within_sector"`` -- enumerate the Cartesian C(n, 2) pairs within each
  GICS sector group supplied via the ``sector_map`` kwarg.
* ``"within_sub_industry"`` -- same as ``"within_sector"`` but operates on a
  sub-industry mapping.

Because point-in-time GICS data is out of scope for v1, ``sector_map``
defaults to ``None`` which lumps every ticker into a single ``"unknown"``
bucket. Callers wanting genuine intra-sector enumeration must supply an
explicit ``{ticker: group}`` dict. The function emits a warning and refuses
to run when more than 1500 pairs would be produced unless ``force=True``.
"""

from __future__ import annotations

import logging
import warnings
from itertools import combinations
from typing import TYPE_CHECKING, Literal

import pandas as pd

from pairs._exceptions import InputError
from pairs.data import is_blacklisted, load_pair_universe
from pairs.selection.results import Candidate

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

__all__ = ["generate_candidates"]

_HARD_CAP: int = 1500
_CURATED_UNIVERSE_DEFAULT: str = "curated_25_v1"


def _ticker_universe(prices: pd.DataFrame) -> list[str]:
    """Return uppercase column tickers from a price panel."""
    return [str(t).upper() for t in prices.columns]


def _filter_blacklisted(tickers: Iterable[str], asof: pd.Timestamp) -> list[str]:
    """Drop blacklisted tickers as of ``asof``."""
    return [t for t in tickers if not is_blacklisted(t, asof)]


def _from_curated(asof: pd.Timestamp, name: str) -> list[Candidate]:
    """Materialise candidates from a curated pair universe.

    Pairs whose either leg is blacklisted at ``asof`` are filtered out and
    logged at DEBUG level.
    """
    universe = load_pair_universe(name)
    out: list[Candidate] = []
    for spec in universe.pairs:
        if is_blacklisted(spec.a, asof) or is_blacklisted(spec.b, asof):
            logger.debug("Skipping curated pair %s/%s: blacklisted at %s", spec.a, spec.b, asof)
            continue
        out.append(Candidate(ticker_a=spec.a, ticker_b=spec.b))
    return out


def _enumerate_within_group(
    tickers: list[str],
    sector_map: dict[str, str] | None,
    attribute: Literal["sector", "sub_industry"],
) -> list[Candidate]:
    """Enumerate combinations within each group of the given attribute."""
    groups: dict[str, list[str]] = {}
    for ticker in tickers:
        group = sector_map.get(ticker, "unknown") if sector_map else "unknown"
        groups.setdefault(group, []).append(ticker)

    out: list[Candidate] = []
    for group, members in groups.items():
        members_sorted = sorted(set(members))
        for a, b in combinations(members_sorted, 2):
            if attribute == "sector":
                out.append(Candidate(ticker_a=a, ticker_b=b, sector=group))
            else:
                out.append(Candidate(ticker_a=a, ticker_b=b, sub_industry=group))
    return out


def generate_candidates(
    universe: str,
    prices: pd.DataFrame,
    *,
    mode: Literal["curated", "within_sector", "within_sub_industry"],
    asof: pd.Timestamp,
    sector_map: dict[str, str] | None = None,
    force: bool = False,
) -> list[Candidate]:
    """Generate candidate pairs for a screening run.

    Parameters
    ----------
    universe
        Universe identifier. For ``mode="curated"`` this names the YAML
        pair universe to load (e.g. ``"curated_25_v1"``). For the other
        modes it is informational; the function uses ``prices.columns`` to
        determine the eligible ticker set.
    prices
        Wide DataFrame indexed by date with tickers as columns. The column
        index defines the eligible ticker universe for enumeration modes
        and is ignored for ``"curated"``.
    mode
        One of ``"curated"``, ``"within_sector"``, ``"within_sub_industry"``.
    asof
        Snapshot date used to consult the corporate-action blacklist.
        Pairs whose either leg is blacklisted at this date are dropped.
    sector_map
        Optional ``{ticker: group}`` mapping. ``None`` (v1 default) groups
        every ticker under ``"unknown"`` so enumeration modes produce the
        full ``C(n, 2)`` set.
    force
        Bypass the ``n > 1500`` hard cap that exists to prevent runaway
        screens. Default ``False`` -- in that case the function raises
        :class:`InputError` when the cap would be exceeded.

    Returns
    -------
    list[Candidate]
        Freshly constructed candidates. The order is deterministic given
        the inputs but is otherwise unspecified.

    Raises
    ------
    InputError
        If ``mode`` is unknown, ``prices`` is not a DataFrame, or the cap
        is exceeded without ``force=True``.

    Notes
    -----
    The ``sector_map`` parameter is a temporary v1 hack -- real point-in-time
    GICS data is not yet wired in. Callers may pre-compute a snapshot via
    an external data provider and pass it in here without coupling.
    """
    if not isinstance(prices, pd.DataFrame):
        msg = "prices must be a pandas DataFrame"
        raise InputError(msg)
    asof = pd.Timestamp(asof)

    if mode == "curated":
        return _from_curated(asof, universe or _CURATED_UNIVERSE_DEFAULT)

    if mode not in {"within_sector", "within_sub_industry"}:
        msg = (
            f"unknown mode {mode!r}; expected one of "
            "'curated', 'within_sector', 'within_sub_industry'"
        )
        raise InputError(msg)

    tickers = _filter_blacklisted(_ticker_universe(prices), asof)
    attribute: Literal["sector", "sub_industry"] = (
        "sector" if mode == "within_sector" else "sub_industry"
    )
    candidates = _enumerate_within_group(tickers, sector_map, attribute)

    if len(candidates) > _HARD_CAP and not force:
        msg = (
            f"refusing to emit {len(candidates)} candidates "
            f"(> hard cap {_HARD_CAP}); pass force=True to override"
        )
        warnings.warn(msg, stacklevel=2)
        raise InputError(msg)
    if len(candidates) > _HARD_CAP:
        warnings.warn(
            f"emitting {len(candidates)} candidates exceeds the {_HARD_CAP} cap; "
            "screening cost will be substantial",
            stacklevel=2,
        )
    return candidates
