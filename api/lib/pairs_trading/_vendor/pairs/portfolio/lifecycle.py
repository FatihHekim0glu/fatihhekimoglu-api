"""Pair lifecycle management: cooldowns, retests, walk-forward reselections.

Pairs in a stat-arb book do not live forever. They get stopped out, their
underlying cointegration weakens, and the universe is periodically refreshed
by the walk-forward harness. :class:`PairLifecycle` centralises that book-
keeping so the daily loop only has to ask one question: "which pairs may
trade today?".

Lifecycle uses dependency injection for the cointegration retest and the
half-life lookup so the portfolio module does not have a hard dependency on
the cointegration or spread sub-packages at import time.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from pairs._exceptions import InputError

__all__ = ["PairLifecycle"]


@dataclass
class PairLifecycle:
    """Bookkeeping for cooldowns, retests, and walk-forward reselections.

    Attributes
    ----------
    cointegration_retest : callable
        Function called as ``cointegration_retest(pair_id, asof, prices)`` that
        returns an object exposing a ``.cointegrated`` boolean attribute (the
        same interface as :class:`pairs.cointegration.PipelineResult`).
    half_life_lookup : callable
        Function called as ``half_life_lookup(pair_id) -> float`` returning the
        pair's current half-life in trading days.
    min_cooldown_days : int, default ``10``
        Minimum cooldown applied after any stop-out, even when the half-life
        is shorter.
    """

    cointegration_retest: Callable[[str, pd.Timestamp, pd.DataFrame], Any]
    half_life_lookup: Callable[[str], float]
    min_cooldown_days: int = 10
    _stopped_out: dict[str, pd.Timestamp] = field(default_factory=dict)
    _universe: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if int(self.min_cooldown_days) < 0:
            msg = f"min_cooldown_days must be non-negative, got {self.min_cooldown_days!r}"
            raise InputError(msg)

    def _required_cooldown(self, pair_id: str) -> int:
        try:
            hl = float(self.half_life_lookup(pair_id))
        except Exception:
            hl = 0.0
        if not math.isfinite(hl) or hl < 0.0:
            hl = 0.0
        return int(max(int(self.min_cooldown_days), math.ceil(1.0 * hl)))

    def on_stop_out(self, pair_id: str, asof: pd.Timestamp) -> None:
        """Record that ``pair_id`` was stopped out at ``asof``."""
        self._stopped_out[str(pair_id)] = pd.Timestamp(asof)

    def on_walkforward_reselect(
        self,
        new_universe: Iterable[str],
        asof: pd.Timestamp,  # noqa: ARG002
    ) -> None:
        """Refresh the universe of tradable pairs.

        Pairs absent from ``new_universe`` are dropped from the cooldown
        registry. Pairs that survive the reselection have their cooldowns
        cleared so that the new in-sample fit can re-enter without delay.
        """
        self._universe = {str(p) for p in new_universe}
        self._stopped_out = {}

    def cooldown_remaining(self, pair_id: str, asof: pd.Timestamp) -> int:
        """Return the number of trading days the cooldown still has to run."""
        pid = str(pair_id)
        if pid not in self._stopped_out:
            return 0
        elapsed = (pd.Timestamp(asof) - self._stopped_out[pid]).days
        remaining = self._required_cooldown(pid) - elapsed
        return max(0, int(remaining))

    def can_reenter(self, pair_id: str, asof: pd.Timestamp, prices: pd.DataFrame) -> bool:
        """Return ``True`` when the cooldown has elapsed *and* the retest passes."""
        if self.cooldown_remaining(pair_id, asof) > 0:
            return False
        try:
            result = self.cointegration_retest(str(pair_id), pd.Timestamp(asof), prices)
        except Exception:
            return False
        return bool(getattr(result, "cointegrated", False))

    def active_set(
        self,
        candidates: Iterable[str],
        asof: pd.Timestamp,
        prices: pd.DataFrame,
    ) -> set[str]:
        """Return the subset of ``candidates`` cleared to trade at ``asof``."""
        active: set[str] = set()
        for pid in candidates:
            pid_s = str(pid)
            if pid_s in self._stopped_out:
                if self.can_reenter(pid_s, asof, prices):
                    self._stopped_out.pop(pid_s, None)
                    active.add(pid_s)
            else:
                active.add(pid_s)
        return active
