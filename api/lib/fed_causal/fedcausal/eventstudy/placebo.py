"""Placebo-date randomization — the PRIMARY significance source.

The honest significance of the observed CAR is its percentile against a null
distribution built by re-running the SAME event study on random NON-event dates:

1. Draw ``n_placebo`` placebo "announcement" dates uniformly from the trading-day
   grid, EXCLUDING every real event window (plus a buffer) so no real effect can
   contaminate the null.
2. For each placebo date, build leakage-safe windows, fit the estimation-window
   market model, and compute the cross-sectional mean CAR.
3. The observed CAR's percentile within this placebo distribution is the honest
   significance: a real, exploitable effect sits far in the tail; noise sits near
   the middle (the percentile is ~uniform).

LEAKAGE GUARD (tested): placebo dates exclude all real event windows. On a
no-effect panel the observed-CAR percentile is ~uniform.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import TYPE_CHECKING, Any

import numpy as np

from fedcausal._constants import (
    DEFAULT_ESTIMATION_GAP,
    DEFAULT_ESTIMATION_WINDOW,
)
from fedcausal._exceptions import EventCalendarError, InsufficientDataError
from fedcausal._rng import make_rng
from fedcausal.events.windows import build_windows
from fedcausal.eventstudy.abnormal import cumulative_abnormal_returns

if TYPE_CHECKING:
    import pandas as pd

    from fedcausal._typing import ModelKind

#: Extra trading-day buffer placed around each real event window when carving the
#: forbidden set, so a placebo window cannot brush a real event window.
_PLACEBO_BUFFER: int = 1


@dataclass(frozen=True, slots=True)
class PlaceboResult:
    """The placebo null distribution and the observed CAR's place within it.

    Attributes
    ----------
    observed_car:
        The observed cross-sectional mean CAR over the real events.
    placebo_cars:
        The null distribution of cross-sectional mean CARs from placebo dates.
    percentile:
        The percentile (0-100) of ``abs(observed_car)`` within the placebo
        distribution of ``abs`` CARs — the honest two-sided significance.
    p_value:
        ``1 - percentile/100`` (the placebo tail probability); the PRIMARY
        significance figure for the verdict.
    n_placebo:
        The number of valid placebo draws actually used.
    """

    observed_car: float
    placebo_cars: np.ndarray
    percentile: float
    p_value: float
    n_placebo: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable summary ``dict`` (excludes the full draw array)."""
        payload = asdict(self)
        payload.pop("placebo_cars", None)
        payload["observed_car"] = float(self.observed_car)
        payload["percentile"] = float(self.percentile)
        payload["p_value"] = float(self.p_value)
        payload["n_placebo"] = int(self.n_placebo)
        return payload


def _forbidden_positions(
    grid: pd.DatetimeIndex,
    event_dates: list[date],
    *,
    event_half_width: int,
    estimation_window: int,
    estimation_gap: int,
) -> set[int]:
    """Return the set of grid positions a placebo anchor may NOT take.

    A position is forbidden when:

    - its event window (plus a buffer) would overlap any REAL event window
      (the leakage guard — no real effect may leak into the null), OR
    - it lacks a complete pre-event estimation window (leading burn-in), OR
    - its event window would run off the end of the grid (trailing pad).
    """
    n = len(grid)
    forbidden: set[int] = set()

    # Leading burn-in: every anchor needs estimation_window + gap + k days of
    # pre-event history (mirrors fedcausal.events.windows.build_windows).
    burn_in = estimation_window + estimation_gap + event_half_width
    forbidden.update(range(0, min(burn_in, n)))

    # Trailing pad: the event window must fit fully on the grid.
    tail_start = max(0, n - event_half_width)
    forbidden.update(range(tail_start, n))

    # Exclude every real event window (plus a buffer) so no real effect bleeds in.
    half = event_half_width + _PLACEBO_BUFFER
    import pandas as pd

    for event_date in event_dates:
        pos = int(grid.searchsorted(pd.Timestamp(event_date), side="left"))
        lo = max(0, pos - half)
        hi = min(n, pos + half + 1)
        forbidden.update(range(lo, hi))

    return forbidden


def sample_placebo_dates(
    grid: pd.DatetimeIndex,
    event_dates: list[date],
    *,
    n_placebo: int,
    event_half_width: int,
    estimation_window: int = DEFAULT_ESTIMATION_WINDOW,
    estimation_gap: int = DEFAULT_ESTIMATION_GAP,
    seed: int = 7,
) -> list[date]:
    """Draw placebo dates that EXCLUDE every real event window (leakage guard).

    Builds the set of forbidden trading days (every real event window plus a
    buffer, and the leading burn-in that lacks pre-event history) and samples
    ``n_placebo`` dates uniformly from the remaining eligible days, seeded.

    Parameters
    ----------
    grid:
        The sorted trading-day index of the panel.
    event_dates:
        The real FOMC announcement dates to exclude.
    n_placebo:
        Number of placebo dates to draw.
    event_half_width:
        The event half-width ``k`` (defines each forbidden window's size).
    estimation_window:
        Estimation-window length (to exclude the leading burn-in).
    estimation_gap:
        The estimation/event gap.
    seed:
        Master RNG seed for the draw.

    Returns
    -------
    list[date]
        The sampled placebo dates (all outside every real event window). Sampling
        is WITHOUT replacement when enough eligible days exist, else WITH
        replacement (so the requested ``n_placebo`` is always honoured).

    Raises
    ------
    EventCalendarError
        If no eligible non-event dates remain at all.
    """
    if n_placebo < 1:
        raise EventCalendarError(f"n_placebo must be >= 1, got {n_placebo}.")

    n = len(grid)
    forbidden = _forbidden_positions(
        grid,
        event_dates,
        event_half_width=event_half_width,
        estimation_window=estimation_window,
        estimation_gap=estimation_gap,
    )
    eligible = np.array([i for i in range(n) if i not in forbidden], dtype=np.int64)
    if eligible.size == 0:
        raise EventCalendarError(
            "no eligible non-event dates remain to draw placebo anchors "
            "(every position is forbidden by the burn-in / event-window guards)."
        )

    rng = make_rng(seed)
    replace = eligible.size < n_placebo
    chosen = rng.choice(eligible, size=n_placebo, replace=replace)
    return [grid[int(pos)].date() for pos in chosen]


def _placebo_anchor_car(
    returns: pd.DataFrame,
    market: pd.Series,
    anchor_date: date,
    *,
    event_half_width: int,
    estimation_window: int,
    estimation_gap: int,
    model: ModelKind,
) -> float | None:
    """Compute the cross-sectional mean CAR for one placebo anchor (or ``None``).

    Returns ``None`` when the anchor lacks usable history (already excluded by the
    forbidden-set guard, but kept defensive so a single bad draw never aborts the
    run).
    """
    grid = returns.index
    try:
        windows = build_windows(
            grid,  # type: ignore[arg-type]
            anchor_date,
            event_half_width=event_half_width,
            estimation_window=estimation_window,
            estimation_gap=estimation_gap,
        )
    except (InsufficientDataError, EventCalendarError):  # pragma: no cover - guarded
        return None
    result = cumulative_abnormal_returns(returns, market, windows, model=model)
    car_values = result.car.to_numpy(dtype=np.float64)
    finite = car_values[np.isfinite(car_values)]
    return float(finite.mean()) if finite.size else None


def _block_means(anchors: np.ndarray, block: int) -> np.ndarray:
    """Average ``anchors`` in consecutive blocks of size ``block`` (drop empties).

    Each block mirrors the observed "mean over ``n_events``" statistic. A trailing
    partial block is averaged on its own so no valid placebo anchor is wasted.
    """
    if block <= 1:
        return anchors
    n_full = anchors.size // block
    means: list[float] = []
    if n_full:
        reshaped = anchors[: n_full * block].reshape(n_full, block)
        means.extend(reshaped.mean(axis=1).tolist())
    remainder = anchors[n_full * block :]
    if remainder.size:
        means.append(float(remainder.mean()))
    return np.asarray(means, dtype=np.float64)


def placebo_distribution(
    returns: pd.DataFrame,
    market: pd.Series,
    event_dates: list[date],
    observed_car: float,
    *,
    n_placebo: int,
    event_half_width: int = 1,
    estimation_window: int = DEFAULT_ESTIMATION_WINDOW,
    estimation_gap: int = DEFAULT_ESTIMATION_GAP,
    model: ModelKind = "market",
    seed: int = 7,
) -> PlaceboResult:
    """Build the placebo null and locate the observed CAR's percentile.

    Re-runs the cross-sectional mean CAR on each placebo date (using the same
    leakage-safe windowing and estimation-window-only model) and returns the
    observed CAR's two-sided percentile/p-value within that null.

    Parameters
    ----------
    returns:
        Wide panel of single-name simple returns.
    market:
        The market-factor return series.
    event_dates:
        The real FOMC announcement dates (excluded from the placebo draw).
    observed_car:
        The observed cross-sectional mean CAR over the real events.
    n_placebo:
        Number of placebo draws.
    event_half_width:
        The event half-width ``k``.
    estimation_window:
        Estimation-window length.
    estimation_gap:
        The estimation/event gap.
    model:
        The expected-return model.
    seed:
        Master RNG seed.

    Returns
    -------
    PlaceboResult
        The null distribution and the observed CAR's percentile/p-value.

    Notes
    -----
    The observed statistic is the cross-sectional mean CAR averaged over the
    ``len(event_dates)`` REAL events. To compare like with like, EACH placebo
    draw averages the cross-sectional mean CAR over a block of the SAME number of
    placebo anchors, so the null has the same variance scaling as the observed
    statistic (a single-date placebo would be far noisier and bias the percentile
    high). On a no-effect panel this makes the observed percentile ~uniform.

    Raises
    ------
    EventCalendarError
        If no eligible placebo dates remain.
    """
    grid = returns.index
    n_events = max(1, len(event_dates))
    # Draw a block of ``n_events`` placebo anchors per placebo replicate so each
    # draw mirrors the observed "mean over n_events" statistic.
    placebo_dates = sample_placebo_dates(
        grid,  # type: ignore[arg-type]
        event_dates,
        n_placebo=n_placebo * n_events,
        event_half_width=event_half_width,
        estimation_window=estimation_window,
        estimation_gap=estimation_gap,
        seed=seed,
    )

    per_anchor: list[float] = []
    for placebo_date in placebo_dates:
        car = _placebo_anchor_car(
            returns,
            market,
            placebo_date,
            event_half_width=event_half_width,
            estimation_window=estimation_window,
            estimation_gap=estimation_gap,
            model=model,
        )
        if car is not None:
            per_anchor.append(car)

    anchors = np.asarray(per_anchor, dtype=np.float64)
    if anchors.size == 0:  # pragma: no cover - defensive only
        raise EventCalendarError("no valid placebo CARs could be computed.")

    # Average consecutive blocks of ``n_events`` anchors into one placebo draw,
    # matching the observed statistic's structure. A trailing partial block (if
    # some anchors were skipped) is averaged on its own so no draw is wasted.
    draws = _block_means(anchors, n_events)

    # Two-sided percentile: where does |observed| fall within the |placebo| null?
    obs = abs(float(observed_car))
    abs_draws = np.abs(draws)
    # Fraction of placebo |CAR| at or below the observed |CAR| -> percentile.
    percentile = float(np.mean(abs_draws <= obs) * 100.0)
    # Two-sided tail probability: fraction of placebo draws AT LEAST as extreme.
    p_value = float(np.mean(abs_draws >= obs))
    p_value = min(max(p_value, 0.0), 1.0)

    return PlaceboResult(
        observed_car=float(observed_car),
        placebo_cars=draws,
        percentile=percentile,
        p_value=p_value,
        n_placebo=int(draws.size),
    )
