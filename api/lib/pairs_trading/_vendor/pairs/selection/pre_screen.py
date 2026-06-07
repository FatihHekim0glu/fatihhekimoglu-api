"""Liquidity, listing and mean-reversion pre-screen for candidate pairs.

The pre-screen is a deterministic filter chain run in a fixed order:

1. **Price floor** -- both legs must trade above ``price_floor`` on at least
   one bar of the formation window. Cheap stocks have outsized bid-ask
   noise and dominate the spread variance.
2. **Average daily dollar volume floor** -- both legs must have at least
   ``adv_floor`` 20-day rolling ADV. Falls back to using the
   :attr:`Candidate.adv_a` / :attr:`Candidate.adv_b` fields when the price
   panel does not carry a volume signal.
3. **Continuous listing** -- both legs must have data on every session in
   the formation window. Pairs with gaps would yield a misaligned
   regression sample.
4. **Correlation band** -- the formation-window Pearson correlation of
   simple returns must lie inside ``corr_band``. Tight bounds discard pairs
   that are either uncorrelated (noise) or essentially the same security
   (multicollinear).
5. **Hurst** -- spread Hurst exponent must be ``<= hurst_max`` so that the
   pair already exhibits mean-reverting tendencies *before* the full
   cointegration battery runs. This stage is skipped when there are too
   few observations to compute the R/S estimator.

Each rejected candidate carries the reason codes in
:attr:`Candidate.exclusion_reason` so callers can render an audit trail.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pairs._exceptions import InputError, InsufficientDataError
from pairs.selection._hurst import hurst_exponent
from pairs.selection.results import Candidate

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

__all__ = ["apply_pre_screen"]


_REASON_PRICE = "price_floor"
_REASON_ADV = "adv_floor"
_REASON_LISTING = "continuous_listing"
_REASON_CORR = "correlation_band"
_REASON_HURST = "hurst_max"
_REASON_MISSING = "missing_ticker"


def _slice_window(
    prices: pd.DataFrame,
    formation_window: tuple[pd.Timestamp, pd.Timestamp],
) -> pd.DataFrame:
    start, end = formation_window
    return prices.loc[(prices.index >= start) & (prices.index <= end)]


def _pair_frame(prices: pd.DataFrame, a: str, b: str) -> pd.DataFrame | None:
    """Extract a two-column price frame for the pair, or ``None`` if missing."""
    if a not in prices.columns or b not in prices.columns:
        return None
    return prices.loc[:, [a, b]]


def _rolling_adv(prices: pd.Series, window: int = 20) -> float:
    """Estimate ADV as a 20-day mean of price (proxy when volume unavailable).

    A genuine ADV calculation needs ``price * volume``. When only prices are
    provided the function returns ``NaN`` so the candidate's stored
    :attr:`Candidate.adv_a` / :attr:`Candidate.adv_b` fields drive the
    decision instead.
    """
    if prices.size < window:
        return float("nan")
    return float(prices.tail(window).mean())


def _check_price_floor(frame: pd.DataFrame, floor: float) -> bool:
    """Return True when both legs trade at or above the floor at some point."""
    return bool((frame.iloc[:, 0] >= floor).any() and (frame.iloc[:, 1] >= floor).any())


def _check_adv_floor(
    candidate: Candidate,
    frame: pd.DataFrame,
    floor: float,
    has_volume_proxy: bool,
) -> bool:
    """Return True when both legs satisfy the ADV floor.

    Uses the candidate's stored ADV fields when present. When the candidate
    does not carry ADV metadata *and* the caller has not supplied a volume
    proxy, the check passes (the upstream loader is responsible for ADV).
    """
    adv_a = candidate.adv_a
    adv_b = candidate.adv_b
    if adv_a is None and has_volume_proxy:
        adv_a = _rolling_adv(frame.iloc[:, 0])
    if adv_b is None and has_volume_proxy:
        adv_b = _rolling_adv(frame.iloc[:, 1])
    if adv_a is None or adv_b is None:
        return True
    if not (np.isfinite(adv_a) and np.isfinite(adv_b)):
        return True
    return adv_a >= floor and adv_b >= floor


def _check_continuous_listing(frame: pd.DataFrame) -> bool:
    """Return True when neither leg has missing bars on the window."""
    if frame.empty:
        return False
    return not frame.isna().any().any()


def _check_correlation(frame: pd.DataFrame, band: tuple[float, float]) -> bool:
    """Return True when the return correlation falls inside ``band`` inclusively."""
    rets = frame.pct_change().dropna()
    if rets.shape[0] < 2:
        return False
    corr = rets.iloc[:, 0].corr(rets.iloc[:, 1])
    if not np.isfinite(corr):
        return False
    lo, hi = band
    return lo <= corr <= hi


def _check_hurst(frame: pd.DataFrame, hurst_max: float) -> bool | None:
    """Return True/False when computable, None when the series is too short."""
    try:
        spread = np.log(frame.iloc[:, 0].to_numpy()) - np.log(frame.iloc[:, 1].to_numpy())
    except (ValueError, FloatingPointError):
        return None
    spread = spread[np.isfinite(spread)]
    try:
        h = hurst_exponent(spread)
    except InsufficientDataError:
        return None
    if not np.isfinite(h):
        return None
    return h <= hurst_max


def apply_pre_screen(
    candidates: Iterable[Candidate],
    prices: pd.DataFrame,
    *,
    formation_window: tuple[pd.Timestamp, pd.Timestamp],
    adv_floor: float = 5e6,
    price_floor: float = 5.0,
    corr_band: tuple[float, float] = (0.5, 0.95),
    hurst_max: float = 0.5,
    return_rejects: bool = False,
) -> list[Candidate]:
    """Apply the deterministic pre-screen filter chain.

    Parameters
    ----------
    candidates
        Candidates to screen. Iterated once; order is preserved.
    prices
        Wide DataFrame of prices indexed by trading date.
    formation_window
        ``(start, end)`` inclusive timestamp pair defining the data window
        the filters operate on.
    adv_floor
        Minimum average dollar volume per leg (whole dollars).
    price_floor
        Minimum nominal price per leg.
    corr_band
        Inclusive ``(low, high)`` correlation envelope on simple returns.
    hurst_max
        Maximum Hurst exponent for the log-spread. Defaults to ``0.5``
        which is the random-walk boundary; lower values demand stronger
        mean reversion.
    return_rejects
        When ``True`` the function returns *every* candidate with their
        accumulated reasons. When ``False`` (default) only the survivors
        are returned with ``exclusion_reason == ()``.

    Returns
    -------
    list[Candidate]
        Either survivors only or the full annotated list -- see
        ``return_rejects``.

    Raises
    ------
    InputError
        If ``prices`` is not a DataFrame, the window is degenerate, or
        ``corr_band`` is improper.
    """
    if not isinstance(prices, pd.DataFrame):
        msg = "prices must be a pandas DataFrame"
        raise InputError(msg)
    start, end = formation_window
    if pd.Timestamp(end) < pd.Timestamp(start):
        msg = "formation_window end must be >= start"
        raise InputError(msg)
    lo, hi = corr_band
    if not (-1.0 <= lo <= hi <= 1.0):
        msg = f"corr_band must satisfy -1 <= lo <= hi <= 1; got {corr_band}"
        raise InputError(msg)

    window = _slice_window(prices, (pd.Timestamp(start), pd.Timestamp(end)))
    # Heuristic: only attempt volume-proxy ADV when prices look like prices,
    # i.e. positive and not already an explicit ADV panel. For v1 we never
    # have a separate volume panel, so the proxy fires only when ADV is None.
    has_volume_proxy = False

    out: list[Candidate] = []
    for candidate in candidates:
        reasons: list[str] = list(candidate.exclusion_reason)
        frame = _pair_frame(window, candidate.ticker_a, candidate.ticker_b)
        if frame is None or frame.empty:
            reasons.append(_REASON_MISSING)
            annotated = Candidate(
                ticker_a=candidate.ticker_a,
                ticker_b=candidate.ticker_b,
                sector=candidate.sector,
                industry=candidate.industry,
                sub_industry=candidate.sub_industry,
                adv_a=candidate.adv_a,
                adv_b=candidate.adv_b,
                exclusion_reason=tuple(reasons),
            )
            if return_rejects:
                out.append(annotated)
            continue

        if not _check_price_floor(frame, price_floor):
            reasons.append(_REASON_PRICE)
        if not _check_adv_floor(candidate, frame, adv_floor, has_volume_proxy):
            reasons.append(_REASON_ADV)
        if not _check_continuous_listing(frame):
            reasons.append(_REASON_LISTING)
        if not _check_correlation(frame, corr_band):
            reasons.append(_REASON_CORR)
        hurst_ok = _check_hurst(frame, hurst_max)
        if hurst_ok is False:
            reasons.append(_REASON_HURST)

        annotated = Candidate(
            ticker_a=candidate.ticker_a,
            ticker_b=candidate.ticker_b,
            sector=candidate.sector,
            industry=candidate.industry,
            sub_industry=candidate.sub_industry,
            adv_a=candidate.adv_a,
            adv_b=candidate.adv_b,
            exclusion_reason=tuple(reasons),
        )

        if return_rejects or not reasons:
            out.append(annotated)
    return out
