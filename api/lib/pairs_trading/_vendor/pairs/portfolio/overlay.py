"""Portfolio-level overlays.

Two overlays live here:

* :func:`vol_target_overlay` -- multiplies portfolio returns by a clipped,
  one-bar-lagged multiplier so that realised volatility tracks a target.
* :func:`drawdown_killswitch` -- a three-state machine (ARMED, TRIPPED,
  LADDER_HALF) that scales the book down to zero after a drawdown breach and
  ladders back through 50% and 100% gross exposure.

Both overlays are applied to the *pre-overlay* portfolio return stream and
return a multiplier series indexed identically to the input. They are
strictly causal: the multiplier at time ``t`` depends only on data up to and
including ``t - 1``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from pairs._exceptions import InputError
from pairs.portfolio.results import KillswitchEvent

__all__ = ["drawdown_killswitch", "vol_target_overlay"]


def vol_target_overlay(
    portfolio_returns: pd.Series,
    *,
    target_vol: float = 0.08,
    window: int = 20,
    clip: tuple[float, float] = (0.5, 1.5),
    ann_factor: int = 252,
) -> pd.Series:
    """Return a one-bar-lagged volatility-target multiplier series.

    The multiplier at time ``t`` is::

        m_t = clip(target_vol / (sigma_{t-1} * sqrt(ann_factor)), *clip)

    where ``sigma_{t-1}`` is the rolling standard deviation of
    ``portfolio_returns`` over the trailing ``window`` bars (computed with
    ``ddof=1``). The leading bars during which the rolling std cannot be
    estimated receive a multiplier of ``1.0``.

    Parameters
    ----------
    portfolio_returns : pandas.Series
        Pre-overlay return stream.
    target_vol : float, default ``0.08``
        Annualised volatility target.
    window : int, default ``20``
        Rolling window for the realised volatility estimate.
    clip : tuple of float, default ``(0.5, 1.5)``
        Multiplier bounds applied after computing the raw ratio.
    ann_factor : int, default ``252``
        Annualisation factor used to convert per-bar std into annualised vol.

    Returns
    -------
    pandas.Series
        Multiplier indexed identically to ``portfolio_returns``.
    """
    if not isinstance(portfolio_returns, pd.Series):
        msg = "portfolio_returns must be a pandas Series"
        raise InputError(msg)
    if float(target_vol) <= 0.0:
        msg = "target_vol must be positive"
        raise InputError(msg)
    if int(window) <= 1:
        msg = "window must be > 1"
        raise InputError(msg)
    lo, hi = float(clip[0]), float(clip[1])
    if not (0.0 < lo <= hi):
        msg = "clip must satisfy 0 < lo <= hi"
        raise InputError(msg)
    if int(ann_factor) <= 0:
        msg = "ann_factor must be positive"
        raise InputError(msg)

    realised = portfolio_returns.rolling(window=int(window), min_periods=int(window)).std(ddof=1)
    realised_ann = realised * math.sqrt(int(ann_factor))
    ratio = float(target_vol) / realised_ann
    ratio = ratio.replace([np.inf, -np.inf], hi)
    ratio = ratio.clip(lower=lo, upper=hi)
    # One-bar lag: multiplier at t uses realised vol up to t-1.
    multiplier = ratio.shift(1)
    multiplier = multiplier.fillna(1.0)
    return multiplier


def drawdown_killswitch(
    equity: pd.Series,
    *,
    dd_threshold: float = 0.08,
    dd_window: int = 20,
    ladder_days: int = 10,
) -> tuple[pd.Series, list[KillswitchEvent]]:
    """Return a (multiplier, events) pair implementing a drawdown killswitch.

    The state machine has three states, each with a multiplier applied at the
    *next* bar (one-bar lag):

    * ``ARMED`` -- multiplier 1.0; trips when the rolling drawdown over the
      last ``dd_window`` bars exceeds ``dd_threshold``.
    * ``TRIPPED`` -- multiplier 0.0; transitions to ``LADDER_HALF`` after
      ``ladder_days`` bars without a new equity low.
    * ``LADDER_HALF`` -- multiplier 0.5; transitions back to ``ARMED`` after
      a further ``ladder_days`` bars without a new low.

    A fresh equity low while not in ``ARMED`` resets the no-new-low counter.

    Parameters
    ----------
    equity : pandas.Series
        Pre-killswitch equity curve.
    dd_threshold : float, default ``0.08``
        Drawdown fraction that trips the killswitch.
    dd_window : int, default ``20``
        Rolling window for the drawdown computation.
    ladder_days : int, default ``10``
        Bars of no new low required to step up one rung.

    Returns
    -------
    (pandas.Series, list of KillswitchEvent)
        Multiplier series (indexed identically to ``equity``) and the ordered
        audit log of state transitions.
    """
    if not isinstance(equity, pd.Series):
        msg = "equity must be a pandas Series"
        raise InputError(msg)
    if not (0.0 < float(dd_threshold) < 1.0):
        msg = "dd_threshold must lie in (0, 1)"
        raise InputError(msg)
    if int(dd_window) <= 1:
        msg = "dd_window must be > 1"
        raise InputError(msg)
    if int(ladder_days) <= 0:
        msg = "ladder_days must be positive"
        raise InputError(msg)

    n = len(equity)
    multiplier = pd.Series(1.0, index=equity.index, dtype=float)
    events: list[KillswitchEvent] = []
    if n == 0:
        return multiplier, events

    roll_max = equity.rolling(window=int(dd_window), min_periods=1).max()
    drawdown = 1.0 - equity / roll_max

    state = "ARMED"
    current_mult = 1.0
    bars_in_state = 0
    rung_low = float(equity.iloc[0])

    for t in range(n):
        # Apply current state's multiplier at this bar (one-bar lag: the
        # state decided at the end of bar t-1 governs the multiplier at t).
        multiplier.iloc[t] = current_mult
        eq_t = float(equity.iloc[t])
        dd_t = float(drawdown.iloc[t]) if not pd.isna(drawdown.iloc[t]) else 0.0

        if state == "ARMED":
            if dd_t >= float(dd_threshold):
                events.append(
                    KillswitchEvent(
                        asof=equity.index[t],
                        trigger="dd_threshold",
                        drawdown=dd_t,
                        gross_before=current_mult,
                        gross_after=0.0,
                    )
                )
                state = "TRIPPED"
                current_mult = 0.0
                bars_in_state = 0
                rung_low = eq_t
        else:
            if eq_t < rung_low:
                rung_low = eq_t
                bars_in_state = 0
            else:
                bars_in_state += 1
            if bars_in_state >= int(ladder_days):
                if state == "TRIPPED":
                    events.append(
                        KillswitchEvent(
                            asof=equity.index[t],
                            trigger="recover_half",
                            drawdown=dd_t,
                            gross_before=current_mult,
                            gross_after=0.5,
                        )
                    )
                    state = "LADDER_HALF"
                    current_mult = 0.5
                    bars_in_state = 0
                elif state == "LADDER_HALF":
                    events.append(
                        KillswitchEvent(
                            asof=equity.index[t],
                            trigger="recover_full",
                            drawdown=dd_t,
                            gross_before=current_mult,
                            gross_after=1.0,
                        )
                    )
                    state = "ARMED"
                    current_mult = 1.0
                    bars_in_state = 0

    # One-bar lag: shift the multiplier so trips/recoveries take effect next bar.
    shifted = multiplier.shift(1).fillna(1.0)
    return shifted, events
