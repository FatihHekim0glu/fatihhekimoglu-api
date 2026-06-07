"""Hysteresis state machine for the pairs-trading rule.

The state machine consumes an absolute z-score series and emits a discrete
position in ``{-1, 0, +1}``. The transitions are:

* **Flat -> short** when ``z > z_entry``.
* **Flat -> long** when ``z < -z_entry``.
* **Open -> flat** when ``|z| < z_exit`` (hysteresis -- the bar must reverse
  far enough past the entry band).
* **Open -> flat (z-stop)** when ``|z| > z_stop``; this overrides the exit
  band so a runaway divergence is closed out.
* **Open -> flat (time-stop)** when the bar counter on the current trade
  exceeds ``time_stop_bars``.

The transitions are applied with a single forward-fill loop because the bar
counter for the time stop is path-dependent. Everything else is computed once
with NumPy before the loop runs so the hot path stays tight.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pairs._exceptions import InputError

__all__ = ["RuleOutput", "apply_rules"]


@dataclass(frozen=True, slots=True)
class RuleOutput:
    """Raw per-bar arrays produced by :func:`apply_rules`.

    Attributes
    ----------
    positions : numpy.ndarray of int8
        Discrete position series in ``{-1, 0, +1}``.
    entries : numpy.ndarray of bool
        ``True`` at bars where the position transitions away from flat.
    exits : numpy.ndarray of bool
        ``True`` at bars where the position transitions back to flat.
    stops : numpy.ndarray of bool
        ``True`` at exits driven by ``|z| > z_stop``.
    time_stops : numpy.ndarray of bool
        ``True`` at exits driven by the bar counter exceeding
        ``time_stop_bars``.
    """

    positions: np.ndarray
    entries: np.ndarray
    exits: np.ndarray
    stops: np.ndarray
    time_stops: np.ndarray


def apply_rules(
    zscore: pd.Series,
    *,
    z_entry: float,
    z_exit: float,
    z_stop: float,
    time_stop_bars: int | None,
) -> RuleOutput:
    """Run the hysteresis state machine over ``zscore``.

    Parameters
    ----------
    zscore : pandas.Series
        Driving z-score. NaN bars are treated as "no signal": positions are
        held but new entries are not opened.
    z_entry, z_exit, z_stop : float
        Hysteresis bands. Must satisfy ``z_exit < z_entry < z_stop``.
    time_stop_bars : int or None
        Maximum bars to remain in any single trade. ``None`` disables.

    Returns
    -------
    RuleOutput
        Position arrays and per-bar transition flags.

    Notes
    -----
    The loop is bounded by ``len(zscore)`` and uses contiguous NumPy arrays
    pre-sliced from the input Series, so the per-bar overhead is essentially
    pure Python attribute access plus integer arithmetic.
    """
    if not isinstance(zscore, pd.Series):
        msg = "zscore must be a pandas Series"
        raise InputError(msg)
    entry = float(z_entry)
    exit_ = float(z_exit)
    stop = float(z_stop)
    if not 0.0 <= exit_ < entry < stop:
        msg = (
            "rule thresholds must satisfy 0 <= z_exit < z_entry < z_stop, "
            f"got exit={exit_!r}, entry={entry!r}, stop={stop!r}"
        )
        raise InputError(msg)
    if time_stop_bars is not None and int(time_stop_bars) <= 0:
        msg = f"time_stop_bars must be positive when set, got {time_stop_bars!r}"
        raise InputError(msg)

    z_values = np.asarray(zscore.to_numpy(dtype=float, copy=False))
    n = z_values.shape[0]
    positions = np.zeros(n, dtype=np.int8)
    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    stops = np.zeros(n, dtype=bool)
    time_stops = np.zeros(n, dtype=bool)

    abs_z = np.abs(z_values)
    is_nan = np.isnan(z_values)

    current_pos = 0
    bars_in_position = 0
    ts_limit = -1 if time_stop_bars is None else int(time_stop_bars)

    for i in range(n):
        previous_pos = current_pos
        if is_nan[i]:
            # Hold whatever we had; time-stop counter still advances so a stale
            # NaN bar inside a trade cannot extend the hold indefinitely.
            if current_pos != 0:
                bars_in_position += 1
                if ts_limit > 0 and bars_in_position > ts_limit:
                    current_pos = 0
                    bars_in_position = 0
                    exits[i] = True
                    time_stops[i] = True
            positions[i] = current_pos
            continue

        z_val = z_values[i]
        a_z = abs_z[i]

        if current_pos == 0:
            if a_z > entry:
                # Short the spread when z is large positive; long when large negative.
                current_pos = -1 if z_val > 0 else 1
                bars_in_position = 1
                entries[i] = True
            positions[i] = current_pos
            continue

        # We are in a trade. Check stops first (z-stop wins over time-stop wins over exit).
        if a_z > stop:
            current_pos = 0
            bars_in_position = 0
            exits[i] = True
            stops[i] = True
        elif ts_limit > 0 and bars_in_position + 1 > ts_limit:
            current_pos = 0
            bars_in_position = 0
            exits[i] = True
            time_stops[i] = True
        elif a_z < exit_:
            current_pos = 0
            bars_in_position = 0
            exits[i] = True
        else:
            bars_in_position += 1

        positions[i] = current_pos
        # An entry flag and an exit flag never coincide because the only way to
        # become non-flat after exiting is to wait for the next bar.
        if previous_pos == 0 and current_pos != 0:
            entries[i] = True

    return RuleOutput(
        positions=positions,
        entries=entries,
        exits=exits,
        stops=stops,
        time_stops=time_stops,
    )
