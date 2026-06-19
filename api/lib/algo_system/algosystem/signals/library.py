"""Pure, STRICTLY CAUSAL signal library — target positions from closed bars only.

Each signal is a PURE function mapping the bar history up to and INCLUDING the
CLOSED bar ``t`` to a target position for bar ``t+1``. THE CAUSALITY CONTRACT:

- the signal at bar ``t`` reads ONLY closed bars ``<= t`` (typically the ``close``
  series); it NEVER reads the forming/partial bar ``t+1`` or any future bar;
- the emitted position vector is for the ``t -> t+1`` holding period, so the
  backtester applies ``position.shift(1)`` and the order fills at bar ``t+1``'s
  OPEN (the no-look-ahead next-bar fill);
- perturbing the forming/future bar MUST NOT change the position emitted at ``t``
  (a property test asserts this invariance).

Signals:

- :func:`ma_crossover` — long when the fast SMA is above the slow SMA, else flat /
  short (``fast < slow``);
- :func:`momentum` — long when the trailing ``lookback``-bar return is positive,
  else flat / short;
- :func:`flat` — the always-flat baseline (zero position; the zero-edge floor).

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from algosystem._exceptions import ValidationError
from algosystem._typing import FloatArray
from algosystem._validation import ensure_series


@dataclass(frozen=True, slots=True)
class SignalSpec:
    """Immutable specification of a single signal + its parameters.

    Used to enumerate the honest multiplicity grid (#signals x #param configs) that
    feeds the Deflated-Sharpe ``n_trials`` and the PBO/CSCV configuration matrix.

    Attributes
    ----------
    name:
        The signal name (``"ma_crossover"`` / ``"momentum"`` / ``"flat"``).
    params:
        The signal's keyword parameters (e.g. ``{"fast": 10, "slow": 50}``).
    """

    name: str
    params: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the signal name (the ``params`` mapping is taken as given)."""
        if self.name not in {"ma_crossover", "momentum", "flat"}:
            raise ValidationError(
                f"SignalSpec: unknown signal {self.name!r}; "
                "expected one of {'ma_crossover', 'momentum', 'flat'}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this spec."""
        return asdict(self)


def ma_crossover(close: pd.Series, *, fast: int = 10, slow: int = 50) -> FloatArray:
    r"""Moving-average crossover target positions (strictly causal).

    For each bar ``t``, computes the trailing simple moving averages
    ``SMA_fast(t)`` and ``SMA_slow(t)`` over the CLOSED ``close`` history ``<= t``,
    and emits ``+1`` (long) when ``SMA_fast(t) > SMA_slow(t)`` else ``-1`` (short)
    or ``0`` (flat, per the author's convention). The position at ``t`` is for the
    ``t -> t+1`` holding period; it reads NO future or forming bar. Bars before the
    slow window is full emit ``0`` (no position).

    Parameters
    ----------
    close:
        The single-asset CLOSE-price series (closed bars only).
    fast:
        The fast SMA window (``>= 1``).
    slow:
        The slow SMA window (``> fast``).

    Returns
    -------
    FloatArray
        The per-bar target-position sequence (same length as ``close``), for the
        ``t -> t+1`` holding period.

    Raises
    ------
    ValidationError
        If ``fast < 1``, ``slow <= fast``, or ``close`` is malformed.
    """
    if fast < 1:
        raise ValidationError(f"ma_crossover: fast must be >= 1, got {fast}.")
    if slow <= fast:
        raise ValidationError(f"ma_crossover: slow ({slow}) must be > fast ({fast}).")

    series = ensure_series(close, name="close")

    # Trailing simple moving averages over the CLOSED history <= t. With
    # ``min_periods`` equal to each window, every average at bar ``t`` reads only
    # ``close[t - window + 1 : t + 1]`` (closed bars <= t) and is NaN until the
    # window is full — never a future or forming bar. This is what makes the
    # signal STRICTLY CAUSAL.
    sma_fast = series.rolling(window=fast, min_periods=fast).mean()
    sma_slow = series.rolling(window=slow, min_periods=slow).mean()

    positions = np.zeros(series.size, dtype="float64")
    fast_arr = sma_fast.to_numpy(dtype="float64")
    slow_arr = sma_slow.to_numpy(dtype="float64")

    # Only score bars where BOTH averages are defined (the slow window is full).
    # Long (+1) when the fast SMA is strictly above the slow SMA, else short (-1);
    # bars in the warm-up (before the slow window fills) stay flat (0).
    ready = np.isfinite(fast_arr) & np.isfinite(slow_arr)
    positions[ready & (fast_arr > slow_arr)] = 1.0
    positions[ready & (fast_arr <= slow_arr)] = -1.0
    return positions


def momentum(close: pd.Series, *, lookback: int = 20) -> FloatArray:
    r"""Time-series momentum target positions (strictly causal).

    For each bar ``t``, computes the trailing ``lookback``-bar simple return
    ``close_t / close_{t - lookback} - 1`` over the CLOSED history ``<= t`` and
    emits ``+1`` (long) when it is positive else ``-1`` (short) or ``0`` (flat, per
    the author's convention). The position at ``t`` is for the ``t -> t+1``
    holding period; it reads NO future or forming bar. Bars before the lookback is
    full emit ``0``.

    Parameters
    ----------
    close:
        The single-asset CLOSE-price series (closed bars only).
    lookback:
        The momentum lookback window (``>= 1``).

    Returns
    -------
    FloatArray
        The per-bar target-position sequence (same length as ``close``), for the
        ``t -> t+1`` holding period.

    Raises
    ------
    ValidationError
        If ``lookback < 1`` or ``close`` is malformed.
    """
    if lookback < 1:
        raise ValidationError(f"momentum: lookback must be >= 1, got {lookback}.")

    series = ensure_series(close, name="close")
    close_arr = series.to_numpy(dtype="float64")
    n = close_arr.size

    positions = np.zeros(n, dtype="float64")
    if n <= lookback:
        # Not a single full lookback window: every bar stays flat (warm-up).
        return positions

    # Trailing ``lookback``-bar simple return ``close_t / close_{t-lookback} - 1``
    # for every bar ``t >= lookback`` — reads only CLOSED bars <= t (the current
    # close and the close ``lookback`` bars back), never a future/forming bar.
    trailing_return = close_arr[lookback:] / close_arr[:-lookback] - 1.0
    scored = positions[lookback:]
    scored[trailing_return > 0.0] = 1.0
    scored[trailing_return <= 0.0] = -1.0
    positions[lookback:] = scored
    return positions


def flat(close: pd.Series) -> FloatArray:
    """Always-flat baseline target positions (the zero-edge floor).

    Emits a zero position at every bar — the trivial no-trade baseline that earns
    no return and pays no cost, the floor against which any edge claim is judged.

    Parameters
    ----------
    close:
        The single-asset CLOSE-price series (used only for its length).

    Returns
    -------
    FloatArray
        A zero vector the same length as ``close``.

    Raises
    ------
    ValidationError
        If ``close`` is malformed.
    """
    series = ensure_series(close, name="close")
    return np.zeros(series.size, dtype="float64")


def build_signal(spec: SignalSpec, close: pd.Series) -> FloatArray:
    """Dispatch a :class:`SignalSpec` to its signal function and return the positions.

    The single entry point the backtester / PBO grid uses to evaluate any
    enumerated configuration uniformly.

    Parameters
    ----------
    spec:
        The signal specification (name + params).
    close:
        The single-asset CLOSE-price series.

    Returns
    -------
    FloatArray
        The per-bar target-position sequence.

    Raises
    ------
    ValidationError
        If ``spec.name`` is unknown or ``flat`` is given parameters.
    """
    if spec.name == "ma_crossover":
        return ma_crossover(close, **spec.params)
    if spec.name == "momentum":
        return momentum(close, **spec.params)
    if spec.name == "flat":
        if spec.params:
            raise ValidationError(f"build_signal: 'flat' takes no parameters, got {spec.params!r}.")
        return flat(close)
    # Unreachable: SignalSpec.__post_init__ already restricts ``name`` to the known
    # set; kept as a defensive guard for an externally-mutated spec.
    raise ValidationError(f"build_signal: unknown signal {spec.name!r}.")
