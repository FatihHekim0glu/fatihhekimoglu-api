"""Typed exception hierarchy for the algo-system library.

A single base (:class:`AlgoSystemError`) lets callers catch any library-raised
error with one ``except`` clause, while the specific subclasses let them
distinguish data-shape problems from execution / parity problems. Importing this
module has no side effects.
"""

from __future__ import annotations

# quantcore-candidate: mirrors rl-trader:src/rltrader/_exceptions.py
# (RlTraderError base + a domain-specific subclass for the execution path),
# reframed for the signal -> backtest -> simulated-execution domain.


class AlgoSystemError(Exception):
    """Base class for every exception raised by :mod:`algosystem`.

    Catching ``AlgoSystemError`` catches all library-specific failures while
    letting unrelated exceptions (e.g. ``KeyboardInterrupt``) propagate.
    """


class ValidationError(AlgoSystemError):
    """Raised when an input fails a shape, dtype, alignment, or domain check.

    Examples: a price/return path with the wrong shape, a ``fast`` window not
    strictly smaller than ``slow``, a negative ``cost_bps`` or ``slippage_bps``,
    a position sequence whose length does not match the return path, or a target
    position outside ``[-1, 1]``.
    """


class InsufficientDataError(ValidationError):
    """Raised when there are too few observations for the requested operation.

    For example, a price path shorter than ``slow + 1`` (so not a single causal
    signal/return step can be formed), or a walk-forward split with an empty
    train or test fold after the purge and embargo. It subclasses
    :class:`ValidationError` because "not enough data" is a special case of a
    failed input precondition.
    """


class ParityError(AlgoSystemError):
    """Raised when the backtest<->live parity oracle detects a divergence.

    Reserved for the LOAD-BEARING look-ahead catch: when the vectorized backtest
    equity curve disagrees with the simulated paper-broker equity curve beyond
    the tolerance (``1e-10``), the vectorized path is peeking at a future bar (a
    look-ahead bug) or a fill-timing bug has crept in. The FastAPI router maps
    this to a 502 (an internal pipeline-integrity failure), distinct from the 422
    raised for request :class:`ValidationError`.
    """


class BarFinalityError(AlgoSystemError):
    """Raised when an order would be triggered by a partial / unclosed bar.

    Reserved for the bar-finality guard: the signal at bar ``t`` may act only on
    CLOSED bars ``<= t``; a forming / partial bar can NEVER trigger an order.
    Attempting to emit an order against a bar that is not yet final is a
    causality violation and is rejected here.
    """
