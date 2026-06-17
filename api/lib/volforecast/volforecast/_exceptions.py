"""Typed exception hierarchy for the volforecast library.

A single base (:class:`VolForecastError`) lets callers catch any library-raised
error with one ``except`` clause, while the specific subclasses let them
distinguish data-shape problems from numerical / estimation-degeneracy problems.
Importing this module has no side effects.
"""

from __future__ import annotations

# quantcore-candidate: mirrors risk-metrics:src/riskmetrics/_exceptions.py


class VolForecastError(Exception):
    """Base class for every exception raised by :mod:`volforecast`.

    Catching ``VolForecastError`` catches all library-specific failures while
    letting unrelated exceptions (e.g. ``KeyboardInterrupt``) propagate.
    """


class ValidationError(VolForecastError):
    """Raised when an input fails a shape, dtype, alignment, or domain check.

    Examples: an OHLC frame missing a required column, a non-positive horizon, a
    negative ``gap``, a forecast/actual length mismatch, or a ``cost_bps < 0``.
    """


class InsufficientDataError(ValidationError):
    """Raised when there are too few observations to estimate the requested quantity.

    For example, a return series shorter than the walk-forward warm-up window, or
    a train fold too small to fit a GARCH model. It subclasses
    :class:`ValidationError` because "not enough data" is a special case of a
    failed input precondition.
    """


class ConvergenceError(VolForecastError):
    """Raised when an iterative estimator fails to converge.

    Reserved for the maximum-likelihood GARCH fit (and the hand-rolled parity
    oracle) when the optimizer does not reach a finite, well-defined optimum
    within its iteration budget.
    """
