"""Typed exception hierarchy for the quantcore library.

A single base (:class:`QuantCoreError`) lets callers catch any library-raised
error with one ``except`` clause, while :class:`ValidationError` distinguishes
input-precondition failures (shape, dtype, alignment, domain) from unrelated
exceptions. Mirrors the ``ValidationError`` raised by every sibling repo's
``_exceptions`` module so a portfolio-wide migration keeps the same catch
semantics. Importing this module has no side effects.
"""

from __future__ import annotations


class QuantCoreError(Exception):
    """Base class for every exception raised by :mod:`quantcore`.

    Catching ``QuantCoreError`` catches all library-specific failures while
    letting unrelated exceptions (e.g. ``KeyboardInterrupt``) propagate.
    """


class ValidationError(QuantCoreError):
    """Raised when an input fails a shape, dtype, alignment, or domain check.

    Examples: a non-1-D return series, two series of mismatched length, a
    negative ``variance_of_trial_sharpes``, an ``n_trials < 1``, a p-value
    outside ``[0, 1]``, or an even-only ``n_splits`` that is odd.
    """
