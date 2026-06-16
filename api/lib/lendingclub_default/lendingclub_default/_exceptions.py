"""Typed exception hierarchy for the lendingclub-default library.

A single base (:class:`LendingClubDefaultError`) lets callers catch any
library-raised error with one ``except`` clause, while the specific subclasses
let them distinguish data-shape problems from leakage / split / artifact
failures. Importing this module has no side effects.
"""

from __future__ import annotations

# quantcore-candidate: mirrors risk-metrics:src/riskmetrics/_exceptions.py


class LendingClubDefaultError(Exception):
    """Base class for every exception raised by :mod:`lendingclub_default`.

    Catching ``LendingClubDefaultError`` catches all library-specific failures
    while letting unrelated exceptions (e.g. ``KeyboardInterrupt``) propagate.
    """


class ValidationError(LendingClubDefaultError):
    """Raised when an input fails a shape, dtype, alignment, or domain check.

    Examples: a panel missing required application-time columns, a negative
    ``loan_amnt``, an out-of-range ``int_rate``, or a calibration map asked to
    map a probability outside ``[0, 1]``.
    """


class InsufficientDataError(ValidationError):
    """Raised when there are too few observations to fit or evaluate.

    For example, a temporal split that leaves an empty train or test fold, or a
    vintage cohort with too few defaults to estimate a stable target encoding.
    It subclasses :class:`ValidationError` because "not enough data" is a special
    case of a failed input precondition.
    """


class LeakageError(LendingClubDefaultError):
    """Raised when a known post-funding (leakage) column survives preprocessing.

    Post-funding columns (``recoveries``, ``total_pymnt*``, ``out_prncp*``,
    ``last_pymnt_*``, ...) encode the loan outcome and MUST be dropped before any
    model sees them. This error is the hard backstop behind the leakage property
    test: if a leakage column reaches the feature matrix, the run must fail loud.
    """


class TemporalSplitError(LendingClubDefaultError):
    """Raised when a temporal vintage split would leak the future into the past.

    The split is by ``issue_d``: train on vintages at or before a cutoff, test on
    strictly later vintages. This error guards the invariant that no train-fold
    row has an ``issue_d`` after any test-fold row (no look-ahead).
    """


class ArtifactError(LendingClubDefaultError):
    """Raised when a serialized model artifact cannot be loaded or is malformed.

    Covers a missing/corrupt booster JSON, a fitted-pipeline mismatch, or an
    artifact whose recorded schema does not match the request features. The API
    layer maps this to a ``502`` (the artifact, not the request, is at fault).
    """
