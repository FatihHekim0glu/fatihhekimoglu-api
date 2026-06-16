"""Typed exception hierarchy for the HRP library.

A single base (:class:`HRPError`) lets callers catch any library-raised error
with one ``except`` clause, while the specific subclasses let them distinguish
data-shape problems from numerical-degeneracy problems. Importing this module
has no side effects.
"""

from __future__ import annotations

# quantcore-candidate: mirrors risk-metrics:src/riskmetrics/_exceptions.py


class HRPError(Exception):
    """Base class for every exception raised by :mod:`stockclusters`.

    Catching ``HRPError`` catches all library-specific failures while letting
    unrelated exceptions (e.g. ``KeyboardInterrupt``) propagate.
    """


class ValidationError(HRPError):
    """Raised when an input fails a shape, dtype, alignment, or domain check.

    Examples: a non-square covariance matrix, a returns panel with mismatched
    index/columns, a negative ``cost_bps``, or a ``lookback_window`` smaller
    than ``n_assets + 1``.
    """


class InsufficientDataError(ValidationError):
    """Raised when there are too few observations to estimate the requested quantity.

    For example, fewer in-sample rows than ``n_assets + 1`` (so the sample
    covariance is rank-deficient by construction), or an empty rebalance window.
    It subclasses :class:`ValidationError` because "not enough data" is a special
    case of a failed input precondition.
    """


class SingularCovarianceError(HRPError):
    """Raised when a covariance matrix is singular (or numerically so) where invertibility is required.

    HRP itself never inverts the full covariance and so must *not* raise this —
    surviving a singular covariance is the paper's headline robustness claim.
    This error is reserved for the Markowitz adapter and any code path that
    genuinely requires a Cholesky factor / matrix inverse.
    """
