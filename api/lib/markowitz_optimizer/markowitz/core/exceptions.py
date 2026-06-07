"""Domain-specific exception hierarchy for the core subpackage.

All numerical and structural failures raised by ``markowitz.core`` derive
from :class:`CoreError`, allowing callers to catch the whole family with a
single ``except`` clause while still permitting fine-grained handling of
specific failure modes (e.g. singular covariance, infeasible frontier).
"""

from __future__ import annotations


class CoreError(Exception):
    """Base class for all errors raised by ``markowitz.core``."""


class SingularCovarianceError(CoreError):
    """Raised when a covariance matrix is not positive definite.

    The instance attaches diagnostic information that is helpful for
    downstream regularization heuristics or user-facing error messages.

    Parameters
    ----------
    msg:
        Human-readable explanation of the failure.
    condition_number:
        Ratio ``lambda_max / lambda_min`` of the offending matrix. Set to
        ``float('inf')`` when the matrix is exactly singular.
    min_eigenvalue:
        Smallest eigenvalue of the offending matrix (possibly negative).
    """

    def __init__(
        self,
        msg: str = "",
        *,
        condition_number: float = float("inf"),
        min_eigenvalue: float = 0.0,
    ) -> None:
        super().__init__(msg)
        self.condition_number = condition_number
        self.min_eigenvalue = min_eigenvalue


class InfeasibleFrontierError(CoreError):
    """Raised when the requested frontier point does not exist.

    Examples include requesting the tangency portfolio at a risk-free
    rate that coincides with the GMV expected return, or asking for an
    efficient portfolio at a return that lies outside the attainable
    band of a degenerate frontier.
    """


class NumericalError(CoreError):
    """Raised for numerical failures that are not covered by the more
    specific subclasses (e.g. NaN inputs, non-finite intermediate results,
    or breakdowns of an iterative solver).
    """
