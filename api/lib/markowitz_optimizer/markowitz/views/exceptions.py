"""Exception hierarchy for the Black-Litterman view-construction subsystem."""

from __future__ import annotations


class BlackLittermanError(Exception):
    """Base class for all errors raised by :mod:`markowitz.views`."""


class ViewValidationError(BlackLittermanError):
    """Raised when a user-supplied view fails structural validation.

    Examples
    --------
    * A relative view whose pick-row coefficients do not sum to zero.
    * A view that references an asset which is not present in the universe.
    * A confidence value outside the closed interval ``[0, 1]``.
    """


class TauScaleError(BlackLittermanError):
    """Raised when the prior-scaling parameter ``tau`` is invalid (non-positive)."""


class OmegaSpecificationError(BlackLittermanError):
    """Raised when the view-uncertainty matrix ``Omega`` is mis-specified.

    Triggered for shape mismatches, non-positive-definite supplied matrices, or
    inconsistent combinations of ``omega``, ``omega_method`` and per-view
    confidences.
    """
