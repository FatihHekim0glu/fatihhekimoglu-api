"""Black-Litterman posterior construction from absolute and relative views.

Public re-exports
-----------------
* :class:`BlackLitterman` -- the main posterior-construction class.
* :class:`Views`, :class:`AbsoluteView`, :class:`RelativeView` --
  view-specification dataclasses and their container.
* :func:`implied_returns`, :func:`infer_delta` -- equilibrium helpers.
* :func:`idzorek_omega`, :func:`idzorek_omega_approx` -- confidence-based
  ``Omega`` construction.
* Exception hierarchy rooted at :class:`BlackLittermanError`.
"""

from __future__ import annotations

from .black_litterman import BlackLitterman
from .equilibrium import implied_returns, infer_delta
from .exceptions import (
    BlackLittermanError,
    OmegaSpecificationError,
    TauScaleError,
    ViewValidationError,
)
from .idzorek import idzorek_omega, idzorek_omega_approx
from .view_specs import AbsoluteView, RelativeView, Views, ViewSpec

__all__ = [
    "AbsoluteView",
    "BlackLitterman",
    "BlackLittermanError",
    "OmegaSpecificationError",
    "RelativeView",
    "TauScaleError",
    "ViewSpec",
    "ViewValidationError",
    "Views",
    "idzorek_omega",
    "idzorek_omega_approx",
    "implied_returns",
    "infer_delta",
]
