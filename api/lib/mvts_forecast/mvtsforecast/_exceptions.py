"""Typed exception hierarchy for the mvts-forecast library.

A single base (:class:`MvtsForecastError`) lets callers catch any library-raised
error with one ``except`` clause, while the specific subclasses let them
distinguish data-shape problems from missing-artifact / model-load problems.
Importing this module has no side effects.
"""

from __future__ import annotations

# quantcore-candidate: mirrors lstm-forecast:src/lstmforecast/_exceptions.py


class MvtsForecastError(Exception):
    """Base class for every exception raised by :mod:`mvtsforecast`.

    Catching ``MvtsForecastError`` catches all library-specific failures while
    letting unrelated exceptions (e.g. ``KeyboardInterrupt``) propagate.
    """


class ValidationError(MvtsForecastError):
    """Raised when an input fails a shape, dtype, alignment, or domain check.

    Examples: a returns panel with a non-monotonic index, a ``look_back`` smaller
    than one, a target column absent from the basket, a negative ``cost_bps``, or
    an encoder input that would leak the target's same-day transform.
    """


class InsufficientDataError(ValidationError):
    """Raised when there are too few observations for the requested operation.

    For example, fewer rows than ``look_back + 1`` (so not a single supervised
    window/label pair can be formed), or a walk-forward split with an empty
    train or test fold after purge and embargo. It subclasses
    :class:`ValidationError` because "not enough data" is a special case of a
    failed input precondition.
    """


class ArtifactError(MvtsForecastError):
    """Raised when a shipped ONNX artifact cannot be located, loaded, or run.

    Reserved for the serve path: a missing ``artifacts/*.onnx`` file, a corrupt
    model, an onnxruntime session that fails to initialize, or an input tensor
    whose shape does not match the exported graph's expected signature. The
    FastAPI router maps this to a 502 (artifact-load failure), distinct from the
    422 raised for request :class:`ValidationError`.
    """
