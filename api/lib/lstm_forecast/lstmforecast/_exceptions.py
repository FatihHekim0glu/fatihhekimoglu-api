"""Typed exception hierarchy for the lstm-forecast library.

A single base (:class:`LstmForecastError`) lets callers catch any library-raised
error with one ``except`` clause, while the specific subclasses let them
distinguish data-shape problems from missing-artifact / model-load problems.
Importing this module has no side effects.
"""

from __future__ import annotations

# quantcore-candidate: mirrors risk-metrics:src/riskmetrics/_exceptions.py


class LstmForecastError(Exception):
    """Base class for every exception raised by :mod:`lstmforecast`.

    Catching ``LstmForecastError`` catches all library-specific failures while
    letting unrelated exceptions (e.g. ``KeyboardInterrupt``) propagate.
    """


class ValidationError(LstmForecastError):
    """Raised when an input fails a shape, dtype, alignment, or domain check.

    Examples: a non-monotonic price index, a ``look_back`` smaller than one, a
    negative ``cost_bps``, or a feature matrix whose rows do not align with the
    target return series.
    """


class InsufficientDataError(ValidationError):
    """Raised when there are too few observations for the requested operation.

    For example, fewer rows than ``look_back + 1`` (so not a single supervised
    sequence/label pair can be formed), or a walk-forward split with an empty
    train, validation, or test fold after purge and embargo. It subclasses
    :class:`ValidationError` because "not enough data" is a special case of a
    failed input precondition.
    """


class ArtifactError(LstmForecastError):
    """Raised when the shipped ONNX artifact cannot be located, loaded, or run.

    Reserved for the serve path: a missing ``artifacts/*.onnx`` file, a corrupt
    model, an onnxruntime session that fails to initialize, or an input tensor
    whose shape does not match the exported graph's expected signature. The
    FastAPI router maps this to a 502 (artifact-load failure), distinct from the
    422 raised for request :class:`ValidationError`.
    """
