"""Typed exception hierarchy for the nn-vs-bs library.

A single base (:class:`NnVsBsError`) lets callers catch any library-raised error
with one ``except`` clause, while the specific subclasses let them distinguish
input-shape/domain problems, missing-data problems, leakage guards, and
missing-artifact / model-load problems. Importing this module has no side
effects.
"""

from __future__ import annotations

# quantcore-candidate: mirrors risk-metrics:src/riskmetrics/_exceptions.py


class NnVsBsError(Exception):
    """Base class for every exception raised by :mod:`nnvsbs`.

    Catching ``NnVsBsError`` catches all library-specific failures while letting
    unrelated exceptions (e.g. ``KeyboardInterrupt``) propagate.
    """


class ValidationError(NnVsBsError):
    """Raised when an input fails a shape, dtype, alignment, or domain check.

    Examples: a non-positive spot/strike, a negative time-to-expiry, a volatility
    outside ``(0, inf)``, or a feature matrix whose rows do not align with the
    target price series. The FastAPI router maps this to a 422.
    """


class InsufficientDataError(ValidationError):
    """Raised when there are too few observations for the requested operation.

    For example, a synthetic grid with zero quotes, or a quote-date group split
    that leaves an empty train or test fold after the embargo. It subclasses
    :class:`ValidationError` because "not enough data" is a special case of a
    failed input precondition.
    """


class LeakageError(NnVsBsError):
    """Raised when a feature/split would leak the prediction target.

    Reserved for the correctness guards: an implied-vol-derived column reaching
    the NN feature matrix when the label is price (IV is a Black-Scholes
    inversion of the target price), or a quote-date split whose test fold is not
    strictly in the future of its train fold. This is a programming/contract
    error, not a user-input error, so it does not subclass
    :class:`ValidationError`.
    """


class ConvergenceError(NnVsBsError):
    """Raised when an iterative solver fails to converge.

    Reserved for the implied-volatility root finder: a Newton/Brent iteration
    that does not reach the requested tolerance within the iteration budget, or a
    target price outside the no-arbitrage bounds for which no implied volatility
    exists.
    """


class ArtifactError(NnVsBsError):
    """Raised when the shipped ONNX artifact cannot be located, loaded, or run.

    Reserved for the serve path: a missing ``artifacts/*.onnx`` file, a corrupt
    model, an onnxruntime session that fails to initialize, or an input tensor
    whose shape does not match the exported graph's expected signature. The
    FastAPI router maps this to a 502 (artifact-load failure), distinct from the
    422 raised for request :class:`ValidationError`.
    """
