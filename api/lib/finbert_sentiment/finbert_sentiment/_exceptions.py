"""Typed exception hierarchy for the finbert-sentiment library.

A single base (:class:`FinbertSentimentError`) lets callers catch any
library-raised error with one ``except`` clause, while the specific subclasses
let them distinguish data-shape problems from missing-artifact / model-load
problems. Importing this module has no side effects.
"""

from __future__ import annotations

# quantcore-candidate: mirrors risk-metrics:src/riskmetrics/_exceptions.py


class FinbertSentimentError(Exception):
    """Base class for every exception raised by :mod:`finbert_sentiment`.

    Catching ``FinbertSentimentError`` catches all library-specific failures
    while letting unrelated exceptions (e.g. ``KeyboardInterrupt``) propagate.
    """


class ValidationError(FinbertSentimentError):
    """Raised when an input fails a shape, dtype, length, or domain check.

    Examples: an empty text batch, a batch larger than the per-request cap, a
    label outside the 3-way class space, or a score matrix whose row count does
    not match the number of input texts.
    """


class InsufficientDataError(ValidationError):
    """Raised when there are too few examples for the requested operation.

    For example, a stratified group split that cannot place at least one example
    of every class into each of train/val/test, or a bootstrap over an empty
    evaluation set. It subclasses :class:`ValidationError` because "not enough
    data" is a special case of a failed input precondition.
    """


class ArtifactError(FinbertSentimentError):
    """Raised when a served model artifact cannot be located, loaded, or run.

    Reserved for the serve path: a missing ONNX graph or ``tokenizer.json``, a
    corrupt model, an onnxruntime session that fails to initialize, or an input
    whose shape does not match the exported graph's signature. The FastAPI
    router maps this to a 502 (artifact-load failure), distinct from the 422
    raised for request :class:`ValidationError`.
    """
