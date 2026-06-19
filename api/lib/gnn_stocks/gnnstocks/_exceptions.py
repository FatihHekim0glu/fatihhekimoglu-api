"""Typed exception hierarchy for the gnn-stocks library.

A single base (:class:`GnnStocksError`) lets callers catch any library-raised
error with one ``except`` clause, while the specific subclasses let them
distinguish data-shape problems from missing-artifact / model-load problems.
Importing this module has no side effects.
"""

from __future__ import annotations

# quantcore-candidate: mirrors mvts-forecast:src/mvtsforecast/_exceptions.py


class GnnStocksError(Exception):
    """Base class for every exception raised by :mod:`gnnstocks`.

    Catching ``GnnStocksError`` catches all library-specific failures while
    letting unrelated exceptions (e.g. ``KeyboardInterrupt``) propagate.
    """


class ValidationError(GnnStocksError):
    """Raised when an input fails a shape, dtype, alignment, or domain check.

    Examples: a non-square adjacency matrix, a feature panel with a
    mismatched node axis, a ``k`` larger than the node count for the k-NN
    adjacency, a negative ``cost_bps``, or a forward-return label that would
    leak into the node feature matrix.
    """


class InsufficientDataError(ValidationError):
    """Raised when there are too few observations for the requested operation.

    For example, a train window shorter than the feature lookback (so a single
    fold's correlation adjacency cannot be estimated), an empty point-in-time
    universe after the as-of membership filter, or a walk-forward split with an
    empty train or test fold after purge and embargo. It subclasses
    :class:`ValidationError` because "not enough data" is a special case of a
    failed input precondition.
    """


class ArtifactError(GnnStocksError):
    """Raised when a shipped ONNX artifact cannot be located, loaded, or run.

    Reserved for the serve path: a missing ``artifacts/*.onnx`` file (the
    dense-adjacency GCN / GraphSAGE / ridge graph), a corrupt model, an
    onnxruntime session that fails to initialize, or an adjacency/feature input
    whose shape does not match the exported graph's expected signature. The
    FastAPI router maps this to a 502 (artifact-load failure), distinct from the
    422 raised for request :class:`ValidationError`.
    """
