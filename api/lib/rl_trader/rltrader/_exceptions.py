"""Typed exception hierarchy for the rl-trader library.

A single base (:class:`RlTraderError`) lets callers catch any library-raised
error with one ``except`` clause, while the specific subclasses let them
distinguish data-shape problems from missing-artifact / policy-load problems.
Importing this module has no side effects.
"""

from __future__ import annotations

# quantcore-candidate: mirrors gnn-stocks:src/gnnstocks/_exceptions.py
# (RlTraderError base + ArtifactError for the ONNX serve path).


class RlTraderError(Exception):
    """Base class for every exception raised by :mod:`rltrader`.

    Catching ``RlTraderError`` catches all library-specific failures while
    letting unrelated exceptions (e.g. ``KeyboardInterrupt``) propagate.
    """


class ValidationError(RlTraderError):
    """Raised when an input fails a shape, dtype, alignment, or domain check.

    Examples: a price/return panel with the wrong shape, a ``lookback`` larger
    than the available history, a negative ``cost_bps`` or ``slippage_bps``, an
    action sequence whose length does not match the price path, or a target
    position outside ``[-1, 1]``.
    """


class InsufficientDataError(ValidationError):
    """Raised when there are too few observations for the requested operation.

    For example, a price path shorter than ``lookback + 1`` (so not a single
    causal observation/reward step can be formed), or a walk-forward split with
    an empty train or test fold after the purge and embargo. It subclasses
    :class:`ValidationError` because "not enough data" is a special case of a
    failed input precondition.
    """


class ArtifactError(RlTraderError):
    """Raised when a shipped ONNX policy artifact cannot be located, loaded, or run.

    Reserved for the serve path: a missing ``artifacts/policy.onnx`` file, a
    corrupt graph, an onnxruntime session that fails to initialize, or an
    observation input whose shape does not match the exported policy's expected
    signature. The FastAPI router maps this to a 502 (artifact-load failure),
    distinct from the 422 raised for request :class:`ValidationError`.
    """
