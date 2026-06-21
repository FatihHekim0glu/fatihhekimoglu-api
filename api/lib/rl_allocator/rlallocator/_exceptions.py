"""Typed exception hierarchy for the rl-allocator library.

A single base (:class:`RlAllocatorError`) lets callers catch any library-raised
error with one ``except`` clause, while the specific subclasses let them
distinguish data-shape problems from missing-artifact / policy-load problems and
from a violated parity / look-ahead invariant. Importing this module has no side
effects.
"""

from __future__ import annotations

# quantcore-candidate: mirrors rl-trader:src/rltrader/_exceptions.py, reframed for
# the multi-asset/portfolio domain (RlAllocatorError base + ArtifactError +
# ParityError for the vectorized<->stepwise look-ahead guard).


class RlAllocatorError(Exception):
    """Base class for every exception raised by :mod:`rlallocator`.

    Catching ``RlAllocatorError`` catches all library-specific failures while
    letting unrelated exceptions (e.g. ``KeyboardInterrupt``) propagate.
    """


class ValidationError(RlAllocatorError):
    """Raised when an input fails a shape, dtype, alignment, or domain check.

    Examples: a multi-asset return panel with the wrong shape, a ``lookback``
    larger than the available history, a negative ``cost_bps``, a weight vector
    that is not a valid simplex (does not sum to one, or has a negative entry in
    the long-only regime), or a walk-forward split with an empty train or test
    fold after the purge and embargo.
    """


class InsufficientDataError(ValidationError):
    """Raised when there are too few observations for the requested operation.

    For example, a return panel shorter than ``lookback + 1`` (so not a single
    causal observation/reward step can be formed), fewer observations than assets
    for a full-rank sample covariance, or a walk-forward split with an empty train
    or test fold after the purge and embargo. It subclasses :class:`ValidationError`
    because "not enough data" is a special case of a failed input precondition.
    """


class ArtifactError(RlAllocatorError):
    """Raised when a shipped ONNX policy artifact cannot be located, loaded, or run.

    Reserved for the serve path: a missing ``artifacts/policy.onnx`` file, a
    corrupt graph, an onnxruntime session that fails to initialize, or an
    observation input whose shape does not match the exported policy's expected
    signature. The FastAPI router maps this to a 502 (artifact-load failure),
    distinct from the 422 raised for request :class:`ValidationError`.
    """


class ParityError(RlAllocatorError):
    """Raised when the vectorized backtester disagrees with the step-by-step env.

    THE LOOK-AHEAD GUARD failure mode: the vectorized multi-asset backtester and
    the step-by-step env rollout must produce the SAME per-bar net-reward / equity
    curve for any weight path, to 1e-10. A disagreement beyond that tolerance means
    the vectorized path peeked at a future bar (a look-ahead bug), and the parity
    oracle raises this so the build fails loudly. The deliberately-leaky negative
    control in the parity suite asserts this is actually raised when it should be.
    """


class SingularCovarianceError(ValidationError):
    """Raised when a covariance matrix cannot be factored for a baseline allocator.

    The Markowitz minimum-variance / mean-variance baseline solves a Cholesky
    system of the (train-window) covariance; a singular / non-positive-definite
    matrix cannot be factored and raises this. It subclasses :class:`ValidationError`
    because a degenerate covariance is a failed numerical precondition; the
    equal-weight and (diagonal) risk-parity baselines survive that case.
    """
