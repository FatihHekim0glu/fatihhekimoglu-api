"""Parity oracle: vectorized backtester == step-by-step env rollout to 1e-10.

THE LOAD-BEARING LOOK-AHEAD GUARD. The vectorized multi-asset backtester
(:func:`rlallocator.env.backtester.vectorized_backtest`) and the step-by-step env
rollout (:meth:`rlallocator.env.portfolio_env.PortfolioEnv.rollout`) must produce
the SAME per-bar net-reward / equity curve for ANY weight path, to 1e-10. A
Hypothesis property test drives random weight paths through both paths; any mismatch
beyond the tolerance indicates the vectorized path peeked at a future bar (a
look-ahead bug) and FAILS the build. This module provides the assertion seam both
the property suite and the train-time export probe call.

The :func:`leaky_backtest` function is a DELIBERATELY-LEAKY negative control: it
scores the weights at ``t`` against the CONTEMPORANEOUS ``r_t`` (instead of the
next-bar ``r_{t+1}``), so it peeks at the future. The parity oracle MUST catch the
disagreement between the leaky path and the honest env rollout — the regression
suite asserts :func:`assert_parity` raises :class:`ParityError` for it, proving the
oracle is not vacuous.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from rlallocator._exceptions import ParityError, ValidationError
from rlallocator._typing import FloatArray, ReturnPanel, WeightPath
from rlallocator.env.backtester import vectorized_backtest
from rlallocator.env.portfolio_env import PortfolioEnv, PortfolioEnvConfig

#: The parity tolerance: the two paths must agree to this absolute max-diff.
PARITY_TOL: float = 1e-10


@dataclass(frozen=True, slots=True)
class ParityReport:
    """Immutable report of a vectorized-vs-stepwise parity check.

    Attributes
    ----------
    max_abs_diff:
        The maximum absolute per-bar difference between the vectorized net returns
        and the step-by-step rollout.
    tol:
        The tolerance the check was run against (``1e-10``).
    passed:
        ``True`` iff ``max_abs_diff <= tol`` (no look-ahead detected).
    n_bars:
        The number of bars compared.
    """

    max_abs_diff: float
    tol: float
    passed: bool
    n_bars: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this report."""
        return asdict(self)


def _coerce_weight_path(weights: WeightPath, *, n_assets: int) -> FloatArray:
    """Coerce a weight path to a finite ``(n_bars, n_assets)`` float64 array."""
    w = np.asarray(weights, dtype="float64")
    if w.ndim == 1:
        w = w.reshape(-1, 1)
    if w.ndim != 2 or w.shape[1] != n_assets:
        raise ValidationError(
            f"weights must be a 2-D (n_bars, {n_assets}) path, got shape {w.shape}."
        )
    if not bool(np.isfinite(w).all()):
        raise ValidationError("weights contains non-finite values.")
    return w


def check_parity(
    returns: ReturnPanel,
    weights: WeightPath,
    *,
    cost_bps: float = 10.0,
    tol: float = PARITY_TOL,
) -> ParityReport:
    """Compare the vectorized backtester to the step-by-step env rollout.

    Runs the SAME ``(returns, weights, cost_bps)`` through both the vectorized
    backtester and a step-by-step :class:`PortfolioEnv` rollout — both applying the
    IDENTICAL simplex projection to the raw weight rows — and reports the maximum
    absolute per-bar difference against ``tol``. The check PASSES iff the two agree
    to ``tol`` (the look-ahead guard); a failure means the vectorized path peeked at
    the future.

    Parameters
    ----------
    returns:
        The multi-asset per-bar return panel.
    weights:
        The per-bar raw weight (action) path to replay through both paths.
    cost_bps:
        Per-side turnover cost in basis points (applied IDENTICALLY to both).
    tol:
        The absolute max-diff tolerance (default ``1e-10``).

    Returns
    -------
    ParityReport
        The max abs diff, the tolerance, the pass flag, and the bar count.

    Raises
    ------
    ValidationError
        If the inputs are malformed or shape-mismatched.
    """
    if not np.isfinite(tol) or tol < 0.0:
        raise ValidationError(f"tol must be finite and >= 0, got {tol!r}.")

    panel = np.asarray(returns, dtype="float64")
    if panel.ndim == 1:
        panel = panel.reshape(-1, 1)
    n_assets = int(panel.shape[1])
    raw = _coerce_weight_path(weights, n_assets=n_assets)

    # A look-back of 1 keeps the env constructible for tiny weight paths the oracle
    # stresses; the look-back affects observations only, never the scored net-reward.
    config = PortfolioEnvConfig(lookback=1, cost_bps=cost_bps)
    env = PortfolioEnv(panel, config)
    stepwise = env.rollout(raw)
    # Score the IDENTICAL simplex-projected weight path through the vectorized path.
    projected = env.resolved_weight_path(raw)
    result = vectorized_backtest(panel, projected, cost_bps=cost_bps, initial_weights=None)
    vectorized = np.asarray(result.net_returns, dtype="float64")

    if stepwise.shape != vectorized.shape:  # pragma: no cover - guarded upstream
        raise ValidationError(
            f"parity shape mismatch: stepwise {stepwise.shape} vs vectorized {vectorized.shape}."
        )
    max_abs_diff = float(np.max(np.abs(stepwise - vectorized))) if stepwise.size else 0.0
    return ParityReport(
        max_abs_diff=max_abs_diff,
        tol=float(tol),
        passed=bool(max_abs_diff <= tol),
        n_bars=int(stepwise.size),
    )


def assert_parity(
    returns: ReturnPanel,
    weights: WeightPath,
    *,
    cost_bps: float = 10.0,
    tol: float = PARITY_TOL,
) -> FloatArray:
    """Assert vectorized-vs-stepwise parity to ``tol`` and return the agreed curve.

    Convenience wrapper over :func:`check_parity` that RAISES :class:`ParityError`
    when the two paths disagree beyond ``tol`` (so the train-time export probe and
    the property suite fail loudly on any look-ahead). On success returns the agreed
    per-bar net-reward series.

    Parameters
    ----------
    returns:
        The multi-asset per-bar return panel.
    weights:
        The per-bar raw weight (action) path.
    cost_bps:
        Per-side turnover cost in basis points (applied IDENTICALLY to both paths).
    tol:
        The absolute max-diff tolerance (default ``1e-10``).

    Returns
    -------
    FloatArray
        The agreed per-bar net-reward series (both paths produce this).

    Raises
    ------
    ParityError
        If the parity check fails (a look-ahead bug).
    ValidationError
        If the inputs are malformed.
    """
    report = check_parity(returns, weights, cost_bps=cost_bps, tol=tol)
    if not report.passed:
        raise ParityError(
            "vectorized backtester disagrees with the step-by-step env rollout "
            f"(max_abs_diff={report.max_abs_diff:.3e} > tol={report.tol:.3e}); "
            "the vectorized path is peeking at a future bar (look-ahead)."
        )
    panel = np.asarray(returns, dtype="float64")
    if panel.ndim == 1:
        panel = panel.reshape(-1, 1)
    env = PortfolioEnv(panel, PortfolioEnvConfig(lookback=1, cost_bps=cost_bps))
    return env.rollout(_coerce_weight_path(weights, n_assets=int(panel.shape[1])))


def leaky_backtest(
    returns: ReturnPanel,
    weights: WeightPath,
    *,
    cost_bps: float = 10.0,
) -> FloatArray:
    """DELIBERATELY-LEAKY negative control: score weights at ``t`` against ``r_t``.

    A wrong backtester that scores the weights set at bar ``t`` against the
    CONTEMPORANEOUS return ``r_t`` instead of the next-bar return ``r_{t+1}`` — it
    peeks at the future. The parity oracle MUST catch the disagreement between this
    leaky path and the honest env rollout; the regression suite asserts
    :func:`assert_parity_against` raises :class:`ParityError` when fed this control,
    proving the oracle is not vacuous.

    Parameters
    ----------
    returns:
        The multi-asset per-bar return panel.
    weights:
        The per-bar raw weight (action) path.
    cost_bps:
        Per-side turnover cost in basis points.

    Returns
    -------
    FloatArray
        The (leaky) per-bar net-reward series scored against ``r_t`` (look-ahead).

    Raises
    ------
    ValidationError
        If the inputs are malformed.
    """
    panel = np.asarray(returns, dtype="float64")
    if panel.ndim == 1:
        panel = panel.reshape(-1, 1)
    n_assets = int(panel.shape[1])
    raw = _coerce_weight_path(weights, n_assets=n_assets)
    env = PortfolioEnv(panel, PortfolioEnvConfig(lookback=1, cost_bps=cost_bps))
    projected = env.resolved_weight_path(raw)

    n_scored = panel.shape[0] - 1
    pos = projected[:n_scored]
    # LEAK: score against the CONTEMPORANEOUS r_t (panel[:n_scored]) instead of the
    # honest next-bar r_{t+1} (panel[1:]). This is the look-ahead the oracle catches.
    contemporaneous = panel[:n_scored]
    gross = np.einsum("ta,ta->t", pos, contemporaneous)
    prev = np.empty_like(pos)
    prev[0] = np.zeros(n_assets, dtype="float64")
    if n_scored > 1:
        prev[1:] = pos[:-1]
    turnover = np.abs(pos - prev).sum(axis=1)
    cost = cost_bps / 10_000.0 * turnover
    leaky: FloatArray = (gross - cost).astype("float64")
    return leaky


def assert_parity_against(
    candidate_net_returns: FloatArray,
    returns: ReturnPanel,
    weights: WeightPath,
    *,
    cost_bps: float = 10.0,
    tol: float = PARITY_TOL,
) -> None:
    """Assert a CANDIDATE net-return series matches the honest env rollout to ``tol``.

    The general negative-control seam: compares an arbitrary candidate per-bar
    net-return series (e.g. the output of :func:`leaky_backtest`) against the honest
    step-by-step env rollout for the same ``(returns, weights, cost_bps)``, raising
    :class:`ParityError` on any disagreement beyond ``tol``. The regression suite
    feeds it the leaky control to prove the oracle actually catches look-ahead.

    Parameters
    ----------
    candidate_net_returns:
        The candidate per-bar net-return series to validate against the env.
    returns:
        The multi-asset per-bar return panel.
    weights:
        The per-bar raw weight (action) path.
    cost_bps:
        Per-side turnover cost in basis points.
    tol:
        The absolute max-diff tolerance (default ``1e-10``).

    Raises
    ------
    ParityError
        If the candidate disagrees with the honest env rollout beyond ``tol``.
    ValidationError
        If the inputs are malformed or length-mismatched.
    """
    panel = np.asarray(returns, dtype="float64")
    if panel.ndim == 1:
        panel = panel.reshape(-1, 1)
    env = PortfolioEnv(panel, PortfolioEnvConfig(lookback=1, cost_bps=cost_bps))
    honest = env.rollout(_coerce_weight_path(weights, n_assets=int(panel.shape[1])))
    candidate = np.asarray(candidate_net_returns, dtype="float64").ravel()
    if candidate.shape != honest.shape:
        raise ValidationError(
            f"candidate_net_returns shape {candidate.shape} must match the honest "
            f"rollout {honest.shape}."
        )
    max_abs_diff = float(np.max(np.abs(candidate - honest))) if honest.size else 0.0
    if max_abs_diff > tol:
        raise ParityError(
            "candidate net returns disagree with the honest step-by-step env rollout "
            f"(max_abs_diff={max_abs_diff:.3e} > tol={tol:.3e}); the candidate path is "
            "peeking at a future bar (look-ahead)."
        )
