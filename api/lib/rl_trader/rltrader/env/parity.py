"""Parity oracle: vectorized backtester == step-by-step env rollout to 1e-10.

THE LOAD-BEARING LOOK-AHEAD GUARD. The vectorized backtester
(:func:`rltrader.env.backtester.vectorized_backtest`) and the step-by-step env
rollout (:meth:`rltrader.env.trading_env.TradingEnv.rollout`) must produce the
SAME per-bar net-reward / equity curve for ANY action sequence, to 1e-10. A
Hypothesis property test drives random action sequences through both paths; any
mismatch beyond the tolerance indicates the vectorized path peeked at a future bar
(a look-ahead bug) and FAILS the build. This module provides the assertion seam
both the property suite and the train-time export probe call.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from rltrader._exceptions import ValidationError
from rltrader._typing import ActionSequence, FloatArray, ReturnSeries
from rltrader.env.backtester import vectorized_backtest
from rltrader.env.trading_env import EnvConfig, TradingEnv

#: The parity tolerance: the two paths must agree to this absolute max-diff.
PARITY_TOL: float = 1e-10


@dataclass(frozen=True, slots=True)
class ParityReport:
    """Immutable report of a vectorized-vs-stepwise parity check.

    Attributes
    ----------
    max_abs_diff:
        The maximum absolute per-bar difference between the vectorized equity
        curve and the step-by-step rollout.
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


def check_parity(
    returns: ReturnSeries,
    actions: ActionSequence,
    *,
    cost_bps: float = 5.0,
    slippage_bps: float = 1.0,
    tol: float = PARITY_TOL,
) -> ParityReport:
    """Compare the vectorized backtester to the step-by-step env rollout.

    Runs the SAME ``(returns, actions, cost_bps, slippage_bps)`` through both the
    vectorized backtester and a step-by-step :class:`TradingEnv` rollout, and
    reports the maximum absolute per-bar difference against ``tol``. The check
    PASSES iff the two agree to ``tol`` (the look-ahead guard); a failure means the
    vectorized path peeked at the future.

    Parameters
    ----------
    returns:
        The single-asset per-bar return path.
    actions:
        The per-bar action / target-position sequence to replay through both paths.
    cost_bps:
        Per-side transaction cost in basis points (applied IDENTICALLY to both).
    slippage_bps:
        Per-trade slippage in basis points (applied IDENTICALLY to both).
    tol:
        The absolute max-diff tolerance (default ``1e-10``).

    Returns
    -------
    ParityReport
        The max abs diff, the tolerance, the pass flag, and the bar count.

    Raises
    ------
    ValidationError
        If the inputs are malformed or length-mismatched.
    """
    if not np.isfinite(tol) or tol < 0.0:
        raise ValidationError(f"tol must be finite and >= 0, got {tol!r}.")

    # The env's friction (cost + slippage) must be charged IDENTICALLY to the
    # backtester, so drive both paths from the same config. A look-back of 1 keeps
    # the env constructible for the tiny action sequences the oracle stresses; the
    # look-back affects observations only, never the scored net-reward sequence.
    config = EnvConfig(lookback=1, cost_bps=cost_bps, slippage_bps=slippage_bps)
    env = TradingEnv(returns, config)
    actions_arr: FloatArray = np.asarray(actions, dtype="float64").ravel()
    stepwise: FloatArray = env.rollout(actions_arr)

    result = vectorized_backtest(
        returns,
        actions,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
        initial_position=0.0,
    )
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
    returns: ReturnSeries,
    actions: ActionSequence,
    *,
    cost_bps: float = 5.0,
    slippage_bps: float = 1.0,
    tol: float = PARITY_TOL,
) -> FloatArray:
    """Assert vectorized-vs-stepwise parity to ``tol`` and return the agreed curve.

    Convenience wrapper over :func:`check_parity` that RAISES
    :class:`rltrader._exceptions.ValidationError` when the two paths disagree
    beyond ``tol`` (so the train-time export probe and the property suite fail
    loudly on any look-ahead). On success returns the agreed per-bar net-reward
    series.

    Parameters
    ----------
    returns:
        The single-asset per-bar return path.
    actions:
        The per-bar action / target-position sequence.
    cost_bps, slippage_bps:
        Frictions applied IDENTICALLY to both paths.
    tol:
        The absolute max-diff tolerance (default ``1e-10``).

    Returns
    -------
    FloatArray
        The agreed per-bar net-reward series (both paths produce this).

    Raises
    ------
    ValidationError
        If the parity check fails (a look-ahead bug) or the inputs are malformed.
    """
    report = check_parity(
        returns, actions, cost_bps=cost_bps, slippage_bps=slippage_bps, tol=tol
    )
    if not report.passed:
        raise ValidationError(
            "vectorized backtester disagrees with the step-by-step env rollout "
            f"(max_abs_diff={report.max_abs_diff:.3e} > tol={report.tol:.3e}); "
            "the vectorized path is peeking at a future bar (look-ahead)."
        )
    # Recompute the agreed series to return it (both paths produce it identically).
    actions_arr: FloatArray = np.asarray(actions, dtype="float64").ravel()
    return TradingEnv(
        returns,
        EnvConfig(lookback=1, cost_bps=cost_bps, slippage_bps=slippage_bps),
    ).rollout(actions_arr)
