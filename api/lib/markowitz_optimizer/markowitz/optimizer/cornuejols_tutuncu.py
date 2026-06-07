"""Cornuejols--Tutuncu reformulation of the maximum-Sharpe problem.

The vanilla ``maximise (mu^T w - rf) / sqrt(w^T Sigma w)`` is non-convex (a
fractional / quasi-concave objective).  The Cornuejols--Tutuncu trick is to
substitute

.. math::

    y = \\kappa \\, w, \\qquad \\kappa > 0,
    \\qquad (\\mu - rf \\cdot \\mathbf{1})^T y = 1,

which turns the problem into the convex QP

.. math::

    \\min_{y \\ge 0,\\ \\kappa > 0} \\; y^T \\Sigma y
    \\quad \\text{s.t.} \\quad (\\mu - rf)^T y = 1, \\; \\mathbf{1}^T y = \\kappa.

The optimal weights are then recovered by ``w = y / sum(y)``.  The
construction is only well-posed if at least one asset has a positive
excess return; we expose :func:`detect_degeneracy` to catch that case
*before* the solver wastes time on it.

This module deliberately exposes only the *reformulation* and the
*back-transform*; the actual call into the solver lives in
:mod:`markowitz.optimizer.solvers`, and orchestration is the job of
:class:`markowitz.optimizer.mean_variance.MeanVariance`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from markowitz.optimizer.exceptions import InfeasibleError, NumericalError

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    import cvxpy as cp


_BACK_TRANSFORM_EPS = 1e-12


@dataclass(frozen=True)
class ReformulatedProblem:
    """Container for the CVXPY problem and its decision variables.

    The caller is expected to solve ``problem`` and then read the optimal
    values off ``y`` and ``kappa``.
    """

    problem: cp.Problem
    y: cp.Variable
    kappa: cp.Variable


def detect_degeneracy(
    mu: np.ndarray,
    risk_free_rate: float,
    weight_bounds: tuple[float | None, float | None] | None,
    long_only: bool,
) -> None:
    """Raise :class:`InfeasibleError` if no positive-Sharpe portfolio exists.

    The reformulation requires that the equality ``(mu - rf)^T y = 1`` is
    achievable with the sign structure permitted by the bounds.  In the
    long-only / non-negative-weights case this simply means at least one
    asset must have a strictly positive excess return.
    """
    del weight_bounds  # used by callers for richer diagnostics; not needed here
    excess = np.asarray(mu, dtype=float) - float(risk_free_rate)
    if long_only and float(np.max(excess)) <= 0.0:
        raise InfeasibleError(
            "Max-Sharpe is degenerate: no asset has excess return above the "
            "risk-free rate, so the long-only tangency portfolio is "
            "undefined.",
            solver_status="degenerate",
        )
    if not long_only and float(np.max(np.abs(excess))) <= 0.0:
        raise InfeasibleError(
            "Max-Sharpe is degenerate: all assets have zero excess return.",
            solver_status="degenerate",
        )


def reformulate_max_sharpe(
    mu: np.ndarray,
    Sigma: np.ndarray,
    risk_free_rate: float,
    weight_bounds: tuple[float | None, float | None] | None,
    *,
    long_only: bool = True,
    extra_constraint_builders: Sequence[Any] | None = None,
) -> ReformulatedProblem:
    """Build the CVXPY problem in ``(y, kappa)`` space.

    Parameters
    ----------
    mu, Sigma:
        Expected returns and covariance matrix.
    risk_free_rate:
        Reference rate ``rf``; the excess returns ``mu - rf`` define the
        Sharpe numerator.
    weight_bounds:
        Box ``(lower, upper)`` applied *in weight space*.  Each bound is
        translated into a ``y``-space constraint of the form
        ``y_i >= lower * kappa`` / ``y_i <= upper * kappa``.  Pass ``None``
        to omit box constraints entirely.
    long_only:
        If ``True``, also enforces ``y >= 0``.  When ``weight_bounds`` already
        implies ``lower >= 0`` this is redundant but harmless.
    extra_constraint_builders:
        Optional iterable of callables ``(y, kappa) -> list[cvxpy.Constraint]``
        through which callers can inject additional linear constraints in
        ``y``-space.  Non-linear constraints are not supported by the
        reformulation and should be applied to the standard QP path instead.
    """
    # deferred import - cvxpy is optional
    import cvxpy as cp_  # noqa: PLC0415

    mu = np.asarray(mu, dtype=float).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=float)
    n = mu.shape[0]
    if Sigma.shape != (n, n):
        raise ValueError(f"Sigma shape {Sigma.shape} incompatible with mu length {n}")

    excess = mu - float(risk_free_rate)

    y = cp_.Variable(n, name="y")
    kappa = cp_.Variable(nonneg=True, name="kappa")

    constraints: list[cp.Constraint] = [
        excess @ y == 1.0,
        cp_.sum(y) == kappa,
    ]
    if long_only:
        constraints.append(y >= 0)
    if weight_bounds is not None:
        lower, upper = weight_bounds
        if lower is not None:
            constraints.append(y >= float(lower) * kappa)
        if upper is not None:
            constraints.append(y <= float(upper) * kappa)
    if extra_constraint_builders:
        for builder in extra_constraint_builders:
            constraints.extend(builder(y, kappa))

    objective = cp_.Minimize(cp_.quad_form(y, cp_.psd_wrap(Sigma)))
    problem = cp_.Problem(objective, constraints)
    return ReformulatedProblem(problem=problem, y=y, kappa=kappa)


def back_transform(y_value: np.ndarray) -> np.ndarray:
    """Recover the portfolio weights ``w = y / sum(y)``.

    Raises
    ------
    NumericalError
        If ``sum(y)`` is below :data:`_BACK_TRANSFORM_EPS`; in that regime
        the back-transform is numerically meaningless.
    """
    y_arr = np.asarray(y_value, dtype=float).reshape(-1)
    total = float(np.sum(y_arr))
    if not np.isfinite(total) or abs(total) <= _BACK_TRANSFORM_EPS:
        raise NumericalError(
            f"Cornuejols--Tutuncu back-transform failed: sum(y) is numerically zero ({total!r}).",
            solver_status="degenerate",
        )
    return y_arr / total
