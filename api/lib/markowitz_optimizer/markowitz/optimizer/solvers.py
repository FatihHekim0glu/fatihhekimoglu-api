"""Thin wrapper around CVXPY's ``Problem.solve`` with typed error mapping.

The wrapper exists for three reasons:

1.  Centralise the mapping from CVXPY's free-form status strings to our
    typed exception hierarchy (so the optimizer body never needs to
    grow a sprawling ``if status == ...`` chain).
2.  Provide a single place to plumb solver options through (default
    ``CLARABEL`` with sensible tolerances, override via kwargs).
3.  Provide a tiny ``SolverResult`` container so callers can introspect
    objective value / iteration count without poking at private
    attributes of the CVXPY problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from markowitz.optimizer.exceptions import (
    InfeasibleError,
    SolverError,
    UnboundedError,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import cvxpy as cp


_OK_STATUSES = frozenset({"optimal", "optimal_inaccurate"})
_INFEASIBLE_STATUSES = frozenset({"infeasible", "infeasible_inaccurate"})
_UNBOUNDED_STATUSES = frozenset({"unbounded", "unbounded_inaccurate"})


@dataclass(frozen=True)
class SolverResult:
    """Outcome of a successful solve."""

    status: str
    objective_value: float
    solver_name: str


def solve_problem(
    problem: cp.Problem,
    *,
    solver: str = "CLARABEL",
    solver_options: dict[str, Any] | None = None,
) -> SolverResult:
    """Solve ``problem`` and translate failures to typed exceptions.

    Parameters
    ----------
    problem:
        The CVXPY problem to solve.
    solver:
        Solver name passed straight through to ``Problem.solve``.  Defaults
        to ``CLARABEL`` which ships with CVXPY 1.5+ and handles QPs and
        SOCPs without a separate install.
    solver_options:
        Additional keyword arguments forwarded to the solver backend.
    """
    options = dict(solver_options or {})
    try:
        problem.solve(solver=solver, **options)
    except Exception as exc:
        raise SolverError(
            f"Solver {solver!r} raised an exception: {exc}",
            solver_status="solver_error",
            solver_message=str(exc),
        ) from exc

    status = str(problem.status) if problem.status is not None else "unknown"
    objective_value = (
        float(problem.value)
        if problem.value is not None and _is_finite(problem.value)
        else float("nan")
    )

    if status in _OK_STATUSES:
        return SolverResult(
            status=status,
            objective_value=objective_value,
            solver_name=solver,
        )
    if status in _INFEASIBLE_STATUSES:
        raise InfeasibleError(
            f"Problem is infeasible (solver status: {status!r}).",
            solver_status=status,
        )
    if status in _UNBOUNDED_STATUSES:
        raise UnboundedError(
            f"Problem is unbounded (solver status: {status!r}).",
            solver_status=status,
        )
    raise SolverError(
        f"Solver {solver!r} returned unrecognised status {status!r}.",
        solver_status=status,
    )


def _is_finite(value: Any) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return False
    return v == v and v not in (float("inf"), float("-inf"))
