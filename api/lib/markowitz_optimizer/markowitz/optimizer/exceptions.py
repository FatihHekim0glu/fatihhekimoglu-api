"""Typed exception hierarchy used by the numerical optimizer.

All errors raised from :mod:`markowitz.optimizer` descend from
:class:`OptimizationError` so that callers can use a single broad ``except``
clause when they merely want to detect that "the solver failed", while still
having the option to discriminate between fundamentally different failure
modes (infeasibility, unboundedness, numerical breakdown, solver crash).

The carriers (``solver_status``, ``solver_message``, ``constraints_violated``)
are attached as plain attributes -- *not* as constructor positional arguments
-- so that the exception message itself stays compact and the full diagnostic
payload remains programmatically accessible.
"""

from __future__ import annotations


class OptimizationError(Exception):
    """Base class for any failure originating in the optimizer layer.

    Parameters
    ----------
    message:
        Human-readable description of what went wrong.
    solver_status:
        The raw status string returned by the underlying solver (e.g.
        ``"infeasible"``, ``"unbounded"``, ``"solver_error"``).  May be
        ``None`` when the error is raised before any solver was invoked
        (e.g. by an upfront feasibility check).
    solver_message:
        Free-form additional message produced by the solver, when available.
    constraints_violated:
        Names of the constraints that the post-solve verification step
        flagged as violated, when the error was raised after a solve.
    """

    def __init__(
        self,
        message: str,
        *,
        solver_status: str | None = None,
        solver_message: str | None = None,
        constraints_violated: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.solver_status = solver_status
        self.solver_message = solver_message
        self.constraints_violated = constraints_violated


class InfeasibleError(OptimizationError):
    """Raised when the feasible region is empty.

    Either the solver returned ``infeasible`` / ``infeasible_inaccurate``,
    or an upfront sanity check (e.g. ``max(mu) <= rf`` for max-Sharpe)
    detected that no admissible portfolio exists.
    """


class UnboundedError(OptimizationError):
    """Raised when the solver reports the objective is unbounded.

    This typically signals a missing leverage / box constraint on a problem
    that permits arbitrarily large weights.
    """


class SolverError(OptimizationError):
    """Raised when the solver fails for a reason other than (in)feasibility.

    Examples include solver-internal numerical breakdowns, license errors,
    or a status string the optimizer does not recognise.
    """


class NumericalError(OptimizationError):
    """Raised when a derived numerical operation cannot proceed.

    Used for cases where the *solver itself* succeeded but a downstream
    arithmetic step is undefined -- e.g. attempting the Cornuejols--Tutuncu
    back-transform when ``sum(y)`` is effectively zero.
    """
