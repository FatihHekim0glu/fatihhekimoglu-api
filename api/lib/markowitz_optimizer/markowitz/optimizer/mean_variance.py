"""High-level mean-variance optimizer.

:class:`MeanVariance` is the public entry point for the numerical layer.
It hides the bookkeeping around

* validating / canonicalising inputs (numpy arrays, pandas Series / DataFrames),
* building a *fresh* :class:`cvxpy.Problem` per objective call,
* turning the user's declarative constraint list into CVXPY constraints,
* dispatching the Cornuejols--Tutuncu reformulation for max-Sharpe, and
* mapping solver failures to the typed exception hierarchy.

The class is deliberately stateless across solves -- the only mutable state
is the constraint list and the most recent solution (kept for convenience
in :pyattr:`last_weights`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from markowitz.optimizer.constraints import (
    Constraint,
    ConstraintContext,
    LongOnly,
    WeightBounds,
)
from markowitz.optimizer.cornuejols_tutuncu import (
    back_transform,
    detect_degeneracy,
    reformulate_max_sharpe,
)
from markowitz.optimizer.exceptions import (
    InfeasibleError,
    OptimizationError,
    SolverError,
)
from markowitz.optimizer.solvers import solve_problem

try:  # pragma: no cover - import guard, exercised only when cvxpy missing
    import cvxpy as _cp

    _CVXPY_AVAILABLE = True
    _CVXPY_IMPORT_ERROR: ImportError | None = None
except ImportError as _exc:  # pragma: no cover
    _cp = None
    _CVXPY_AVAILABLE = False
    _CVXPY_IMPORT_ERROR = _exc

if TYPE_CHECKING:  # pragma: no cover
    import cvxpy as cp


_REGULARIZE_DEFAULT = 0.0


class MeanVariance:
    """Convex mean-variance optimizer over a single horizon.

    Parameters
    ----------
    mu:
        Expected returns; either a ``pandas.Series`` indexed by ticker or a
        1-D numpy array.  When an ndarray is passed the assets are named
        ``"A0", "A1", ...``.
    sigma:
        Covariance matrix; either a ``pandas.DataFrame`` whose row / column
        index matches ``mu`` or a 2-D numpy array.
    weight_bounds:
        Default ``(lower, upper)`` box constraint applied to every asset.
        Use ``(0.0, 1.0)`` for long-only, ``(-1.0, 1.0)`` for unconstrained
        long-short.  Pass ``(None, None)`` to disable the box.
    solver:
        CVXPY solver name forwarded to :func:`solve_problem`.
    solver_options:
        Extra keyword arguments forwarded to the solver backend.
    regularize:
        Ridge term added to ``Sigma`` (``Sigma + regularize * I``) before
        building the QP.  Helps numerical stability on ill-conditioned
        covariance matrices; should be left at ``0`` for parity with
        analytical baselines.
    """

    def __init__(
        self,
        mu: pd.Series | np.ndarray,
        sigma: pd.DataFrame | np.ndarray,
        weight_bounds: tuple[float | None, float | None] = (0.0, 1.0),
        *,
        solver: str = "CLARABEL",
        solver_options: dict[str, Any] | None = None,
        regularize: float = _REGULARIZE_DEFAULT,
    ) -> None:
        if not _CVXPY_AVAILABLE:  # pragma: no cover - tested via importorskip
            raise ImportError(
                "MeanVariance requires the optional 'cvxpy' dependency. "
                "Install with: pip install 'markowitz-optimizer[robust]'"
            ) from _CVXPY_IMPORT_ERROR

        tickers, mu_arr = _canonicalise_mu(mu)
        sigma_arr = _canonicalise_sigma(sigma, tickers)

        if regularize < 0:
            raise ValueError("regularize must be non-negative")
        if regularize > 0:
            sigma_arr = sigma_arr + regularize * np.eye(sigma_arr.shape[0])

        self._tickers: tuple[str, ...] = tuple(tickers)
        self._mu: np.ndarray = mu_arr
        self._sigma: np.ndarray = sigma_arr
        self._weight_bounds: tuple[float | None, float | None] = weight_bounds
        self._solver: str = solver
        self._solver_options: dict[str, Any] = dict(solver_options or {})
        self._regularize: float = float(regularize)
        self._constraints: list[Constraint] = []
        self._last_weights: pd.Series | None = None

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def tickers(self) -> list[str]:
        return list(self._tickers)

    @property
    def n_assets(self) -> int:
        return len(self._tickers)

    @property
    def last_weights(self) -> pd.Series | None:
        return self._last_weights

    @property
    def constraints(self) -> list[Constraint]:
        return list(self._constraints)

    # ------------------------------------------------------------------
    # Constraint management
    # ------------------------------------------------------------------

    def add_constraint(self, constraint: Constraint) -> MeanVariance:
        """Append a constraint and return ``self`` for chaining."""
        self._constraints.append(constraint)
        return self

    def clear_constraints(self) -> MeanVariance:
        """Drop every previously-added constraint."""
        self._constraints.clear()
        return self

    # ------------------------------------------------------------------
    # Objectives
    # ------------------------------------------------------------------

    def min_volatility(self) -> pd.Series:
        """Minimise ``w^T Sigma w`` subject to ``sum(w) = 1`` and constraints."""
        w = _cp.Variable(self.n_assets, name="w")
        objective = _cp.Minimize(_cp.quad_form(w, _cp.psd_wrap(self._sigma)))
        constraints = self._base_constraints(w) + self._user_constraints(w)
        problem = _cp.Problem(objective, constraints)
        solve_problem(
            problem,
            solver=self._solver,
            solver_options=self._solver_options,
        )
        return self._finalise(w.value)

    def max_sharpe(self, risk_free_rate: float = 0.0) -> pd.Series:
        """Maximise the Sharpe ratio via the Cornuejols--Tutuncu reformulation."""
        long_only = bool(self._weight_bounds[0] is not None and self._weight_bounds[0] >= 0)
        detect_degeneracy(self._mu, risk_free_rate, self._weight_bounds, long_only=long_only)

        # Translate the user's linear (in w) constraints into y-space:
        # the substitution w = y / kappa means any "Aw <= b * 1" becomes
        # "Ay <= b * kappa".  Box constraints we already wired in above.
        # For now we only allow user constraints on the *standard* QP path
        # (min-vol / efficient-* / max-quadratic-utility).  If users have
        # added extra constraints we still respect the basic LongOnly /
        # WeightBounds ones via reformulation; richer constraints fall
        # through to a runtime error rather than being silently dropped.
        extra_builders = self._collect_y_space_builders()

        reform = reformulate_max_sharpe(
            self._mu,
            self._sigma,
            risk_free_rate=risk_free_rate,
            weight_bounds=self._weight_bounds,
            long_only=long_only,
            extra_constraint_builders=extra_builders,
        )
        solve_problem(
            reform.problem,
            solver=self._solver,
            solver_options=self._solver_options,
        )
        if reform.y.value is None:  # pragma: no cover - defensive
            raise SolverError(
                "Max-Sharpe reformulation produced no y vector.",
                solver_status="no_solution",
            )
        weights = back_transform(np.asarray(reform.y.value))
        return self._finalise(weights)

    def max_quadratic_utility(self, risk_aversion: float = 1.0) -> pd.Series:
        """Maximise ``mu^T w - (lambda/2) w^T Sigma w``.

        With ``risk_aversion = lambda``.  ``lambda > 0`` is required.
        """
        if risk_aversion <= 0:
            raise ValueError("risk_aversion must be strictly positive")
        w = _cp.Variable(self.n_assets, name="w")
        utility = self._mu @ w - 0.5 * risk_aversion * _cp.quad_form(w, _cp.psd_wrap(self._sigma))
        objective = _cp.Maximize(utility)
        constraints = self._base_constraints(w) + self._user_constraints(w)
        problem = _cp.Problem(objective, constraints)
        solve_problem(
            problem,
            solver=self._solver,
            solver_options=self._solver_options,
        )
        return self._finalise(w.value)

    def efficient_return(self, target_return: float) -> pd.Series:
        """Minimise variance subject to ``mu^T w >= target_return``."""
        w = _cp.Variable(self.n_assets, name="w")
        objective = _cp.Minimize(_cp.quad_form(w, _cp.psd_wrap(self._sigma)))
        constraints = (
            self._base_constraints(w)
            + self._user_constraints(w)
            + [self._mu @ w >= float(target_return)]
        )
        problem = _cp.Problem(objective, constraints)
        try:
            solve_problem(
                problem,
                solver=self._solver,
                solver_options=self._solver_options,
            )
        except OptimizationError as exc:
            if isinstance(exc, InfeasibleError):
                raise InfeasibleError(
                    f"No feasible portfolio attains target_return={target_return}.",
                    solver_status=exc.solver_status,
                ) from exc
            raise
        return self._finalise(w.value)

    def efficient_risk(self, target_volatility: float) -> pd.Series:
        """Maximise ``mu^T w`` subject to ``sqrt(w^T Sigma w) <= target_vol``."""
        if target_volatility <= 0:
            raise ValueError("target_volatility must be strictly positive")
        w = _cp.Variable(self.n_assets, name="w")
        objective = _cp.Maximize(self._mu @ w)
        risk_cap = _cp.quad_form(w, _cp.psd_wrap(self._sigma)) <= (float(target_volatility) ** 2)
        constraints = self._base_constraints(w) + self._user_constraints(w) + [risk_cap]
        problem = _cp.Problem(objective, constraints)
        solve_problem(
            problem,
            solver=self._solver,
            solver_options=self._solver_options,
        )
        return self._finalise(w.value)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def portfolio_performance(
        self,
        weights: pd.Series | np.ndarray | None = None,
        *,
        risk_free_rate: float = 0.0,
    ) -> tuple[float, float, float]:
        """Return ``(expected_return, volatility, sharpe_ratio)``.

        If ``weights`` is ``None`` the most-recently-computed weights are
        used; an error is raised when no solve has happened yet.
        """
        if weights is None:
            if self._last_weights is None:
                raise ValueError("No weights available -- call an optimizer first or pass weights.")
            w_arr = self._last_weights.to_numpy(dtype=float)
        else:
            w_arr = _align_weights(weights, self._tickers)

        expected_return = float(self._mu @ w_arr)
        variance = float(w_arr @ self._sigma @ w_arr)
        volatility = float(np.sqrt(max(variance, 0.0)))
        if volatility <= 0.0:
            sharpe = float("nan")
        else:
            sharpe = (expected_return - float(risk_free_rate)) / volatility
        return expected_return, volatility, sharpe

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _base_constraints(self, w: cp.Variable) -> list[cp.Constraint]:
        """Default budget + box constraints applied to every standard QP."""
        constraints: list[cp.Constraint] = [_cp.sum(w) == 1]
        lower, upper = self._weight_bounds
        if lower is not None:
            constraints.append(w >= float(lower))
        if upper is not None:
            constraints.append(w <= float(upper))
        return constraints

    def _user_constraints(self, w: cp.Variable) -> list[cp.Constraint]:
        if not self._constraints:
            return []
        ctx = ConstraintContext(
            tickers=self._tickers,
            n_assets=self.n_assets,
            long_only=bool(self._weight_bounds[0] is not None and self._weight_bounds[0] >= 0),
        )
        out: list[cp.Constraint] = []
        for c in self._constraints:
            out.extend(c.apply_cvxpy(w, ctx))
        return out

    def _collect_y_space_builders(self) -> list[Any]:
        """Translate compatible user constraints into y-space for max-Sharpe.

        Only constraints whose w-space form is linear translate cleanly via
        ``w = y / kappa``; anything else triggers an explicit error so that
        users are not silently given an unconstrained max-Sharpe answer.
        """
        builders: list[Any] = []
        for c in self._constraints:
            if isinstance(c, LongOnly):
                builders.append(lambda y, _kappa: [y >= 0])
            elif isinstance(c, WeightBounds):
                builders.append(_make_box_builder(c, self.n_assets))
            else:
                raise OptimizationError(
                    "max_sharpe with custom constraint "
                    f"{type(c).__name__!r} is not supported via the "
                    "Cornuejols-Tutuncu reformulation. Use efficient_return "
                    "or max_quadratic_utility instead.",
                    solver_status="unsupported_constraint",
                    constraints_violated=[c.description],
                )
        return builders

    def _finalise(self, raw_weights: np.ndarray | None) -> pd.Series:
        if raw_weights is None:
            raise SolverError(
                "Solver reported success but returned no weight vector.",
                solver_status="no_solution",
            )
        weights = np.asarray(raw_weights, dtype=float).reshape(-1)
        # Hard-clip tiny negative numerical noise on long-only problems
        lower = self._weight_bounds[0]
        if lower is not None and lower >= 0:
            weights = np.where(weights < 0, 0.0, weights)
        series = pd.Series(weights, index=list(self._tickers), name="weight")
        self._last_weights = series
        return series


# ---------------------------------------------------------------------------
# Input canonicalisation helpers
# ---------------------------------------------------------------------------


def _canonicalise_mu(mu: pd.Series | np.ndarray) -> tuple[tuple[str, ...], np.ndarray]:
    if isinstance(mu, pd.Series):
        tickers = tuple(str(t) for t in mu.index)
        arr = mu.to_numpy(dtype=float)
    else:
        arr = np.asarray(mu, dtype=float).reshape(-1)
        tickers = tuple(f"A{i}" for i in range(arr.shape[0]))
    if arr.ndim != 1:
        raise ValueError("mu must be 1-dimensional")
    if arr.size == 0:
        raise ValueError("mu must contain at least one asset")
    return tickers, arr


def _canonicalise_sigma(sigma: pd.DataFrame | np.ndarray, tickers: tuple[str, ...]) -> np.ndarray:
    n = len(tickers)
    if isinstance(sigma, pd.DataFrame):
        sigma_df: pd.DataFrame = sigma
        if list(sigma_df.index) != list(tickers) or list(sigma_df.columns) != list(tickers):
            # Try to reindex to the canonical order; raises KeyError if mismatched.
            sigma_df = sigma_df.reindex(index=list(tickers), columns=list(tickers))
            if sigma_df.isna().any().any():
                raise ValueError("sigma index/columns do not align with mu's index")
        arr = sigma_df.to_numpy(dtype=float)
    else:
        arr = np.asarray(sigma, dtype=float)
    if arr.shape != (n, n):
        raise ValueError(f"sigma shape {arr.shape} incompatible with {n} assets")
    # Symmetrise to guard against tiny asymmetry from estimators.
    arr = 0.5 * (arr + arr.T)
    return np.asarray(arr, dtype=np.float64)


def _align_weights(weights: pd.Series | np.ndarray, tickers: tuple[str, ...]) -> np.ndarray:
    if isinstance(weights, pd.Series):
        return weights.reindex(list(tickers)).fillna(0.0).to_numpy(dtype=float)
    arr = np.asarray(weights, dtype=float).reshape(-1)
    if arr.shape != (len(tickers),):
        raise ValueError(f"weights shape {arr.shape} does not match {len(tickers)} tickers")
    return arr


def _make_box_builder(c: WeightBounds, n: int) -> Any:
    """Build a y-space translator for a :class:`WeightBounds` constraint."""
    lower = c._resolve(c.lower, n)
    upper = c._resolve(c.upper, n)

    def _builder(y: Any, kappa: Any) -> list[Any]:
        out: list[Any] = []
        # Lower / upper bounds become y >= lower * kappa / y <= upper * kappa.
        out.append(y >= lower * kappa)
        out.append(y <= upper * kappa)
        return out

    return _builder
