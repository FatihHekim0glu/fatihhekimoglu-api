"""Markowitz allocators (minimum-variance and maximum-Sharpe).

These are the classical mean-variance baselines in the horse race. Min-variance
uses only the (shared) covariance; max-Sharpe additionally needs an
expected-return vector, which is supplied as an ADR-documented shrunk ``mu`` (see
:mod:`hrp.estimators.mu`).

Both follow the never-invert-Sigma discipline (Cholesky solve, not explicit
inverse) and raise :class:`SingularCovarianceError` when the covariance cannot be
factored — in deliberate contrast to HRP, which survives that case.

cvxpy is LAZILY imported inside :func:`max_sharpe_weights` only; importing this
module has no side effects and does not require cvxpy to be installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hrp._exceptions import SingularCovarianceError, ValidationError
from hrp._typing import MatrixLike

# quantcore-candidate: mirrors markowitz-optimizer:src/markowitz/core/linalg.py
# (cholesky_solve, never-invert-Sigma) + optimizer entry points.


def _as_square_cov(cov: MatrixLike, *, name: str) -> tuple[np.ndarray, list[object]]:
    """Coerce ``cov`` to a square float64 ndarray and recover its asset labels.

    Returns the matrix and the asset labels (the DataFrame columns when present,
    otherwise an integer range). Raises :class:`ValidationError` if ``cov`` is not
    a square 2-D matrix.
    """
    if isinstance(cov, pd.DataFrame):
        labels: list[object] = list(cov.columns)
        arr = cov.to_numpy(dtype="float64")
    else:
        arr = np.asarray(cov, dtype="float64")
        labels = list(range(arr.shape[0])) if arr.ndim == 2 else []

    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValidationError(f"{name}: cov must be square, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValidationError(f"{name}: cov contains non-finite entries.")
    return arr, labels


def _cholesky_solve(cov: np.ndarray, rhs: np.ndarray, *, name: str) -> np.ndarray:
    r"""Solve ``cov @ x = rhs`` via a Cholesky factorization (never invert ``cov``).

    Factors :math:`\Sigma = L L^\top` and solves the two triangular systems, so
    :math:`\Sigma^{-1}` is never formed explicitly (never-invert-Sigma discipline).
    Raises :class:`SingularCovarianceError` when ``cov`` is not positive definite.
    """
    # quantcore-candidate: cholesky_solve (never-invert-Sigma).
    try:
        factor = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise SingularCovarianceError(
            f"{name}: covariance is singular / not positive definite "
            "and cannot be Cholesky-factored."
        ) from exc
    # Solve L y = rhs, then L^T x = y.
    y = np.linalg.solve(factor, rhs)
    x = np.linalg.solve(factor.T, y)
    return np.asarray(x, dtype="float64")


def min_var_weights(
    cov: MatrixLike,
    *,
    long_only: bool = True,
) -> pd.Series:
    r"""Global minimum-variance portfolio weights.

    Solves :math:`\min_w w^\top \Sigma\, w` subject to :math:`\mathbf{1}^\top w = 1`
    (and, when ``long_only``, :math:`w \ge 0`). The unconstrained solution
    :math:`w^\star = \dfrac{\Sigma^{-1}\mathbf{1}}{\mathbf{1}^\top \Sigma^{-1}\mathbf{1}}`
    is computed via a Cholesky solve of :math:`\Sigma x = \mathbf{1}` rather than
    forming :math:`\Sigma^{-1}` explicitly (never-invert-Sigma discipline). When
    ``long_only`` and the closed form has negative entries, the long-only QP is
    solved instead.

    Parameters
    ----------
    cov:
        An ``N x N`` covariance matrix.
    long_only:
        If ``True`` (default), enforce :math:`w \ge 0`.

    Returns
    -------
    pandas.Series
        Minimum-variance weights labelled by asset, summing to one.

    Raises
    ------
    ValidationError
        If ``cov`` is not square.
    SingularCovarianceError
        If ``cov`` cannot be Cholesky-factored (singular / not positive
        definite). HRP, by contrast, survives this case.
    """
    sigma, labels = _as_square_cov(cov, name="min_var_weights")
    n = sigma.shape[0]
    ones = np.ones(n, dtype="float64")

    # Closed form via Cholesky solve of Sigma x = 1 (never form Sigma^{-1}).
    # w* = (Sigma^{-1} 1) / (1^T Sigma^{-1} 1).
    z = _cholesky_solve(sigma, ones, name="min_var_weights")
    denom = float(ones @ z)
    if denom == 0.0 or not np.isfinite(denom):
        raise SingularCovarianceError(
            "min_var_weights: degenerate covariance (1^T Sigma^{-1} 1 is zero "
            "or non-finite)."
        )
    w = z / denom

    if long_only and np.any(w < 0.0):
        # The closed form has short positions; solve the long-only QP instead.
        w = _min_var_long_only_qp(sigma, name="min_var_weights")

    return pd.Series(w, index=pd.Index(labels), dtype="float64")


def _min_var_long_only_qp(sigma: np.ndarray, *, name: str) -> np.ndarray:
    r"""Solve the long-only minimum-variance QP via cvxpy (lazy import).

    Minimizes :math:`w^\top \Sigma\, w` subject to :math:`\mathbf{1}^\top w = 1`
    and :math:`w \ge 0`. cvxpy is imported lazily; raises :class:`ImportError`
    when it is not installed.
    """
    import cvxpy as cp

    n = sigma.shape[0]
    # Symmetrize and PSD-project the quadratic form so cvxpy accepts it even when
    # round-off has made Sigma slightly asymmetric.
    psd = 0.5 * (sigma + sigma.T)
    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(psd)))
    constraints = [cp.sum(w) == 1.0, w >= 0.0]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if w.value is None or problem.status not in ("optimal", "optimal_inaccurate"):
        raise SingularCovarianceError(
            f"{name}: long-only minimum-variance QP failed to solve "
            f"(status={problem.status})."
        )
    weights = np.asarray(w.value, dtype="float64").ravel()
    # Clip tiny negatives from the solver and renormalize to the simplex.
    weights = np.clip(weights, 0.0, None)
    total = weights.sum()
    if total <= 0.0 or not np.isfinite(total):
        raise SingularCovarianceError(f"{name}: QP returned a degenerate weight vector.")
    return weights / total


def max_sharpe_weights(
    cov: MatrixLike,
    mu: pd.Series,
    *,
    risk_free: float = 0.0,
    long_only: bool = True,
) -> pd.Series:
    r"""Maximum-Sharpe (tangency) portfolio weights.

    Solves for the long-only tangency portfolio that maximizes the Sharpe ratio
    :math:`\dfrac{w^\top \mu - r_f}{\sqrt{w^\top \Sigma\, w}}` subject to
    :math:`\mathbf{1}^\top w = 1` and :math:`w \ge 0`. The constrained problem is
    solved as a convex program.

    LAZY IMPORT: ``cvxpy`` is imported *inside* this function (it is an optional
    dependency, installed only via the ``cvxpy`` extra). Importing this module
    must not import cvxpy.

    FAIRNESS NOTE: ``mu`` should be the ADR-documented shrunk estimator
    (:func:`hrp.estimators.mu.james_stein_mu`) so the comparison is not rigged by
    mu-estimation noise; passing the naive :func:`hrp.estimators.mu.sample_mu` is
    the reported ablation.

    Parameters
    ----------
    cov:
        An ``N x N`` covariance matrix (shared across allocators).
    mu:
        Per-period expected returns labelled by asset, aligned to ``cov``.
    risk_free:
        The per-period risk-free rate to subtract in the Sharpe numerator.
    long_only:
        If ``True`` (default), enforce :math:`w \ge 0`.

    Returns
    -------
    pandas.Series
        Maximum-Sharpe weights labelled by asset, summing to one.

    Raises
    ------
    ValidationError
        If ``cov`` is not square or ``mu`` is not aligned to ``cov``.
    SingularCovarianceError
        If the problem is degenerate because ``cov`` is singular.
    ImportError
        If ``cvxpy`` is not installed (install the ``cvxpy`` extra).
    """
    # LAZY import: cvxpy is an optional dependency (the ``cvxpy`` extra). Importing
    # this module must not import cvxpy.
    import cvxpy as cp

    sigma, labels = _as_square_cov(cov, name="max_sharpe_weights")
    n = sigma.shape[0]

    # FAIRNESS NOTE: ``mu`` should be the ADR-documented shrunk estimator
    # (james_stein_mu) so the comparison is not rigged by mu-estimation noise.
    mu_series = mu if isinstance(mu, pd.Series) else pd.Series(np.asarray(mu, dtype="float64"))
    if isinstance(cov, pd.DataFrame):
        # Align mu to the covariance's asset labels so the vectors correspond.
        try:
            mu_series = mu_series.reindex(cov.columns)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "max_sharpe_weights: mu could not be aligned to cov's columns."
            ) from exc
    mu_vec = np.asarray(mu_series.to_numpy(dtype="float64"), dtype="float64")
    if mu_vec.shape[0] != n:
        raise ValidationError(
            f"max_sharpe_weights: mu has length {mu_vec.shape[0]} but cov is "
            f"{n}x{n}."
        )
    if not np.all(np.isfinite(mu_vec)):
        raise ValidationError(
            "max_sharpe_weights: mu is not aligned to cov (contains NaN after alignment)."
        )

    # Tangency portfolio via the standard convex reformulation: maximize the
    # Sharpe ratio (a quasi-concave objective) by minimizing y^T Sigma y subject
    # to the excess-return normalization (mu - r_f)^T y = 1, then renormalize
    # w = y / 1^T y onto the budget simplex. The long-only tangency adds y >= 0.
    excess = mu_vec - float(risk_free)
    if np.all(excess <= 0.0):
        raise ValidationError(
            "max_sharpe_weights: no asset has positive excess return over the "
            "risk-free rate; the tangency portfolio is undefined."
        )

    psd = 0.5 * (sigma + sigma.T)
    y = cp.Variable(n)
    constraints = [excess @ y == 1.0]
    if long_only:
        constraints.append(y >= 0.0)
    problem = cp.Problem(cp.Minimize(cp.quad_form(y, cp.psd_wrap(psd))), constraints)
    problem.solve()

    if y.value is None or problem.status not in ("optimal", "optimal_inaccurate"):
        raise SingularCovarianceError(
            "max_sharpe_weights: tangency QP failed to solve "
            f"(status={problem.status}); covariance may be degenerate."
        )

    y_val = np.asarray(y.value, dtype="float64").ravel()
    budget = float(y_val.sum())
    if budget == 0.0 or not np.isfinite(budget):
        raise SingularCovarianceError(
            "max_sharpe_weights: degenerate tangency solution (1^T y is zero)."
        )
    w = y_val / budget

    if long_only:
        w = np.clip(w, 0.0, None)
        total = w.sum()
        if total <= 0.0 or not np.isfinite(total):
            raise SingularCovarianceError(
                "max_sharpe_weights: QP returned a degenerate weight vector."
            )
        w = w / total

    return pd.Series(w, index=pd.Index(labels), dtype="float64")
