"""Numerically stable linear algebra primitives.

The module is intentionally narrow: every covariance-related solve in the
rest of the package is funnelled through :func:`cholesky_solve` so that
no caller is ever tempted to use ``numpy.linalg.inv``.  Inversion via
``inv`` is forbidden because it is both slower and numerically inferior
to a Cholesky-based solve for symmetric positive definite systems.

All public functions accept array-likes and return ``numpy`` arrays with
``dtype=float64``; inputs are validated for finiteness and shape.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.linalg import cho_factor, cho_solve, eigvalsh

from markowitz.core.exceptions import NumericalError, SingularCovarianceError

FloatArray = npt.NDArray[np.float64]


def cholesky_solve(
    Sigma: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    check_finite: bool = True,
    overwrite_b: bool = False,
) -> FloatArray:
    """Solve ``Sigma x = b`` for a symmetric positive definite ``Sigma``.

    The Cholesky factorization is preferred over forming the explicit
    inverse: it is twice as fast for SPD systems and avoids the
    catastrophic cancellation associated with inverting near-singular
    matrices.

    Parameters
    ----------
    Sigma:
        Symmetric positive definite matrix of shape ``(n, n)``.
    b:
        Right-hand side of shape ``(n,)`` or ``(n, k)``.
    check_finite:
        Forwarded to SciPy; when ``True``, the inputs are screened for
        ``NaN``/``inf`` before factorization.
    overwrite_b:
        Permit SciPy to overwrite ``b`` to save memory.  The default of
        ``False`` is the safe choice for general callers.

    Returns
    -------
    x:
        Solution array with the same shape as ``b``.

    Raises
    ------
    SingularCovarianceError
        If ``Sigma`` is not positive definite.
    NumericalError
        If ``check_finite`` is true and an input contains ``NaN``/``inf``.
    """

    Sigma_arr = np.asarray(Sigma, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)

    if Sigma_arr.ndim != 2 or Sigma_arr.shape[0] != Sigma_arr.shape[1]:
        raise ValueError(f"Sigma must be a square 2-D matrix, got shape {Sigma_arr.shape!r}")
    if b_arr.shape[0] != Sigma_arr.shape[0]:
        raise ValueError(f"Shape mismatch: Sigma is {Sigma_arr.shape}, b is {b_arr.shape}")

    if check_finite:
        if not np.all(np.isfinite(Sigma_arr)):
            raise NumericalError("Sigma contains non-finite values")
        if not np.all(np.isfinite(b_arr)):
            raise NumericalError("b contains non-finite values")

    try:
        cho = cho_factor(Sigma_arr, lower=True, check_finite=False)
    except np.linalg.LinAlgError as exc:
        # SciPy raises LinAlgError when the matrix is not PD.
        eigs = eigvalsh(Sigma_arr) if Sigma_arr.size else np.array([0.0])
        raise SingularCovarianceError(
            "Cholesky factorization failed; matrix is not positive definite",
            condition_number=float("inf"),
            min_eigenvalue=float(np.min(eigs)) if eigs.size else 0.0,
        ) from exc

    solution: FloatArray = cho_solve(cho, b_arr, overwrite_b=overwrite_b, check_finite=False)
    return solution


def psd_check(
    Sigma: npt.ArrayLike,
    *,
    tol: float = 1e-10,
    require_symmetric: bool = True,
    symmetry_tol: float = 1e-10,
) -> None:
    """Validate that ``Sigma`` is symmetric positive definite.

    Parameters
    ----------
    Sigma:
        Candidate covariance matrix.
    tol:
        Lower bound for the smallest eigenvalue.  Matrices whose
        minimum eigenvalue is below ``tol`` are rejected.
    require_symmetric:
        Reject matrices that are not symmetric within ``symmetry_tol``.
    symmetry_tol:
        Absolute tolerance for the ``||S - S.T||_inf`` check.

    Raises
    ------
    SingularCovarianceError
        If the matrix is not symmetric (when required), not positive
        definite, or contains non-finite entries.
    ValueError
        If the matrix is not 2-D and square.
    """

    arr = np.asarray(Sigma, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Sigma must be a square 2-D matrix, got shape {arr.shape!r}")
    if not np.all(np.isfinite(arr)):
        raise SingularCovarianceError(
            "Sigma contains non-finite values",
            condition_number=float("inf"),
            min_eigenvalue=float("nan"),
        )

    if require_symmetric:
        asymmetry = float(np.max(np.abs(arr - arr.T))) if arr.size else 0.0
        if asymmetry > symmetry_tol:
            raise SingularCovarianceError(
                f"Sigma is not symmetric (max |S - S.T| = {asymmetry:.3e})",
                condition_number=float("inf"),
                min_eigenvalue=float("nan"),
            )

    # Symmetric eigenvalues; cheaper and more accurate than ``eig``.
    sym = 0.5 * (arr + arr.T)
    eigs = eigvalsh(sym) if sym.size else np.array([0.0])
    lam_min = float(np.min(eigs)) if eigs.size else 0.0
    lam_max = float(np.max(eigs)) if eigs.size else 0.0
    if lam_min < tol:
        cond = float("inf") if lam_min <= 0 else lam_max / lam_min
        raise SingularCovarianceError(
            (f"Sigma is not positive definite (min eigenvalue = {lam_min:.3e} < tol = {tol:.1e})"),
            condition_number=cond,
            min_eigenvalue=lam_min,
        )


def condition_number(Sigma: npt.ArrayLike) -> float:
    """Return ``lambda_max / lambda_min`` for ``Sigma``.

    Returns ``float('inf')`` if the matrix is not positive definite,
    contains non-finite entries, or has a non-positive smallest
    eigenvalue.  No exception is raised.
    """

    arr = np.asarray(Sigma, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.size == 0:
        return float("inf")
    if not np.all(np.isfinite(arr)):
        return float("inf")
    sym = 0.5 * (arr + arr.T)
    try:
        eigs = eigvalsh(sym)
    except np.linalg.LinAlgError:  # pragma: no cover - eigvalsh rarely fails
        return float("inf")
    lam_min = float(np.min(eigs))
    lam_max = float(np.max(eigs))
    if lam_min <= 0.0:
        return float("inf")
    return lam_max / lam_min


def regularize(
    Sigma: npt.ArrayLike,
    *,
    eps: float | Literal["auto"] = "auto",
    method: Literal["ridge"] = "ridge",
) -> FloatArray:
    """Return a regularized copy of ``Sigma`` that is safely SPD.

    The default ``ridge`` method adds ``eps * I`` to a symmetrized copy
    of ``Sigma``.  When ``eps='auto'``, a scale-aware value is chosen
    based on the trace of ``Sigma`` and machine precision.

    Parameters
    ----------
    Sigma:
        Input matrix.
    eps:
        Either an explicit non-negative scalar or the string ``'auto'``.
    method:
        Currently only ``'ridge'`` is supported.

    Returns
    -------
    A new ``(n, n)`` ``float64`` array.
    """

    arr = np.asarray(Sigma, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Sigma must be a square 2-D matrix, got shape {arr.shape!r}")
    if method != "ridge":
        raise ValueError(f"Unsupported regularization method: {method!r}")

    n = arr.shape[0]
    sym = 0.5 * (arr + arr.T)

    if eps == "auto":
        # Scale by mean diagonal so eps is dimensionally consistent with
        # Sigma; floor at 1e-8 so that the regularized matrix is well
        # above the default ``psd_check`` tolerance (1e-10) even when
        # the input has zero or tiny entries.
        diag_mean = float(np.mean(np.abs(np.diag(sym)))) if n else 0.0
        eps_val = max(diag_mean * 1e-8, 1e-8)
    else:
        eps_val = float(eps)
        if eps_val < 0.0:
            raise ValueError(f"eps must be non-negative, got {eps_val}")

    result: np.ndarray = sym + eps_val * np.eye(n, dtype=sym.dtype)
    return result
