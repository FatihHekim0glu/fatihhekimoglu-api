"""Pure-numpy allocation baselines (computed LIVE, torch-free, train-only covariance).

The honest yardsticks the PPO allocator is judged against — all pure numpy, no torch
/ sb3 / onnxruntime / cvxpy / sklearn, so they run LIVE on the serve path:

- :func:`equal_weight` — the 1/N portfolio (DeMiguel-Garlappi-Uppal 2009): equal
  weight on every asset, zero estimation error. The brutal OOS benchmark.
- :func:`markowitz_weights` — the long-only global minimum-variance (Markowitz 1952)
  portfolio, solved via a Cholesky system of the TRAIN-window covariance (never
  inverting Sigma) plus an active-set projection for the long-only constraint.
- :func:`risk_parity_weights` — the equal-risk-contribution portfolio (each asset
  contributes the same marginal risk), solved by a damped fixed-point iteration on
  the TRAIN-window covariance.

Each maps the TRAIN window to a covariance and returns a single weight VECTOR (a
valid long-only simplex); the companion :func:`baseline_weight_path` tiles that
fixed weight across the OOS window (the no-rebalance baseline path) and
:func:`run_baseline` scores it through the SHARED vectorized backtester, so the
baseline equity curve uses the identical strictly-causal, cost-aware accounting as
the RL agent. HONESTY: the covariance is estimated on the TRAIN window ONLY — never
the OOS window — so there is no look-ahead.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rlallocator._exceptions import SingularCovarianceError, ValidationError
from rlallocator._typing import CovMatrix, FloatArray, ReturnPanel, WeightVector
from rlallocator._validation import is_simplex

# quantcore-candidate: mirrors hrp-portfolio:allocate/{naive,markowitz_adapter}.py
# math, re-implemented numpy-pure (no cvxpy) so the serve path stays lean.

#: The names of the shipped allocation baselines (the verdict's yardsticks).
BASELINE_NAMES: tuple[str, ...] = ("equal_weight", "markowitz", "risk_parity")

#: Max iterations + tolerance for the risk-parity fixed-point solver.
_RP_MAX_ITER: int = 1000
_RP_TOL: float = 1e-10


def _coerce_train_panel(returns: ReturnPanel, *, name: str = "train_returns") -> FloatArray:
    """Coerce a TRAIN-window return panel to a finite ``(n_obs, n_assets)`` float64 array."""
    arr = np.asarray(returns, dtype="float64")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValidationError(f"{name} must be a non-empty 2-D (n_obs, n_assets) panel.")
    if not bool(np.isfinite(arr).all()):
        raise ValidationError(f"{name} contains non-finite values.")
    return arr


def _coerce_cov(cov: CovMatrix, *, name: str = "cov") -> FloatArray:
    """Coerce a covariance matrix to a finite, square ``(N, N)`` float64 array."""
    arr = np.asarray(cov, dtype="float64")
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValidationError(f"{name} must be a square 2-D matrix, got shape {arr.shape}.")
    if not bool(np.isfinite(arr).all()):
        raise ValidationError(f"{name} contains non-finite values.")
    return arr


def sample_covariance(train_returns: ReturnPanel) -> FloatArray:
    r"""Sample covariance of the TRAIN-window asset returns (no look-ahead).

    Computes the column-demeaned, unbiased (``ddof=1``) sample covariance
    :math:`\hat{\Sigma} = \frac{1}{T-1} X_c^\top X_c` on the TRAIN window only and
    symmetrizes it. A small ridge proportional to the mean diagonal is added so the
    matrix is positive-definite (Cholesky-factorable) even when ``T`` is close to
    ``N`` — the deliberately-robust default for the baseline allocators.

    Parameters
    ----------
    train_returns:
        The TRAIN-window return panel ``(n_obs, n_assets)``.

    Returns
    -------
    FloatArray
        An ``N x N`` symmetric positive-definite covariance matrix.

    Raises
    ------
    ValidationError
        If the panel is malformed or has fewer than two observations.
    """
    x = _coerce_train_panel(train_returns)
    n_obs = x.shape[0]
    if n_obs < 2:
        raise ValidationError("sample_covariance needs at least two observations.")
    x_centered = x - x.mean(axis=0, keepdims=True)
    cov = (x_centered.T @ x_centered) / (n_obs - 1)
    cov = 0.5 * (cov + cov.T)
    # Ridge for numerical stability (T ~ N): keeps Sigma positive-definite.
    ridge = 1e-8 * float(np.mean(np.diag(cov)) + 1e-12)
    cov = cov + ridge * np.eye(cov.shape[0], dtype="float64")
    return np.asarray(cov, dtype="float64")


def equal_weight(n_assets: int) -> WeightVector:
    r"""Equal-weight (1/N) portfolio weights.

    Returns :math:`w_i = 1 / N` for each of the ``N`` assets — no covariance, no
    mean, no estimation, zero estimation error. The mu-immune, covariance-immune
    yardstick (DeMiguel et al. 2009) the RL allocator must clear net of costs.

    Parameters
    ----------
    n_assets:
        The number of assets ``N`` (``>= 1``).

    Returns
    -------
    WeightVector
        The ``(N,)`` equal-weight simplex.

    Raises
    ------
    ValidationError
        If ``n_assets < 1``.
    """
    if n_assets < 1:
        raise ValidationError(f"equal_weight: n_assets must be >= 1, got {n_assets}.")
    return np.full(n_assets, 1.0 / n_assets, dtype="float64")


def _cholesky_solve(cov: FloatArray, rhs: FloatArray, *, name: str) -> FloatArray:
    r"""Solve ``cov @ x = rhs`` via a Cholesky factorization (never invert ``cov``).

    Factors :math:`\Sigma = L L^\top` and solves the two triangular systems, so
    :math:`\Sigma^{-1}` is never formed explicitly (never-invert-Sigma discipline).
    Raises :class:`SingularCovarianceError` when ``cov`` is not positive definite.
    """
    try:
        factor = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise SingularCovarianceError(
            f"{name}: covariance is singular / not positive definite and cannot be "
            "Cholesky-factored."
        ) from exc
    y = np.linalg.solve(factor, rhs)
    x = np.linalg.solve(factor.T, y)
    return np.asarray(x, dtype="float64")


def markowitz_weights(cov: CovMatrix) -> WeightVector:
    r"""Long-only global minimum-variance (Markowitz) portfolio weights.

    Solves :math:`\min_w w^\top \Sigma\, w` s.t. :math:`\mathbf{1}^\top w = 1` and
    :math:`w \ge 0`. The unconstrained closed form
    :math:`w^\star = \dfrac{\Sigma^{-1}\mathbf{1}}{\mathbf{1}^\top \Sigma^{-1}\mathbf{1}}`
    is computed via a Cholesky solve of :math:`\Sigma x = \mathbf{1}` (never forming
    :math:`\Sigma^{-1}`); when it has negative entries the long-only constraint is
    enforced by an active-set loop that zeroes the most-negative asset and re-solves
    on the surviving sub-universe until all weights are non-negative — a numpy-pure
    substitute for the QP, so the serve path needs no cvxpy.

    HONESTY: ``cov`` must be the TRAIN-window covariance (no look-ahead).

    Parameters
    ----------
    cov:
        The ``N x N`` TRAIN-window covariance matrix.

    Returns
    -------
    WeightVector
        The long-only minimum-variance simplex.

    Raises
    ------
    ValidationError
        If ``cov`` is not square.
    SingularCovarianceError
        If ``cov`` cannot be Cholesky-factored.
    """
    sigma = _coerce_cov(cov, name="markowitz_weights")
    n = sigma.shape[0]
    active = np.ones(n, dtype=bool)

    for _ in range(n):
        idx = np.flatnonzero(active)
        sub = sigma[np.ix_(idx, idx)]
        ones = np.ones(idx.size, dtype="float64")
        z = _cholesky_solve(sub, ones, name="markowitz_weights")
        denom = float(ones @ z)
        if denom == 0.0 or not np.isfinite(denom):
            raise SingularCovarianceError(
                "markowitz_weights: degenerate covariance (1^T Sigma^{-1} 1 is zero / non-finite)."
            )
        sub_w = z / denom
        if bool((sub_w >= -1e-12).all()):
            w = np.zeros(n, dtype="float64")
            w[idx] = np.clip(sub_w, 0.0, None)
            total = float(w.sum())
            if total <= 0.0:  # pragma: no cover - active-set guarantees positive mass.
                return equal_weight(n)
            return (w / total).astype("float64")
        # Drop the most-negative asset and re-solve on the surviving sub-universe.
        drop_local = int(np.argmin(sub_w))
        active[idx[drop_local]] = False

    # Degenerate fall-through (all but one dropped): the survivor takes full weight.
    survivor = active.astype("float64")
    fallback: WeightVector = (survivor / survivor.sum()).astype("float64")
    return fallback


def risk_parity_weights(cov: CovMatrix) -> WeightVector:
    r"""Equal-risk-contribution (risk-parity) portfolio weights.

    Solves for the long-only portfolio where every asset contributes the SAME
    marginal risk, :math:`w_i (\Sigma w)_i = w_j (\Sigma w)_j` for all ``i, j``, via
    the damped fixed-point iteration of Spinu (2013) /
    :math:`w_i \leftarrow w_i \cdot \dfrac{1/(\Sigma w)_i}{\sum_j w_j /(\Sigma w)_j}`
    re-normalized to the simplex each step. Pure numpy — no solver. With a diagonal
    covariance this reduces to the inverse-volatility portfolio.

    HONESTY: ``cov`` must be the TRAIN-window covariance (no look-ahead).

    Parameters
    ----------
    cov:
        The ``N x N`` TRAIN-window covariance matrix.

    Returns
    -------
    WeightVector
        The long-only equal-risk-contribution simplex.

    Raises
    ------
    ValidationError
        If ``cov`` is not square or has a non-positive diagonal entry.
    """
    sigma = _coerce_cov(cov, name="risk_parity_weights")
    n = sigma.shape[0]
    diag = np.diag(sigma)
    if bool((diag <= 0.0).any()):
        raise ValidationError("risk_parity_weights: cov has a non-positive diagonal entry.")

    # Seed from the inverse-volatility portfolio (the diagonal-only fixed point).
    inv_vol = 1.0 / np.sqrt(diag)
    w = inv_vol / float(inv_vol.sum())
    for _ in range(_RP_MAX_ITER):
        marginal = sigma @ w  # (Sigma w)_i
        marginal = np.where(np.abs(marginal) < 1e-300, 1e-300, marginal)
        # ERC fixed point: w_i proportional to 1 / (Sigma w)_i, re-normalized to the
        # simplex. For a diagonal Sigma this converges to inverse-volatility; in
        # general it equalizes the per-asset risk contributions w_i (Sigma w)_i. The
        # raw map oscillates for correlated Sigma, so a 50/50 DAMPED update is taken —
        # the standard fixed-point stabilizer (Spinu 2013) — which converges monotonically.
        update = 1.0 / marginal
        target = update / float(update.sum())
        w_new = 0.5 * w + 0.5 * target
        w_new = w_new / float(w_new.sum())
        if float(np.max(np.abs(w_new - w))) < _RP_TOL:
            w = w_new
            break
        w = w_new
    w = np.clip(w, 0.0, None)
    total = float(w.sum())
    if total <= 0.0:  # pragma: no cover - the iteration keeps the mass positive.
        return equal_weight(n)
    return (w / total).astype("float64")


def baseline_weights(name: str, train_returns: ReturnPanel) -> WeightVector:
    """Build a named baseline's weight VECTOR from the TRAIN-window returns (no look-ahead).

    Routes to the requested baseline (``"equal_weight"`` / ``"markowitz"`` /
    ``"risk_parity"``). The Markowitz and risk-parity baselines estimate their
    covariance on the TRAIN window ONLY via :func:`sample_covariance`; equal-weight
    needs no estimation. The returned vector is a valid long-only simplex.

    Parameters
    ----------
    name:
        One of :data:`BASELINE_NAMES`.
    train_returns:
        The TRAIN-window return panel ``(n_obs, n_assets)`` (defines ``N`` and, for
        the estimated baselines, the covariance).

    Returns
    -------
    WeightVector
        The baseline's ``(N,)`` weight simplex.

    Raises
    ------
    ValidationError
        If ``name`` is unknown or ``train_returns`` is malformed.
    """
    x = _coerce_train_panel(train_returns)
    n_assets = int(x.shape[1])
    if name == "equal_weight":
        return equal_weight(n_assets)
    if name == "markowitz":
        return markowitz_weights(sample_covariance(x))
    if name == "risk_parity":
        return risk_parity_weights(sample_covariance(x))
    raise ValidationError(f"unknown baseline {name!r}; expected one of {sorted(BASELINE_NAMES)}.")


def baseline_weight_path(
    name: str,
    train_returns: ReturnPanel,
    *,
    n_oos_bars: int,
) -> FloatArray:
    """Tile a named baseline's fixed weight vector across the OOS window.

    The baselines are buy-and-hold-the-allocation: estimate the fixed weight vector
    on the TRAIN window and HOLD it across the entire OOS window (the natural
    no-rebalance baseline path), so only the single entry trade incurs turnover. The
    returned ``(n_oos_bars, n_assets)`` weight path is scored through the SHARED
    vectorized backtester.

    Parameters
    ----------
    name:
        One of :data:`BASELINE_NAMES`.
    train_returns:
        The TRAIN-window return panel (covariance / N source).
    n_oos_bars:
        The number of OOS bars to tile the fixed weight across (``>= 1``).

    Returns
    -------
    FloatArray
        The ``(n_oos_bars, n_assets)`` tiled weight path.

    Raises
    ------
    ValidationError
        If ``name`` is unknown, ``train_returns`` malformed, or ``n_oos_bars < 1``.
    """
    if n_oos_bars < 1:
        raise ValidationError(f"baseline_weight_path: n_oos_bars must be >= 1, got {n_oos_bars}.")
    w = baseline_weights(name, train_returns)
    return np.tile(w, (n_oos_bars, 1)).astype("float64")


@dataclass(frozen=True, slots=True)
class BaselineResult:
    """Immutable result of running a baseline weight path through the backtester.

    Attributes
    ----------
    name:
        The baseline label (one of :data:`BASELINE_NAMES`).
    weights:
        The applied per-bar weight path over the scored OOS window.
    net_returns:
        The per-bar net (after-cost) portfolio return series.
    equity_curve:
        The cumulative-wealth curve ``cumprod(1 + net_returns)``.
    turnover:
        Total one-way turnover ``sum_t ||w_t - w_{t-1}||_1`` over the path.
    net_pnl:
        Total compounded net PnL ``equity_curve[-1] - 1``.
    n_bars:
        The number of scored bars.
    """

    name: str
    weights: FloatArray
    net_returns: FloatArray
    equity_curve: FloatArray
    turnover: float
    net_pnl: float
    n_bars: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this baseline result."""
        return {
            "name": str(self.name),
            "weights": [[float(x) for x in row] for row in np.atleast_2d(self.weights)],
            "net_returns": [float(x) for x in np.asarray(self.net_returns).ravel()],
            "equity_curve": [float(x) for x in np.asarray(self.equity_curve).ravel()],
            "turnover": float(self.turnover),
            "net_pnl": float(self.net_pnl),
            "n_bars": int(self.n_bars),
            "meta": dict(self.meta),
        }


def run_baseline(
    name: str,
    train_returns: ReturnPanel,
    oos_returns: ReturnPanel,
    *,
    cost_bps: float = 10.0,
) -> BaselineResult:
    """Estimate a baseline on TRAIN and score it on OOS through the shared backtester.

    Builds the named baseline's TRAIN-window weight vector (no look-ahead), tiles it
    across the OOS window, and evaluates it through the SHARED vectorized backtester
    (:func:`rlallocator.env.backtester.vectorized_backtest`) so the baseline equity
    curve uses the identical strictly-causal, cost-aware accounting as the RL agent —
    the weights at ``t`` earn ``r_{t+1}`` and turnover is charged on ``||Δw||_1``.
    The backtester is imported LAZILY; it is pure numpy (no torch / sb3 /
    onnxruntime), so this runs LIVE.

    Parameters
    ----------
    name:
        One of :data:`BASELINE_NAMES`.
    train_returns:
        The TRAIN-window return panel (covariance / N source; never the OOS window).
    oos_returns:
        The OOS return panel ``(n_oos_bars, n_assets)`` to score the baseline on.
    cost_bps:
        Per-side turnover cost in basis points on ``||Δw||_1``.

    Returns
    -------
    BaselineResult
        The weights, net returns, equity curve, turnover and net PnL.

    Raises
    ------
    ValidationError
        If ``name`` is unknown, the panels are malformed, or asset counts mismatch.
    """
    from rlallocator.env.backtester import vectorized_backtest

    oos = _coerce_train_panel(oos_returns, name="oos_returns")
    train = _coerce_train_panel(train_returns)
    if train.shape[1] != oos.shape[1]:
        raise ValidationError(
            f"train_returns has {train.shape[1]} assets but oos_returns has {oos.shape[1]}."
        )
    weight_path = baseline_weight_path(name, train, n_oos_bars=oos.shape[0])
    result = vectorized_backtest(oos, weight_path, cost_bps=cost_bps, initial_weights=None)
    equity = np.asarray(result.equity_curve, dtype="float64")
    net_pnl = float(equity[-1] - 1.0) if equity.size else 0.0
    return BaselineResult(
        name=name,
        weights=np.asarray(result.weights, dtype="float64"),
        net_returns=np.asarray(result.net_returns, dtype="float64"),
        equity_curve=equity,
        turnover=float(result.turnover),
        net_pnl=net_pnl,
        n_bars=int(result.n_bars),
        meta={"cost_bps": float(cost_bps)},
    )


def is_valid_baseline_weights(weights: WeightVector) -> bool:
    """Return ``True`` iff ``weights`` is a valid long-only simplex (a convenience predicate)."""
    return is_simplex(weights, long_only=True)
