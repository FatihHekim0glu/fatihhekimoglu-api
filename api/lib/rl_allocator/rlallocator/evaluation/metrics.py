"""OOS performance metrics for a multi-asset portfolio net-return series.

The scalar summaries the verdict + API consume, all judged net of simulated
turnover costs:

- :func:`oos_sharpe` — annualized OOS Sharpe of a per-bar portfolio net-return series;
- :func:`max_drawdown` — the worst peak-to-trough drawdown of the equity curve;
- :func:`turnover` — total one-way turnover of a WEIGHT PATH (sum of L1 weight changes);
- :func:`net_pnl` — total compounded net PnL of the equity curve;
- :func:`hac_standard_error` / :func:`andrews_lag` — the Newey-West HAC long-run
  variance of a per-bar series (e.g. the RL-vs-best-baseline net-return differential),
  reused to build the Diebold-Mariano denominator.

Every builder here is pure numpy (no torch / sklearn / scipy at import or call), so
the serve path computes the baseline + RL metrics live and the Diebold-Mariano
layer builds its HAC denominator torch-free. Importing this module has no side
effects.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from rlallocator._constants import PERIODS_PER_YEAR
from rlallocator._exceptions import ValidationError
from rlallocator._typing import FloatArray, WeightPath

# quantcore-candidate: mirrors rl-trader:src/rltrader/evaluation/metrics.py +
# hrp-portfolio:backtest/stats.py, GENERALIZED to the portfolio L1 turnover.


@dataclass(frozen=True, slots=True)
class StrategyMetrics:
    """Immutable bundle of OOS net-of-cost multi-asset portfolio metrics.

    Attributes
    ----------
    oos_sharpe:
        Annualized OOS Sharpe of the per-bar portfolio net-return series.
    max_drawdown:
        The worst peak-to-trough drawdown (``<= 0``) of the equity curve.
    turnover:
        Total one-way turnover of the weight path (sum of per-bar L1 weight changes).
    net_pnl:
        Total compounded net PnL (``equity[-1] - 1``).
    n_bars:
        The number of scored bars.
    """

    oos_sharpe: float
    max_drawdown: float
    turnover: float
    net_pnl: float
    n_bars: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of these metrics."""
        return asdict(self)


def _coerce_series(series: FloatArray, *, name: str = "series") -> FloatArray:
    """Coerce a per-bar series to a non-empty finite 1-D float64 vector.

    The single boundary every scalar metric (Sharpe, drawdown, net PnL) funnels its
    input through, so they all share one definition of "valid series": flattened to
    1-D, non-empty, and finite. NaN/inf are rejected here rather than silently
    propagated into a Sharpe or an equity curve.

    Parameters
    ----------
    series:
        A per-bar portfolio net-return series.
    name:
        Human-readable label for error messages.

    Returns
    -------
    FloatArray
        The coerced 1-D float64 array.

    Raises
    ------
    ValidationError
        If ``series`` is empty or contains any non-finite value.
    """
    arr = np.asarray(series, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValidationError(f"{name} must be non-empty.")
    if not np.isfinite(arr).all():
        raise ValidationError(f"{name} contains non-finite values.")
    return arr


def _coerce_pair(
    series_a: FloatArray,
    series_b: FloatArray,
    *,
    a_name: str = "series_a",
    b_name: str = "series_b",
) -> tuple[FloatArray, FloatArray]:
    """Coerce a pair of per-bar series to aligned finite float64 vectors.

    Both inputs are flattened to 1-D, checked for non-emptiness, equal length, and
    finiteness. The single boundary the Diebold-Mariano differential funnels its
    two performance series through.

    Parameters
    ----------
    series_a, series_b:
        The two per-bar series (e.g. RL net returns and best-baseline net returns).
    a_name, b_name:
        Human-readable labels for error messages.

    Returns
    -------
    tuple[FloatArray, FloatArray]
        The two coerced 1-D float64 arrays.

    Raises
    ------
    ValidationError
        If either array is empty, lengths differ, or any value is non-finite.
    """
    a = np.asarray(series_a, dtype=np.float64).ravel()
    b = np.asarray(series_b, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0:
        raise ValidationError(f"{a_name} and {b_name} must be non-empty.")
    if a.size != b.size:
        raise ValidationError(
            f"{a_name} (len {a.size}) and {b_name} (len {b.size}) must have the same length."
        )
    if not np.isfinite(a).all():
        raise ValidationError(f"{a_name} contains non-finite values.")
    if not np.isfinite(b).all():
        raise ValidationError(f"{b_name} contains non-finite values.")
    return a, b


def oos_sharpe(
    net_returns: FloatArray,
    *,
    risk_free: float = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    r"""Annualized OOS Sharpe ratio of a per-bar portfolio net-return series.

    Computes :math:`\text{SR} = \dfrac{\bar{r} - r_f}{\sigma_r}\sqrt{\text{ppy}}`
    with the sample standard deviation (``ddof=1``). A (numerically) flat series
    has undefined Sharpe and returns NaN. The series MUST already be net of
    turnover costs — there is no gross-Sharpe escape hatch.

    Parameters
    ----------
    net_returns:
        A per-bar NET (after-cost) portfolio return series.
    risk_free:
        Per-bar risk-free rate subtracted from the mean.
    periods_per_year:
        Annualization factor (``252`` for daily bars).

    Returns
    -------
    float
        The annualized OOS Sharpe (NaN if the return volatility is zero).

    Raises
    ------
    ValidationError
        If ``net_returns`` is empty or non-finite.
    """
    arr = _coerce_series(net_returns, name="net_returns")
    # Sample standard deviation (ddof=1): a Sharpe is an estimate, not a population
    # quantity. A single observation has no dispersion estimate -> undefined.
    if arr.size < 2:
        return math.nan
    std = float(np.std(arr, ddof=1))
    # A (numerically) flat series has zero dispersion -> the Sharpe is undefined.
    # NaN propagates honestly rather than dividing by an EPS floor (which would
    # manufacture a finite Sharpe out of a constant series).
    if std <= 0.0:
        return math.nan
    excess_mean = float(np.mean(arr)) - float(risk_free)
    return excess_mean / std * math.sqrt(periods_per_year)


def max_drawdown(net_returns: FloatArray) -> float:
    r"""Maximum drawdown of a per-bar portfolio net-return series (``<= 0``).

    Builds the cumulative wealth curve :math:`W_t = \prod_{s \le t}(1 + r_s)`,
    tracks its running peak, and returns the most negative value of
    :math:`W_t / \max_{s \le t} W_s - 1` (``0.0`` if the series never declines).

    Parameters
    ----------
    net_returns:
        A per-bar portfolio net-return series.

    Returns
    -------
    float
        The maximum drawdown (``<= 0``).

    Raises
    ------
    ValidationError
        If ``net_returns`` is empty or non-finite.
    """
    arr = _coerce_series(net_returns, name="net_returns")
    wealth = np.cumprod(1.0 + arr)
    running_peak = np.maximum.accumulate(wealth)
    # Drawdown at t = W_t / peak_t - 1 (<= 0). The running peak is strictly
    # positive while wealth stays positive; guard the (pathological) non-positive
    # peak so a divide does not produce inf/NaN.
    drawdown = np.where(running_peak > 0.0, wealth / running_peak - 1.0, 0.0)
    return float(min(0.0, float(drawdown.min())))


def turnover(weights: WeightPath, *, initial_weights: FloatArray | None = None) -> float:
    r"""Total one-way turnover of a per-bar WEIGHT PATH (sum of L1 weight changes).

    For a weight path ``w_t`` shaped ``(n_bars, n_assets)`` returns
    :math:`\sum_t \lVert w_t - w_{t-1} \rVert_1` with the first change taken against
    ``initial_weights`` (the book opens from all-cash / flat by default). The cost
    model charges per-side basis points on this turnover, so net Sharpe must be
    non-increasing in turnover (the cost-monotonicity property).

    Parameters
    ----------
    weights:
        The per-bar weight path (``(n_bars, n_assets)``).
    initial_weights:
        The weight vector held before the first bar; ``None`` => all-zero (flat).

    Returns
    -------
    float
        Total one-way turnover (``>= 0``).

    Raises
    ------
    ValidationError
        If ``weights`` is empty, non-finite, or ``initial_weights`` is misshaped.
    """
    w = np.asarray(weights, dtype="float64")
    if w.ndim == 1:
        w = w.reshape(-1, 1)
    if w.ndim != 2 or w.shape[0] == 0:
        raise ValidationError("turnover: weights must be a non-empty 2-D weight path.")
    if not bool(np.isfinite(w).all()):
        raise ValidationError("turnover: weights contains non-finite values.")

    n_assets = w.shape[1]
    if initial_weights is None:
        init = np.zeros(n_assets, dtype="float64")
    else:
        init = np.asarray(initial_weights, dtype="float64").ravel()
        if init.size != n_assets:
            raise ValidationError(
                f"turnover: initial_weights (len {init.size}) must match n_assets ({n_assets})."
            )
    # prev row t = w_{t-1}, with w_{-1} = initial_weights. Matches the vectorized
    # backtester (which opens the book from ``initial_weights``), so the cost charge
    # derived from this turnover is identical to the env's.
    prev = np.empty_like(w)
    prev[0] = init
    if w.shape[0] > 1:
        prev[1:] = w[:-1]
    return float(np.abs(w - prev).sum())


def net_pnl(net_returns: FloatArray) -> float:
    r"""Total compounded net PnL of a per-bar portfolio net-return series.

    Returns :math:`\prod_t (1 + r_t) - 1`, the total compounded return over the
    OOS window net of turnover costs.

    Parameters
    ----------
    net_returns:
        A per-bar portfolio net-return series.

    Returns
    -------
    float
        The total compounded net PnL.

    Raises
    ------
    ValidationError
        If ``net_returns`` is empty or non-finite.
    """
    arr = _coerce_series(net_returns, name="net_returns")
    return float(np.prod(1.0 + arr) - 1.0)


def strategy_metrics(
    net_returns: FloatArray,
    weights: WeightPath,
    *,
    initial_weights: FloatArray | None = None,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> StrategyMetrics:
    """Assemble the full OOS metric bundle (Sharpe, drawdown, turnover, net PnL).

    Parameters
    ----------
    net_returns:
        The per-bar portfolio NET return series.
    weights:
        The per-bar weight path (for turnover).
    initial_weights:
        The weight vector held before the first bar; ``None`` => flat.
    periods_per_year:
        Annualization factor for the Sharpe.

    Returns
    -------
    StrategyMetrics
        The frozen metric bundle.

    Raises
    ------
    ValidationError
        If the inputs are empty / non-finite / length-mismatched.
    """
    net = _coerce_series(net_returns, name="net_returns")
    w = np.asarray(weights, dtype="float64")
    if w.ndim == 1:
        w = w.reshape(-1, 1)
    if int(w.shape[0]) != int(net.size):
        raise ValidationError(
            f"net_returns (len {net.size}) and weights (rows {w.shape[0]}) must have the "
            "same number of scored bars."
        )
    return StrategyMetrics(
        oos_sharpe=oos_sharpe(net, periods_per_year=periods_per_year),
        max_drawdown=max_drawdown(net),
        turnover=turnover(w, initial_weights=initial_weights),
        net_pnl=net_pnl(net),
        n_bars=int(net.size),
    )


def hac_standard_error(series: FloatArray, *, lag: int | None = None) -> float:
    """Newey-West HAC standard error of the sample mean of ``series``.

    Uses Bartlett weights; ``lag=None`` selects the Andrews (1991) automatic
    truncation ``ceil(4 * (T/100)**(2/9))``. Used to build the Diebold-Mariano
    statistic's denominator from the per-bar RL-vs-best-baseline net-return
    differential.

    Parameters
    ----------
    series:
        A 1-D per-bar series (e.g. the DM differential).
    lag:
        Bartlett lag truncation; ``None`` => Andrews rule.

    Returns
    -------
    float
        ``sqrt(omega_hat / T)``, the HAC standard error of the mean.

    Raises
    ------
    ValidationError
        If ``series`` has fewer than two finite observations or ``lag < 0``.
    """
    # quantcore-candidate: Newey-West, Bartlett, Andrews lag.
    arr = np.asarray(series, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    t = arr.size
    if t < 2:
        raise ValidationError("hac_standard_error needs at least two finite observations.")
    if lag is None:
        lag = andrews_lag(t)
    if lag < 0:
        raise ValidationError(f"hac_standard_error: lag must be non-negative, got {lag}.")

    centred = arr - arr.mean()
    gamma0 = float(np.dot(centred, centred) / t)
    omega = gamma0
    max_lag = min(lag, t - 1)
    for h in range(1, max_lag + 1):
        weight = 1.0 - h / (lag + 1.0)
        gamma_h = float(np.dot(centred[h:], centred[:-h]) / t)
        omega += 2.0 * weight * gamma_h
    omega = max(omega, 0.0)
    return float(np.sqrt(omega / t))


def andrews_lag(t: int) -> int:
    """Andrews (1991) automatic Bartlett lag truncation ``ceil(4*(T/100)**(2/9))``.

    Parameters
    ----------
    t:
        Sample size (must be positive).

    Returns
    -------
    int
        The non-negative lag truncation.

    Raises
    ------
    ValidationError
        If ``t <= 0``.
    """
    if t <= 0:
        raise ValidationError(f"andrews_lag: t must be positive, got {t}.")
    return math.ceil(4.0 * math.pow(t / 100.0, 2.0 / 9.0))
