"""Cross-sectional RANKING metrics (NO baseline-free in-sample R²/accuracy).

The honest comparison for next-period cross-sectional return prediction lives in
RANKING space, where per-period skill is measured against a baseline:

- :func:`rank_ic` — per-period Spearman rank correlation between a model's
  cross-sectional scores and the realized next-period forward returns (the
  information coefficient). Validated against ``scipy.stats.spearmanr``;
- :func:`long_short_spread` — the net-of-cost return of a long-top-decile /
  short-bottom-decile portfolio formed from the LAGGED scores (``signal.shift(1)``
  so the portfolio trades on lagged scores, never lookahead);
- :func:`hac_standard_error` / :func:`andrews_lag` — the Newey-West HAC long-run
  variance of a per-period series (e.g. the IC time series or the IC/return
  differential), reused to build the Diebold-Mariano denominator;
- :func:`cross_sectional_metrics` — assembles the frozen
  :class:`CrossSectionalMetrics` bundle.

DEBUNKED TRAP (documented once, never computed as a metric): a baseline-free
in-sample R²/classification accuracy looks deceptively high because message
passing trivially recovers the block/sector structure of the panel (a DESCRIPTIVE
fact) without any OOS ranking skill. We therefore NEVER report a baseline-free
R²/accuracy; all skill is judged by rank-IC and the long-short spread vs. the best
baseline.

Every builder here is pure numpy (no torch / sklearn / scipy at import or call),
so the Diebold-Mariano layer builds its HAC denominator torch-free and the serve
path computes the naive/ridge ranking metrics live. Importing this module has no
side effects.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from gnnstocks._exceptions import ValidationError
from gnnstocks._typing import FloatArray

# quantcore-candidate: HAC long-run variance mirrors
# mvts-forecast:evaluation/metrics.py (Newey-West, Bartlett, Andrews lag).


def _coerce_pair(
    scores: FloatArray,
    forward_returns: FloatArray,
    *,
    scores_name: str = "scores",
    returns_name: str = "forward_returns",
) -> tuple[FloatArray, FloatArray]:
    """Coerce a (scores, forward_returns) pair to aligned finite float64 vectors.

    Both inputs are flattened to 1-D, checked for non-emptiness, equal length, and
    finiteness. This is the single boundary every per-period metric funnels its
    cross-section through.

    Parameters
    ----------
    scores, forward_returns:
        The per-node model scores and realized next-period forward returns.
    scores_name, returns_name:
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
    s = np.asarray(scores, dtype=np.float64).ravel()
    r = np.asarray(forward_returns, dtype=np.float64).ravel()
    if s.size == 0 or r.size == 0:
        raise ValidationError(f"{scores_name} and {returns_name} must be non-empty.")
    if s.size != r.size:
        raise ValidationError(
            f"{scores_name} (len {s.size}) and {returns_name} (len {r.size}) "
            "must have the same length."
        )
    if not np.isfinite(s).all():
        raise ValidationError(f"{scores_name} contains non-finite values.")
    if not np.isfinite(r).all():
        raise ValidationError(f"{returns_name} contains non-finite values.")
    return s, r


@dataclass(frozen=True, slots=True)
class CrossSectionalMetrics:
    """Immutable bundle of OOS cross-sectional RANKING metrics (no baseline-free R²).

    Attributes
    ----------
    mean_rank_ic:
        Mean per-period Spearman rank-IC over the OOS test dates.
    ic_ir:
        The IC information ratio ``mean(IC) / std(IC)`` (per-period).
    ic_hac_tstat:
        The HAC ``mean(IC) / HAC_SE(IC)`` t-statistic of the IC time series.
    ls_mean_return:
        Mean per-period net-of-cost long-short decile-spread return.
    ls_sharpe:
        Per-period Sharpe of the long-short decile-spread net return.
    n_periods:
        Number of OOS rebalance periods scored.
    """

    mean_rank_ic: float
    ic_ir: float
    ic_hac_tstat: float
    ls_mean_return: float
    ls_sharpe: float
    n_periods: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of these metrics."""
        return asdict(self)


def rank_ic(scores: FloatArray, forward_returns: FloatArray) -> float:
    """Return the per-period Spearman rank-IC of ``scores`` vs. ``forward_returns``.

    The information coefficient is the Spearman rank correlation between the
    model's cross-sectional scores and the realized next-period forward returns.
    Validated against ``scipy.stats.spearmanr`` in the parity suite. A rank-IC
    near zero is the honest-null outcome on the synthetic block-factor panel.

    Parameters
    ----------
    scores:
        Per-node model scores for one rebalance date.
    forward_returns:
        Per-node realized next-period forward returns (same length).

    Returns
    -------
    float
        The Spearman rank correlation in ``[-1, 1]``.

    Raises
    ------
    ValidationError
        If the inputs are empty, length-mismatched, or degenerate (no rank
        variance).
    """
    s, r = _coerce_pair(scores, forward_returns)
    if s.size < 2:
        raise ValidationError("rank_ic needs at least two nodes in the cross-section.")
    s_rank = _average_ranks(s)
    r_rank = _average_ranks(r)
    s_centred = s_rank - s_rank.mean()
    r_centred = r_rank - r_rank.mean()
    s_var = float(np.dot(s_centred, s_centred))
    r_var = float(np.dot(r_centred, r_centred))
    if s_var <= 0.0 or r_var <= 0.0:
        # Every score (or every realized return) is identical => the ranks have no
        # dispersion and the Spearman correlation is undefined.
        raise ValidationError(
            "rank_ic: a degenerate cross-section (all scores or all returns tied) "
            "has no rank variance, so the Spearman rank-IC is undefined."
        )
    # Spearman's rho is the Pearson correlation of the (tie-averaged) ranks.
    return float(np.dot(s_centred, r_centred) / math.sqrt(s_var * r_var))


def _average_ranks(values: FloatArray) -> FloatArray:
    """Return tie-averaged ranks of ``values`` (the ``scipy.stats.rankdata`` rule).

    Ties receive the average of the ranks they span, which is exactly the
    convention ``scipy.stats.spearmanr`` uses, so :func:`rank_ic` matches SciPy to
    floating-point tolerance on tied and untied cross-sections alike.

    Parameters
    ----------
    values:
        A 1-D float64 array (already coerced).

    Returns
    -------
    FloatArray
        The 1-D array of average ranks (1-based), same length as ``values``.
    """
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    # Average the ranks of tied groups (equal values share their mean rank).
    sorted_values = values[order]
    start = 0
    for end in range(1, values.size + 1):
        if end == values.size or sorted_values[end] != sorted_values[start]:
            if end - start > 1:
                avg = ranks[order[start:end]].mean()
                ranks[order[start:end]] = avg
            start = end
    return ranks


def long_short_spread(
    scores: FloatArray,
    forward_returns: FloatArray,
    *,
    quantile: float = 0.1,
    cost_bps: float = 1.0,
    prev_scores: FloatArray | None = None,
) -> float:
    """Return the net-of-cost long-short decile-spread return for one period.

    Forms a long-top-``quantile`` / short-bottom-``quantile`` portfolio from the
    cross-sectional scores and earns the spread of the realized forward returns,
    net of a per-side basis-point cost charged on the turnover relative to
    ``prev_scores``. The scores are the LAGGED signal (the caller passes
    ``signal.shift(1)`` scores) so the portfolio never trades on same-period
    information.

    Parameters
    ----------
    scores:
        Per-node LAGGED model scores for the rebalance date.
    forward_returns:
        Per-node realized next-period forward returns.
    quantile:
        The decile fraction for the long/short legs (default ``0.1``).
    cost_bps:
        Per-side transaction cost in basis points charged on turnover.
    prev_scores:
        The previous period's scores for turnover; ``None`` => open from flat.

    Returns
    -------
    float
        The net-of-cost long-short spread return for the period.

    Raises
    ------
    ValidationError
        If inputs are empty/mismatched, ``quantile`` is out of ``(0, 0.5]``, or
        ``cost_bps`` is negative.
    """
    s, r = _coerce_pair(scores, forward_returns)
    if not math.isfinite(quantile) or not 0.0 < quantile <= 0.5:
        raise ValidationError(f"long_short_spread: quantile must be in (0, 0.5], got {quantile!r}.")
    if not math.isfinite(cost_bps) or cost_bps < 0.0:
        raise ValidationError(
            f"long_short_spread: cost_bps must be finite and non-negative, got {cost_bps!r}."
        )
    n = s.size
    k = max(1, math.floor(quantile * n))
    if 2 * k > n:
        raise ValidationError(
            f"long_short_spread: the cross-section of {n} nodes is too small for two "
            f"disjoint {quantile:g}-quantile legs (need >= {2 * k} nodes)."
        )

    weights = _long_short_weights(s, k)
    gross = float(np.dot(weights, r))

    # Per-side basis-point cost charged on one-way turnover relative to the
    # previous rebalance's weights. ``prev_scores=None`` opens the book from flat
    # (full turnover on the first period). One-way turnover is 0.5 * sum|Δw|.
    if prev_scores is None:
        prev_weights = np.zeros_like(weights)
    else:
        p = np.asarray(prev_scores, dtype=np.float64).ravel()
        if p.size != n:
            raise ValidationError(
                f"long_short_spread: prev_scores (len {p.size}) must match scores (len {n})."
            )
        if not np.isfinite(p).all():
            raise ValidationError("long_short_spread: prev_scores contains non-finite values.")
        prev_weights = _long_short_weights(p, k)

    turnover = 0.5 * float(np.abs(weights - prev_weights).sum())
    cost = turnover * cost_bps / 10_000.0
    return gross - cost


def _long_short_weights(scores: FloatArray, k: int) -> FloatArray:
    """Return equal-weight long-top-``k`` / short-bottom-``k`` portfolio weights.

    The ``k`` highest scores each receive ``+1/k`` and the ``k`` lowest each
    ``-1/k`` (a dollar-neutral, gross-leverage-2 long-short book), so the dot
    product with the realized forward returns is the decile-spread return. Ties at
    the decile boundary are broken by the stable argsort order; ``k`` is validated
    by the caller to leave the two legs disjoint.

    Parameters
    ----------
    scores:
        The per-node ranking scores (already coerced to 1-D float64).
    k:
        The number of nodes in each leg.

    Returns
    -------
    FloatArray
        The signed portfolio weights summing to zero with unit per-leg gross.
    """
    order = np.argsort(scores, kind="stable")
    weights = np.zeros(scores.size, dtype=np.float64)
    weights[order[-k:]] = 1.0 / k  # long the top-k scores
    weights[order[:k]] = -1.0 / k  # short the bottom-k scores
    return weights


def hac_standard_error(series: FloatArray, *, lag: int | None = None) -> float:
    """Newey-West HAC standard error of the sample mean of ``series``.

    Uses Bartlett weights; ``lag=None`` selects the Andrews (1991) automatic
    truncation ``ceil(4 * (T/100)**(2/9))``. Used to build the IC t-statistic and
    the Diebold-Mariano statistic's denominator from the per-period IC (or
    IC/return differential) series.

    Parameters
    ----------
    series:
        A 1-D per-period series (e.g. the IC time series or the DM differential).
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
    # quantcore-candidate: mirrors mvts-forecast:evaluation/metrics.py::hac_standard_error.
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


def cross_sectional_metrics(
    scores_by_period: list[FloatArray],
    forward_returns_by_period: list[FloatArray],
    *,
    quantile: float = 0.1,
    cost_bps: float = 1.0,
) -> CrossSectionalMetrics:
    """Assemble the full cross-sectional ranking metric bundle in one call.

    Computes the per-period rank-IC and net-of-cost long-short decile spread
    across all OOS rebalance periods, then summarizes them into a frozen
    :class:`CrossSectionalMetrics` (mean IC, IC information ratio, HAC IC
    t-statistic, mean long-short return, long-short Sharpe). Deliberately omits
    any baseline-free R²/accuracy.

    Parameters
    ----------
    scores_by_period:
        Per-period per-node model scores (one array per rebalance date).
    forward_returns_by_period:
        Per-period per-node realized forward returns (aligned to ``scores``).
    quantile:
        The decile fraction for the long/short legs.
    cost_bps:
        Per-side transaction cost in basis points.

    Returns
    -------
    CrossSectionalMetrics
        The frozen metric bundle.

    Raises
    ------
    ValidationError
        If the period lists are empty or misaligned.
    """
    if len(scores_by_period) == 0 or len(forward_returns_by_period) == 0:
        raise ValidationError("cross_sectional_metrics: the period lists must be non-empty.")
    if len(scores_by_period) != len(forward_returns_by_period):
        raise ValidationError(
            f"cross_sectional_metrics: {len(scores_by_period)} score periods vs. "
            f"{len(forward_returns_by_period)} return periods must match."
        )

    ic_series: list[float] = []
    ls_series: list[float] = []
    prev_scores: FloatArray | None = None
    for scores, fwd in zip(scores_by_period, forward_returns_by_period, strict=True):
        ic_series.append(rank_ic(scores, fwd))
        ls_series.append(
            long_short_spread(
                scores, fwd, quantile=quantile, cost_bps=cost_bps, prev_scores=prev_scores
            )
        )
        prev_scores = np.asarray(scores, dtype=np.float64).ravel()

    ic = np.asarray(ic_series, dtype=np.float64)
    ls = np.asarray(ls_series, dtype=np.float64)
    n_periods = ic.size

    mean_ic = float(ic.mean())
    ic_std = float(ic.std())
    ic_ir = mean_ic / ic_std if ic_std > 0.0 else 0.0
    # HAC t-stat of the IC time series needs >= 2 periods; with a single OOS
    # period the long-run variance is undefined and the t-stat is reported as 0.
    if n_periods >= 2:
        ic_se = hac_standard_error(ic)
        ic_hac_tstat = mean_ic / ic_se if ic_se > 0.0 else 0.0
    else:
        ic_hac_tstat = 0.0

    ls_mean = float(ls.mean())
    ls_std = float(ls.std())
    ls_sharpe = ls_mean / ls_std if ls_std > 0.0 else 0.0

    return CrossSectionalMetrics(
        mean_rank_ic=mean_ic,
        ic_ir=ic_ir,
        ic_hac_tstat=ic_hac_tstat,
        ls_mean_return=ls_mean,
        ls_sharpe=ls_sharpe,
        n_periods=int(n_periods),
    )
