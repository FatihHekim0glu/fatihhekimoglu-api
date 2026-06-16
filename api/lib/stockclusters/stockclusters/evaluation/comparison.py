"""Sharpe-difference inference: Jobson-Korkie-Memmel + stationary block bootstrap.

The headline question — does HRP's OOS Sharpe beat 1/N's? — is answered with two
complementary tools: the Jobson-Korkie (1981) test with Memmel's (2003) correction
for the asymptotic standard error of the Sharpe difference, and a Politis-Romano
(1994) stationary block bootstrap that gives a confidence interval on the gap
without distributional assumptions.

Importing this module has no side effects.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from stockclusters._constants import PERIODS_PER_YEAR
from stockclusters._exceptions import ValidationError
from stockclusters._rng import make_rng
from stockclusters._typing import ReturnsLike
from stockclusters._validation import ensure_series

# quantcore-candidate: mirrors pairs-trading:evaluation/memmel.py (JKM) +
# pairs-trading:evaluation/bootstrap_ci.py (Politis-Romano).


def _norm_sf(x: float) -> float:
    """Standard-normal survival function ``P(Z > x)`` via ``erfc``."""
    # quantcore-candidate: 1 - Phi(x) = 0.5 * erfc(x / sqrt(2)).
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def _per_period_sharpe(excess: np.ndarray) -> float:
    """Per-period (non-annualized) Sharpe of an excess-return array."""
    std = float(excess.std(ddof=1))
    if std == 0.0 or not math.isfinite(std):
        return 0.0
    return float(excess.mean()) / std


def _aligned_excess(
    returns_a: ReturnsLike,
    returns_b: ReturnsLike,
    risk_free: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce, align, and convert two return series to excess-return arrays.

    Both series are coerced to 1-D float Series; if they carry comparable
    indexes they are aligned on the inner intersection, otherwise they are
    aligned positionally (and must share the same length).
    """
    series_a = ensure_series(returns_a, name="returns_a")
    series_b = ensure_series(returns_b, name="returns_b")

    # Prefer label alignment when both indexes overlap; fall back to positional.
    common = series_a.index.intersection(series_b.index)
    if len(common) > 0:
        common = common.sort_values()
        series_a = series_a.reindex(common)
        series_b = series_b.reindex(common)
    elif len(series_a) != len(series_b):
        raise ValidationError(
            "returns_a and returns_b share no common index labels and have "
            f"different lengths ({len(series_a)} vs {len(series_b)}); cannot align."
        )

    arr_a = series_a.to_numpy(dtype="float64") - risk_free
    arr_b = series_b.to_numpy(dtype="float64") - risk_free
    return arr_a, arr_b


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Immutable result of a Sharpe-difference comparison between two strategies.

    Attributes
    ----------
    sharpe_a:
        Annualized Sharpe of strategy A (e.g. HRP).
    sharpe_b:
        Annualized Sharpe of strategy B (e.g. 1/N).
    sharpe_gap:
        The difference ``sharpe_a - sharpe_b``.
    jkm_pvalue:
        Two-sided p-value of the Jobson-Korkie-Memmel test that the gap is zero.
    ci_low:
        Lower bound of the bootstrap confidence interval on the gap.
    ci_high:
        Upper bound of the bootstrap confidence interval on the gap.
    n_bootstrap:
        The number of bootstrap resamples used for the CI.
    """

    sharpe_a: float
    sharpe_b: float
    sharpe_gap: float
    jkm_pvalue: float
    ci_low: float
    ci_high: float
    n_bootstrap: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        return asdict(self)


def jobson_korkie_memmel(
    returns_a: ReturnsLike,
    returns_b: ReturnsLike,
    *,
    risk_free: float = 0.0,
) -> float:
    r"""Two-sided p-value of the Jobson-Korkie-Memmel Sharpe-difference test.

    Tests :math:`H_0: SR_a = SR_b` for two return series observed over the same
    sample. The test statistic uses Memmel's (2003) corrected asymptotic variance
    of the Sharpe difference,

    .. math::

        \widehat{\text{Var}}(\widehat{SR}_a - \widehat{SR}_b) =
        \frac{1}{T}\Big(2 - 2\rho_{ab}
            + \tfrac{1}{2}(SR_a^2 + SR_b^2 - 2\,SR_a SR_b \rho_{ab}^2)\Big),

    where :math:`\rho_{ab}` is the correlation between the two return series. The
    statistic is asymptotically standard-normal under :math:`H_0`.

    Validated against the Memmel (2003) closed form to ``1e-8`` in the parity
    suite.

    Parameters
    ----------
    returns_a, returns_b:
        Two per-period return series over the same sample (aligned internally).
    risk_free:
        Per-period risk-free rate subtracted in each Sharpe ratio.

    Returns
    -------
    float
        The two-sided p-value of the test.

    Raises
    ------
    ValidationError
        If the two series cannot be aligned or are too short.
    """
    arr_a, arr_b = _aligned_excess(returns_a, returns_b, risk_free)
    t = arr_a.shape[0]
    # Memmel's asymptotic variance needs an estimated correlation and at least a
    # handful of points for the standard deviations to be meaningful.
    if t < 3:
        raise ValidationError(
            f"jobson_korkie_memmel needs at least 3 aligned observations, got {t}."
        )

    sr_a = _per_period_sharpe(arr_a)
    sr_b = _per_period_sharpe(arr_b)

    # Contemporaneous correlation between the two return series.
    std_a = float(arr_a.std(ddof=1))
    std_b = float(arr_b.std(ddof=1))
    if std_a == 0.0 or std_b == 0.0:
        # A degenerate (zero-variance) series leaves the Sharpe difference
        # ill-defined; report no evidence against the null.
        return 1.0
    rho = float(np.corrcoef(arr_a, arr_b)[0, 1])
    if not math.isfinite(rho):
        rho = 0.0

    # Memmel (2003) corrected variance of (SR_a - SR_b), per-period Sharpes.
    var_diff = (
        2.0 - 2.0 * rho + 0.5 * (sr_a * sr_a + sr_b * sr_b - 2.0 * sr_a * sr_b * rho * rho)
    ) / t
    if var_diff <= 0.0:
        return 1.0

    z = (sr_a - sr_b) / math.sqrt(var_diff)
    # Two-sided p-value: 2 * P(Z > |z|).
    return 2.0 * _norm_sf(abs(z))


def block_bootstrap_sharpe_gap(
    returns_a: ReturnsLike,
    returns_b: ReturnsLike,
    *,
    n_bootstrap: int = 1000,
    block_size: int | None = None,
    confidence: float = 0.95,
    seed: int = 0,
) -> ComparisonResult:
    r"""Stationary block-bootstrap confidence interval on the Sharpe gap.

    Resamples the two return series JOINTLY (preserving their contemporaneous
    dependence) using the Politis-Romano (1994) stationary bootstrap — blocks of
    geometrically-distributed random length — and computes the Sharpe gap on each
    resample to build a percentile confidence interval.

    REPRODUCIBILITY: resampling draws from a seeded PCG64 generator
    (:func:`stockclusters._rng.make_rng`), so the CI is byte-identical for a fixed ``seed``.

    Parameters
    ----------
    returns_a, returns_b:
        Two per-period return series over the same sample (aligned internally).
    n_bootstrap:
        The number of bootstrap resamples (the OOM/latency watch-point on the
        hot path).
    block_size:
        The expected stationary-bootstrap block length. If ``None``, a
        data-driven default (e.g. :math:`T^{1/3}`) is used.
    confidence:
        The central confidence level (default ``0.95`` -> 2.5%/97.5% percentiles).
    seed:
        Master seed for the bootstrap RNG.

    Returns
    -------
    ComparisonResult
        The frozen comparison bundle (Sharpes, gap, JKM p-value, CI bounds).

    Raises
    ------
    ValidationError
        If the series cannot be aligned, ``n_bootstrap < 1``, or ``confidence``
        is not in ``(0, 1)``.
    """
    if n_bootstrap < 1:
        raise ValidationError(f"n_bootstrap must be >= 1, got {n_bootstrap}.")
    if not 0.0 < confidence < 1.0:
        raise ValidationError(f"confidence must be in (0, 1), got {confidence}.")

    arr_a, arr_b = _aligned_excess(returns_a, returns_b, risk_free=0.0)
    t = arr_a.shape[0]
    if t < 3:
        raise ValidationError(
            f"block_bootstrap_sharpe_gap needs at least 3 aligned observations, got {t}."
        )

    ann = math.sqrt(PERIODS_PER_YEAR)
    sharpe_a = _per_period_sharpe(arr_a) * ann
    sharpe_b = _per_period_sharpe(arr_b) * ann
    sharpe_gap = sharpe_a - sharpe_b

    # Politis-Romano (1994) stationary bootstrap: geometric block lengths with
    # mean ``block_size``; data-driven default of T**(1/3) when unset.
    eff_block = max(1, round(t ** (1.0 / 3.0))) if block_size is None else max(1, int(block_size))
    # Per-step restart probability of the geometric block scheme.
    p_restart = 1.0 / eff_block

    rng = make_rng(seed)
    gaps = np.empty(n_bootstrap, dtype="float64")

    for b in range(n_bootstrap):
        # Build a length-T index by walking blocks: start at a uniform position,
        # then with prob p_restart jump to a new uniform start, else step +1
        # (wrapping circularly). Both series share the SAME index to preserve
        # their contemporaneous dependence.
        idx = np.empty(t, dtype=np.intp)
        pos = int(rng.integers(0, t))
        idx[0] = pos
        restarts = np.asarray(rng.random(t), dtype="float64")
        steps = np.asarray(rng.integers(0, t, size=t), dtype=np.intp)
        for i in range(1, t):
            if restarts[i] < p_restart:
                pos = int(steps[i])
            else:
                pos = pos + 1
                if pos >= t:
                    pos = 0
            idx[i] = pos

        sample_a = arr_a[idx]
        sample_b = arr_b[idx]
        gaps[b] = (_per_period_sharpe(sample_a) - _per_period_sharpe(sample_b)) * ann

    tail = (1.0 - confidence) / 2.0
    ci_low = float(np.quantile(gaps, tail))
    ci_high = float(np.quantile(gaps, 1.0 - tail))

    jkm_pvalue = jobson_korkie_memmel(arr_a, arr_b, risk_free=0.0)

    return ComparisonResult(
        sharpe_a=sharpe_a,
        sharpe_b=sharpe_b,
        sharpe_gap=sharpe_gap,
        jkm_pvalue=jkm_pvalue,
        ci_low=ci_low,
        ci_high=ci_high,
        n_bootstrap=int(n_bootstrap),
        meta={
            "block_size": int(eff_block),
            "confidence": float(confidence),
            "n_obs": int(t),
            "seed": int(seed),
        },
    )
