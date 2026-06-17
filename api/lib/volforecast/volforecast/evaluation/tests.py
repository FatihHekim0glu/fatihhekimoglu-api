"""Forecast-comparison inference: Diebold-Mariano, Hansen SPA, and HAC SEs.

Two complementary tools control for the fact that forecast losses are serially
correlated and that comparing many models inflates the chance of a spurious
"winner":

- **Diebold-Mariano** (1995): a PAIRWISE test of equal predictive accuracy on the
  loss differential, with a HAC (Newey-West / Bartlett) long-run variance so the
  test is valid under autocorrelated losses (Harvey-Leybourne-Newbold small-sample
  correction applied).
- **Hansen SPA** (2005): a test of the COMPOSITE null that NO model in the
  candidate set beats the benchmark, via a studentized-maximum statistic and a
  stationary bootstrap — this is the multiple-testing guard that stops us crowning
  ML a winner by data snooping. If ``arch`` is installed its
  :class:`arch.bootstrap.SPA` is used; otherwise a self-contained fallback runs.

Importing this module has no side effects (``arch`` is imported lazily inside the
SPA function). HAC variance is implemented in NumPy with the Andrews (1991)
automatic lag.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from volforecast._exceptions import ValidationError
from volforecast._rng import make_rng


def _andrews_lag(n: int) -> int:
    """Andrews (1991) automatic Bartlett lag ``ceil(4 (T/100)^{2/9})``."""
    return int(np.ceil(4.0 * (n / 100.0) ** (2.0 / 9.0)))


@dataclass(frozen=True, slots=True)
class DMResult:
    """Immutable result of a pairwise Diebold-Mariano test.

    Attributes
    ----------
    statistic:
        The (HLN small-sample-corrected) DM test statistic.
    p_value:
        The two-sided p-value (Student-t reference, ``T - 1`` df).
    mean_loss_diff:
        The mean loss differential ``L(model_a) - L(model_b)`` (negative ⇒ A
        better).
    n_obs:
        The number of aligned loss observations.
    favored:
        The label of the model with the lower mean loss (``"a"`` or ``"b"``).
    """

    statistic: float
    p_value: float
    mean_loss_diff: float
    n_obs: int
    favored: str

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        return {
            "statistic": float(self.statistic),
            "p_value": float(self.p_value),
            "mean_loss_diff": float(self.mean_loss_diff),
            "n_obs": int(self.n_obs),
            "favored": str(self.favored),
        }


@dataclass(frozen=True, slots=True)
class SPAResult:
    """Immutable result of a Hansen (2005) SPA test.

    Attributes
    ----------
    p_value_consistent, p_value_lower, p_value_upper:
        Hansen's three SPA p-values for the composite null that no candidate
        beats the benchmark; ``consistent`` is the headline value.
    best_model:
        The label of the candidate with the lowest mean loss vs the benchmark.
    n_models:
        The number of candidate models tested.
    n_boot:
        The number of bootstrap replicates.
    """

    p_value_consistent: float
    p_value_lower: float
    p_value_upper: float
    best_model: str
    n_models: int
    n_boot: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        return {
            "p_value_consistent": float(self.p_value_consistent),
            "p_value_lower": float(self.p_value_lower),
            "p_value_upper": float(self.p_value_upper),
            "best_model": str(self.best_model),
            "n_models": int(self.n_models),
            "n_boot": int(self.n_boot),
        }


def newey_west_lrv(x: NDArray[np.float64], *, lag: int | None = None) -> float:
    r"""Newey-West (1987) Bartlett long-run variance of a 1-D series.

    .. math::

        \widehat{\omega} = \gamma_0 + 2\sum_{h=1}^{L}
                            \left(1 - \tfrac{h}{L+1}\right)\gamma_h,

    with the Andrews (1991) automatic lag ``L = ceil(4 (T/100)^{2/9})`` when
    ``lag`` is ``None``. Used as the denominator of the DM statistic so it is
    valid under autocorrelated loss differentials.

    Parameters
    ----------
    x:
        A 1-D float array (e.g. the loss differential).
    lag:
        Bartlett truncation lag; ``None`` selects the Andrews rule.

    Returns
    -------
    float
        The (non-negative) long-run variance estimate.

    Raises
    ------
    ValidationError
        If ``x`` has fewer than two finite observations or ``lag < 0``.
    """
    arr = np.asarray(x, dtype="float64").ravel()
    arr = arr[np.isfinite(arr)]
    if arr.shape[0] < 2:
        raise ValidationError("newey_west_lrv requires at least two finite observations.")
    if lag is not None and lag < 0:
        raise ValidationError(f"newey_west_lrv requires lag >= 0, got {lag}.")

    n = int(arr.shape[0])
    truncation = _andrews_lag(n) if lag is None else int(lag)
    centred = arr - arr.mean()
    gamma0 = float(np.dot(centred, centred) / n)
    omega = gamma0
    max_lag = min(truncation, n - 1)
    for h in range(1, max_lag + 1):
        weight = 1.0 - h / (truncation + 1.0)
        gamma_h = float(np.dot(centred[h:], centred[:-h]) / n)
        omega += 2.0 * weight * gamma_h
    return max(omega, 0.0)


def diebold_mariano(
    loss_a: pd.Series | NDArray[np.float64],
    loss_b: pd.Series | NDArray[np.float64],
    *,
    label_a: str = "a",
    label_b: str = "b",
) -> DMResult:
    r"""Pairwise Diebold-Mariano test of equal predictive accuracy.

    Tests :math:`H_0:\ \mathbb{E}[d_t] = 0` for the loss differential
    :math:`d_t = L^{a}_t - L^{b}_t`, using the HAC long-run variance from
    :func:`newey_west_lrv` and the Harvey-Leybourne-Newbold (1997) small-sample
    correction. A negative ``mean_loss_diff`` favours model A.

    Parameters
    ----------
    loss_a, loss_b:
        Aligned per-observation loss series/arrays for the two models (e.g. from
        :func:`volforecast.evaluation.qlike.qlike_loss_series`).
    label_a, label_b:
        Human-readable labels recorded in the result's ``favored`` field.

    Returns
    -------
    DMResult
        The test statistic, two-sided p-value, mean loss differential, and the
        favoured model.

    Raises
    ------
    ValidationError
        If the two loss series differ in length or have fewer than two
        observations.
    """
    a = np.asarray(
        loss_a.to_numpy() if isinstance(loss_a, pd.Series) else loss_a, dtype="float64"
    ).ravel()
    b = np.asarray(
        loss_b.to_numpy() if isinstance(loss_b, pd.Series) else loss_b, dtype="float64"
    ).ravel()
    if a.shape[0] != b.shape[0]:
        raise ValidationError(
            f"loss_a and loss_b must have equal length, got {a.shape[0]} and {b.shape[0]}."
        )
    diff = a - b
    diff = diff[np.isfinite(diff)]
    n = int(diff.shape[0])
    if n < 2:
        raise ValidationError("diebold_mariano requires at least two finite loss differentials.")

    mean_diff = float(diff.mean())
    favored = label_a if mean_diff < 0.0 else label_b

    lrv = newey_west_lrv(diff)
    if lrv <= 0.0:
        # Degenerate (constant) loss differential: no detectable difference.
        statistic = 0.0
        p_value = 1.0
        return DMResult(
            statistic=statistic,
            p_value=p_value,
            mean_loss_diff=mean_diff,
            n_obs=n,
            favored=favored,
        )

    dm = mean_diff / math.sqrt(lrv / n)
    # Harvey-Leybourne-Newbold (1997) small-sample correction for h = 1 step
    # ahead: scale by sqrt((T - 1) / T) and refer to a Student-t with T - 1 df.
    hln = math.sqrt((n - 1) / n)
    statistic = dm * hln

    # Two-sided p-value against a Student-t(T - 1) reference, computed via the
    # regularized incomplete beta function (no SciPy dependency needed).
    df = n - 1
    p_value = _student_t_sf_two_sided(abs(statistic), df)

    return DMResult(
        statistic=float(statistic),
        p_value=float(p_value),
        mean_loss_diff=mean_diff,
        n_obs=n,
        favored=favored,
    )


def _student_t_sf_two_sided(t_abs: float, df: int) -> float:
    """Two-sided Student-t tail probability ``P(|T_df| >= t_abs)``.

    Uses the regularized incomplete beta identity
    ``P(|T| >= t) = I_{df/(df+t^2)}(df/2, 1/2)``, which avoids a SciPy import and
    is exact to double precision via :func:`math.lgamma`-based continued fraction.
    """
    if t_abs == 0.0:
        return 1.0
    x = df / (df + t_abs * t_abs)
    return float(_betainc(df / 2.0, 0.5, x))


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta ``I_x(a, b)`` (Numerical Recipes, Lentz)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(ln_beta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1.0e-30
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:  # pragma: no cover
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:  # pragma: no cover
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:  # pragma: no cover
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:  # pragma: no cover
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:  # pragma: no cover
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1.0e-12:
            break
    return h


def hansen_spa(
    losses: pd.DataFrame,
    benchmark_loss: pd.Series,
    *,
    n_boot: int = 999,
    seed: int = 7,
) -> SPAResult:
    """Hansen (2005) Superior Predictive Ability test over the model set.

    Tests the composite null that no candidate model has lower expected loss than
    the benchmark, controlling for data snooping across the whole set. Works in
    LOSS space: the loss differential is ``benchmark_loss - losses[model]`` so a
    positive mean means the model beats the benchmark. Uses ``arch.bootstrap.SPA``
    when available (lazy import), else a self-contained stationary-bootstrap
    fallback seeded from :func:`volforecast._rng.make_rng`.

    Parameters
    ----------
    losses:
        A ``(T, M)`` frame of per-observation losses, one column per candidate
        model.
    benchmark_loss:
        The length-``T`` benchmark per-observation loss (e.g. GARCH(1,1) or
        HAR-RV), aligned to ``losses``.
    n_boot:
        Number of bootstrap replicates (``> 0``).
    seed:
        Master seed for the bootstrap RNG.

    Returns
    -------
    SPAResult
        The consistent/lower/upper SPA p-values and the best candidate label.

    Raises
    ------
    ValidationError
        If ``losses`` is empty, misaligned, or ``n_boot <= 0``, or there are too
        few aligned observations.
    """
    if not isinstance(losses, pd.DataFrame):
        raise ValidationError("losses must be a pandas.DataFrame.")
    if not isinstance(benchmark_loss, pd.Series):
        raise ValidationError("benchmark_loss must be a pandas.Series.")
    if losses.shape[1] == 0:
        raise ValidationError("losses must have at least one candidate column.")
    if n_boot <= 0:
        raise ValidationError(f"n_boot must be positive, got {n_boot}.")

    # Inner-align on the common index, then drop any row with a NaN anywhere.
    common = losses.index.intersection(benchmark_loss.index)
    losses_aligned = losses.loc[common].astype("float64")
    bench_aligned = benchmark_loss.loc[common].astype("float64")
    mask = losses_aligned.notna().all(axis=1) & bench_aligned.notna()
    losses_aligned = losses_aligned.loc[mask]
    bench_aligned = bench_aligned.loc[mask]
    if losses_aligned.shape[0] < _SPA_MIN_OBS:
        raise ValidationError(
            f"hansen_spa needs at least {_SPA_MIN_OBS} aligned observations, "
            f"got {losses_aligned.shape[0]}."
        )

    # Excess in benefit space: benchmark_loss - model_loss. A POSITIVE mean means
    # the model has lower loss than the benchmark (i.e. it beats the benchmark).
    excess = bench_aligned.to_numpy(dtype="float64")[:, None] - losses_aligned.to_numpy(
        dtype="float64"
    )
    columns = [str(c) for c in losses_aligned.columns]
    generator = make_rng(seed)

    arch_result = _try_arch_spa(excess, columns, n_boot, generator)
    if arch_result is not None:
        return arch_result
    # No-arch container path: the self-contained fallback (covered directly by
    # ``test_fallback_spa_*``) runs instead. Unreachable when ``arch`` is present.
    return _fallback_spa(excess, columns, n_boot, generator)  # pragma: no cover


#: Minimum aligned observations the SPA bootstrap needs to be meaningful.
_SPA_MIN_OBS: int = 8


def _try_arch_spa(
    excess: NDArray[np.float64],
    columns: list[str],
    n_boot: int,
    generator: np.random.Generator,
) -> SPAResult | None:
    """Run ``arch.bootstrap.SPA`` when available; return ``None`` to fall back.

    ``arch`` works in LOSS space with a benchmark loss as the first argument and
    candidate losses as the rest. Our ``excess`` is benefit space
    (``benchmark - model``), so the loss-space inputs are ``benchmark = 0`` and
    candidate losses ``= -excess`` (lower loss ⇔ larger benefit).
    """
    try:  # pragma: no cover - exercised only when ``arch`` is installed
        from arch.bootstrap import SPA as _ArchSPA  # noqa: N811
    except Exception:  # pragma: no cover - exercised when ``arch`` missing
        return None

    try:  # pragma: no cover - depends on the optional ``arch`` package
        n_obs = excess.shape[0]
        benchmark = np.zeros(n_obs, dtype="float64")
        candidate_losses = -excess
        spa = _ArchSPA(
            benchmark,
            candidate_losses,
            reps=int(n_boot),
            seed=int(generator.integers(0, 2**31 - 1)),
        )
        spa.compute()
        pvals = spa.pvalues
        best_idx = int(np.argmax(excess.mean(axis=0)))
        return SPAResult(
            p_value_consistent=float(np.clip(float(pvals["consistent"]), 0.0, 1.0)),
            p_value_lower=float(np.clip(float(pvals["lower"]), 0.0, 1.0)),
            p_value_upper=float(np.clip(float(pvals["upper"]), 0.0, 1.0)),
            best_model=columns[best_idx],
            n_models=int(excess.shape[1]),
            n_boot=int(n_boot),
        )
    except Exception:  # pragma: no cover - any arch failure falls back
        return None


def _stationary_indices(n: int, block: int, generator: np.random.Generator) -> NDArray[np.int_]:
    """Politis-Romano (1994) stationary-bootstrap index sequence of length ``n``."""
    p = 1.0 / block
    idx = np.empty(n, dtype=np.int_)
    idx[0] = int(generator.integers(0, n))
    restart = generator.random(n) < p
    steps = generator.integers(0, n, size=n)
    for t in range(1, n):
        if restart[t]:
            idx[t] = int(steps[t])
        else:
            idx[t] = (idx[t - 1] + 1) % n
    return idx


def _fallback_spa(
    excess: NDArray[np.float64],
    columns: list[str],
    n_boot: int,
    generator: np.random.Generator,
) -> SPAResult:
    """Self-contained Hansen-SPA via the studentized max and a stationary bootstrap.

    Works in benefit space (``excess = benchmark_loss - model_loss``): a positive
    mean means the candidate beats the benchmark. The studentized maximum
    statistic and Hansen's consistent/lower/upper re-centrings reproduce
    :class:`arch.bootstrap.SPA` to bootstrap noise.
    """
    t, m = excess.shape
    mu = excess.mean(axis=0)
    sigma = excess.std(axis=0, ddof=1)
    sigma_safe = np.where(sigma > 0.0, sigma, 1.0)
    studentised = np.where(sigma > 0.0, np.sqrt(t) * mu / sigma_safe, -np.inf)
    observed = float(np.max(np.concatenate([studentised, [0.0]])))

    block = max(2, round(t ** (1.0 / 3.0)))
    threshold_consistent = -np.sqrt(2.0 * np.log(np.log(max(t, 3))) / t) * sigma_safe
    centring_lower = np.minimum(mu, 0.0)
    keep_cons = mu >= threshold_consistent
    centring_consistent = np.where(keep_cons, mu, 0.0)

    boot_max_cons = np.empty(n_boot, dtype="float64")
    boot_max_lower = np.empty(n_boot, dtype="float64")
    boot_max_upper = np.empty(n_boot, dtype="float64")
    for i in range(n_boot):
        idx = _stationary_indices(t, block, generator)
        sample_mean = excess[idx].mean(axis=0)
        stat_upper = np.where(sigma > 0.0, np.sqrt(t) * (sample_mean - mu) / sigma_safe, -np.inf)
        stat_lower = np.where(
            sigma > 0.0, np.sqrt(t) * (sample_mean - centring_lower) / sigma_safe, -np.inf
        )
        stat_cons = np.where(
            sigma > 0.0, np.sqrt(t) * (sample_mean - centring_consistent) / sigma_safe, -np.inf
        )
        boot_max_upper[i] = max(float(np.max(stat_upper)), 0.0)
        boot_max_lower[i] = max(float(np.max(stat_lower)), 0.0)
        boot_max_cons[i] = max(float(np.max(stat_cons)), 0.0)

    best_idx = int(np.argmax(studentised))
    return SPAResult(
        p_value_consistent=float(np.clip(float(np.mean(boot_max_cons >= observed)), 0.0, 1.0)),
        p_value_lower=float(np.clip(float(np.mean(boot_max_lower >= observed)), 0.0, 1.0)),
        p_value_upper=float(np.clip(float(np.mean(boot_max_upper >= observed)), 0.0, 1.0)),
        best_model=columns[best_idx],
        n_models=int(m),
        n_boot=int(n_boot),
    )
