"""The seed lottery: the across-seed OOS-Sharpe distribution (the overfit check).

The headline honesty mechanism. A PPO allocator is trained across N INDEPENDENT
seeds; each seed produces its own OOS net Sharpe. The DISPERSION of those Sharpes —
not any single-seed equity curve — is the result. This module summarizes that
distribution into a frozen :class:`SeedLotteryResult`:

- the MEDIAN-seed OOS Sharpe (the central tendency, used by the verdict);
- a bootstrap / empirical LOWER bound on the across-seed Sharpe (a percentile of
  the seed Sharpes, or a bootstrap CI lower edge);
- the spread (std + lo/hi) of the seed Sharpes.

The verdict requires the across-seed Sharpe LOWER BOUND to be ``> 0`` — if the
dispersion straddles zero, the apparent skill is a training-path lottery, not a
real edge. NEVER report a single seed's curve as if it were the result.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from rlallocator._exceptions import ValidationError
from rlallocator._rng import make_rng
from rlallocator._typing import FloatArray

# quantcore-candidate: mirrors rl-trader:src/rltrader/evaluation/seed_lottery.py —
# the across-seed percentile / bootstrap of the per-SEED OOS Sharpes (the
# training-path lottery).

#: Below this many seeds the bootstrap of resampled medians is degenerate (a
#: 1-seed "distribution" is a point mass), so the bounds fall back to the empirical
#: percentiles of the seed Sharpes themselves.
_MIN_SEEDS_FOR_BOOTSTRAP: int = 2


@dataclass(frozen=True, slots=True)
class SeedLotteryResult:
    """Immutable summary of the across-seed OOS-Sharpe distribution.

    Attributes
    ----------
    seed_sharpes:
        The per-seed OOS net Sharpe values (one per training seed), in seed order.
    median_sharpe:
        The MEDIAN across-seed OOS Sharpe (the verdict's central estimate).
    sharpe_lo:
        The across-seed Sharpe LOWER bound (bootstrap / empirical percentile). The
        verdict requires this to be strictly ``> 0``.
    sharpe_hi:
        The across-seed Sharpe UPPER bound (symmetric percentile).
    sharpe_std:
        The standard deviation of the per-seed Sharpes (the dispersion magnitude).
    n_seeds:
        The number of training seeds in the lottery.
    """

    seed_sharpes: tuple[float, ...]
    median_sharpe: float
    sharpe_lo: float
    sharpe_hi: float
    sharpe_std: float
    n_seeds: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        out = asdict(self)
        out["seed_sharpes"] = [float(x) for x in self.seed_sharpes]
        return out


def seed_lottery(
    seed_sharpes: FloatArray,
    *,
    alpha: float = 0.05,
    n_bootstrap: int = 2000,
    seed: int = 7,
) -> SeedLotteryResult:
    """Summarize the across-seed OOS-Sharpe distribution (median + dispersion + lower bound).

    Given the per-seed OOS net Sharpe values, returns the MEDIAN, a bootstrap /
    empirical ``alpha``-level LOWER bound (and the symmetric upper bound), and the
    seed-Sharpe standard deviation. The bootstrap resamples the per-seed Sharpes
    with replacement via :func:`rlallocator._rng.make_rng` (seeded, reproducible)
    and takes the ``alpha/2`` and ``1 - alpha/2`` percentiles of the resampled
    medians; with too few seeds it falls back to the empirical percentiles of the
    seed Sharpes themselves.

    Parameters
    ----------
    seed_sharpes:
        The per-seed OOS net Sharpe values (one per training seed).
    alpha:
        The two-sided level for the lower/upper bounds (default ``0.05`` => 95%).
    n_bootstrap:
        Number of bootstrap resamples for the across-seed CI.
    seed:
        Master RNG seed for the bootstrap (reproducible).

    Returns
    -------
    SeedLotteryResult
        The median, lower/upper bounds, dispersion, and the per-seed Sharpes.

    Raises
    ------
    ValidationError
        If ``seed_sharpes`` is empty, non-finite, or ``alpha`` is out of ``(0, 1)``.
    """
    sharpes = _coerce_seed_sharpes(seed_sharpes)
    if not 0.0 < alpha < 1.0:
        raise ValidationError(f"alpha must be in (0, 1), got {alpha}.")
    if n_bootstrap < 1:
        raise ValidationError(f"n_bootstrap must be >= 1, got {n_bootstrap}.")

    n = sharpes.size
    median = float(np.median(sharpes))
    # ddof=1 needs >= 2 points; a single-seed "lottery" has no dispersion estimate.
    sharpe_std = float(np.std(sharpes, ddof=1)) if n >= 2 else 0.0

    lo_pct = 100.0 * (alpha / 2.0)
    hi_pct = 100.0 * (1.0 - alpha / 2.0)

    if n < _MIN_SEEDS_FOR_BOOTSTRAP:
        # Too few seeds to bootstrap a distribution of medians: fall back to the
        # empirical percentiles of the seed Sharpes themselves (for a single seed
        # both edges collapse onto that seed's Sharpe — honestly un-narrowed).
        sharpe_lo = float(np.percentile(sharpes, lo_pct))
        sharpe_hi = float(np.percentile(sharpes, hi_pct))
    else:
        # Resample the per-seed Sharpes WITH REPLACEMENT and take the median of each
        # resample; the alpha/2 and 1-alpha/2 percentiles of those resampled medians
        # are the across-seed lower/upper bounds. Seeded via make_rng for byte-exact
        # reproducibility (no global numpy state).
        gen = make_rng(seed)
        idx = gen.integers(0, n, size=(n_bootstrap, n))
        resampled_medians = np.median(sharpes[idx], axis=1)
        sharpe_lo = float(np.percentile(resampled_medians, lo_pct))
        sharpe_hi = float(np.percentile(resampled_medians, hi_pct))

    return SeedLotteryResult(
        seed_sharpes=tuple(float(x) for x in sharpes),
        median_sharpe=median,
        sharpe_lo=sharpe_lo,
        sharpe_hi=sharpe_hi,
        sharpe_std=sharpe_std,
        n_seeds=int(n),
    )


def variance_of_seed_sharpes(seed_sharpes: FloatArray) -> float:
    """Return the cross-seed variance of the per-seed OOS Sharpes (for the DSR).

    The Deflated Sharpe's ``variance_of_trial_sharpes`` (the dispersion of the
    trial Sharpes across the seed x HP grid) is estimated from the per-seed Sharpe
    dispersion. Returns the sample variance (``ddof=1``) of ``seed_sharpes``.

    Parameters
    ----------
    seed_sharpes:
        The per-seed OOS net Sharpe values.

    Returns
    -------
    float
        The cross-seed sample variance of the per-seed Sharpes (``>= 0``).

    Raises
    ------
    ValidationError
        If ``seed_sharpes`` has fewer than two finite values.
    """
    sharpes = _coerce_seed_sharpes(seed_sharpes)
    if sharpes.size < 2:
        raise ValidationError(
            "variance_of_seed_sharpes needs at least two finite seed Sharpes to estimate "
            f"a dispersion, got {sharpes.size}."
        )
    return float(np.var(sharpes, ddof=1))


def _coerce_seed_sharpes(seed_sharpes: FloatArray) -> FloatArray:
    """Coerce per-seed Sharpes to a non-empty finite 1-D float64 vector.

    Parameters
    ----------
    seed_sharpes:
        The per-seed OOS net Sharpe values.

    Returns
    -------
    FloatArray
        The coerced 1-D float64 array.

    Raises
    ------
    ValidationError
        If ``seed_sharpes`` is empty or contains any non-finite value (a NaN
        Sharpe from a flat seed must be surfaced, never silently dropped).
    """
    arr = np.asarray(seed_sharpes, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValidationError("seed_sharpes must be non-empty.")
    if not np.isfinite(arr).all():
        raise ValidationError("seed_sharpes contains non-finite values.")
    return arr
