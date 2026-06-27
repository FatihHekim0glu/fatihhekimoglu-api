"""Probabilistic and Deflated Sharpe ratios (Bailey & Lopez de Prado, 2014).

These overfitting guards adjust a realized Sharpe ratio for sample length,
non-normality (skew and kurtosis), and — for the Deflated Sharpe — the number of
configurations tried (multiple-testing / selection bias). The Deflated Sharpe is
the honest yardstick that counts the FULL configuration grid as ``n_trials``.

MIGRATED to the shared ``quantcore`` package: the PSR/DSR kernels here are
byte-identical to ``quantcore.probabilistic_sharpe_ratio`` /
``quantcore.deflated_sharpe_ratio`` (validated to 1e-8 against ``scipy.stats``).
These public names are thin wrappers over quantcore that translate
``quantcore.ValidationError`` into this package's own
:class:`hrp._exceptions.ValidationError` with IDENTICAL messages (the two
packages have no shared exception ancestry, so callers that ``except hrp ...
ValidationError`` keep working unchanged). The private ``_norm_cdf`` / ``_norm_ppf``
and ``_EULER_MASCHERONI`` names are re-exported from quantcore for any caller /
test that imports them directly.

Importing this module has no side effects.
"""

from __future__ import annotations

import quantcore as _qc
from quantcore.dsr import _norm_cdf, _norm_ppf  # noqa: F401  (re-export: kept for callers/parity)

from hrp._exceptions import ValidationError

__all__ = ["deflated_sharpe_ratio", "probabilistic_sharpe_ratio"]

# Euler-Mascheroni constant for the expected-maximum order statistic. Kept as a
# module-level name for backward compatibility; sourced from quantcore so the two
# packages cannot drift.
_EULER_MASCHERONI: float = _qc.EULER_MASCHERONI


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    *,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    benchmark_sharpe: float = 0.0,
) -> float:
    r"""Probabilistic Sharpe Ratio: P(true SR > benchmark) given the sample.

    Thin wrapper over :func:`quantcore.probabilistic_sharpe_ratio`. ``kurtosis``
    is the **full** (non-excess) kurtosis (Gaussian = ``3``). Returns the PSR in
    ``[0, 1]``.

    Raises
    ------
    ValidationError
        If ``n_obs < 2`` (or the bracket variance is non-positive).
    """
    try:
        return _qc.probabilistic_sharpe_ratio(
            observed_sharpe,
            n_obs=n_obs,
            skew=skew,
            kurtosis=kurtosis,
            benchmark_sharpe=benchmark_sharpe,
        )
    except _qc.ValidationError as exc:
        raise ValidationError(str(exc)) from exc


def deflated_sharpe_ratio(
    observed_sharpe: float,
    *,
    n_obs: int,
    n_trials: int,
    variance_of_trial_sharpes: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    r"""Deflated Sharpe Ratio: PSR against a multiplicity-inflated benchmark.

    Thin wrapper over :func:`quantcore.deflated_sharpe_ratio`. ``n_trials`` must
    count the FULL explored configuration grid; ``variance_of_trial_sharpes``
    (``V``) must be the REAL cross-trial variance of the per-observation trial
    Sharpes (use ``quantcore.variance_of_trial_sharpes``), never a hardcoded
    constant — with ``V == 0`` or ``N == 1`` the benchmark collapses to ``0`` and
    the DSR degenerates to the plain PSR. Returns the DSR in ``[0, 1]``;
    non-increasing in ``n_trials``.

    Raises
    ------
    ValidationError
        If ``n_obs < 2``, ``n_trials < 1``, or ``variance_of_trial_sharpes < 0``.
    """
    try:
        return _qc.deflated_sharpe_ratio(
            observed_sharpe,
            n_obs=n_obs,
            n_trials=n_trials,
            variance_of_trial_sharpes=variance_of_trial_sharpes,
            skew=skew,
            kurtosis=kurtosis,
        )
    except _qc.ValidationError as exc:
        raise ValidationError(str(exc)) from exc
