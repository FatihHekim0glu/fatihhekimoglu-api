"""Probabilistic and Deflated Sharpe ratios (Bailey & Lopez de Prado, 2014).

These overfitting guards adjust a realized Sharpe ratio for sample length,
non-normality (skew and kurtosis), and - for the Deflated Sharpe - the number of
configurations tried (multiple-testing / selection bias). The Deflated Sharpe is
the honest yardstick that counts the FULL configuration grid as ``n_trials``.

MIGRATED TO ``quantcore``. The PSR/DSR kernel (and the ``_norm_ppf`` /
``_norm_cdf`` helpers) now live in the shared, torch-free ``quantcore`` package
(:mod:`quantcore.dsr`), the single source of truth for the portfolio's
honest-statistics primitives. This module RE-EXPORTS them under their original
public names so every call site and ``regimehmm``'s public API are unchanged. The
kernel is byte-identical (parity verified to 0.0), so the migration is strictly
behavior-preserving.

The only adaptation is the exception TYPE: ``quantcore`` raises
:class:`quantcore.ValidationError` (a ``QuantCoreError`` subclass), whereas the
rest of ``regimehmm`` — and its test-suite ``pytest.raises(...)`` blocks — expect
:class:`regimehmm._exceptions.ValidationError` (an ``HRPError`` subclass). The
thin wrappers below translate the former to the latter with the IDENTICAL message
so the catch semantics (and the regression ``match=`` patterns) are preserved.

Importing this module has no side effects.
"""

from __future__ import annotations

from quantcore import ValidationError as _QuantCoreValidationError
from quantcore.dsr import _norm_cdf  # noqa: F401  (re-export: kept for parity / callers)
from quantcore.dsr import _norm_ppf as _qc_norm_ppf
from quantcore.dsr import deflated_sharpe_ratio as _qc_deflated_sharpe_ratio
from quantcore.dsr import probabilistic_sharpe_ratio as _qc_probabilistic_sharpe_ratio

from regimehmm._exceptions import ValidationError

__all__ = [
    "deflated_sharpe_ratio",
    "probabilistic_sharpe_ratio",
]


def _norm_ppf(p: float) -> float:
    """Standard-normal inverse CDF (re-export of :func:`quantcore.dsr._norm_ppf`).

    Thin wrapper that delegates to the shared quantcore kernel (numerically
    byte-identical) and only translates a domain failure from ``quantcore``'s
    :class:`quantcore.ValidationError` to :class:`regimehmm._exceptions.ValidationError`
    so the existing catch semantics (and the unit-test ``match=`` pattern) are
    preserved. Kept importable because the test-suite exercises it directly.
    """
    try:
        return _qc_norm_ppf(p)
    except _QuantCoreValidationError as exc:
        raise ValidationError(str(exc)) from exc


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    *,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    benchmark_sharpe: float = 0.0,
) -> float:
    r"""Probabilistic Sharpe Ratio: P(true SR > benchmark) given the sample.

    Thin re-export of :func:`quantcore.dsr.probabilistic_sharpe_ratio` (the kernel
    is byte-identical to the former local implementation). See the quantcore
    docstring for the full definition; the only behavioural adaptation is that a
    failed precondition is surfaced as :class:`regimehmm._exceptions.ValidationError`
    (with the identical message) rather than ``quantcore``'s own
    ``ValidationError``.

    Parameters
    ----------
    observed_sharpe:
        The observed per-observation (non-annualized) Sharpe ratio.
    n_obs:
        The number of return observations.
    skew:
        Sample skewness of the returns (``0`` for symmetric).
    kurtosis:
        Sample FULL kurtosis of the returns (``3`` for Gaussian).
    benchmark_sharpe:
        The per-observation benchmark Sharpe to test against (default ``0``).

    Returns
    -------
    float
        The probabilistic Sharpe ratio in ``[0, 1]``.

    Raises
    ------
    ValidationError
        If ``n_obs < 2`` or the bracket variance is non-positive.
    """
    try:
        return _qc_probabilistic_sharpe_ratio(
            observed_sharpe,
            n_obs=n_obs,
            skew=skew,
            kurtosis=kurtosis,
            benchmark_sharpe=benchmark_sharpe,
        )
    except _QuantCoreValidationError as exc:
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

    Thin re-export of :func:`quantcore.dsr.deflated_sharpe_ratio` (the kernel is
    byte-identical to the former local implementation). See the quantcore
    docstring for the full definition; the only behavioural adaptation is that a
    failed precondition is surfaced as :class:`regimehmm._exceptions.ValidationError`
    (with the identical message) rather than ``quantcore``'s own
    ``ValidationError``.

    HONESTY REQUIREMENT: ``n_trials`` must count the FULL explored configuration
    grid; the PSR uses the FULL (non-excess) kurtosis term. The DSR is
    non-increasing in ``n_trials``.

    Parameters
    ----------
    observed_sharpe:
        The observed per-observation (non-annualized) Sharpe ratio of the
        selected configuration.
    n_obs:
        The number of return observations.
    n_trials:
        The FULL number of configurations explored (the multiplicity count).
    variance_of_trial_sharpes:
        The cross-trial variance :math:`V` of the per-observation Sharpe ratios.
    skew:
        Sample skewness of the selected configuration's returns.
    kurtosis:
        Sample FULL kurtosis of the selected configuration's returns.

    Returns
    -------
    float
        The deflated Sharpe ratio in ``[0, 1]``.

    Raises
    ------
    ValidationError
        If ``n_obs < 2``, ``n_trials < 1``, or
        ``variance_of_trial_sharpes < 0``.
    """
    try:
        return _qc_deflated_sharpe_ratio(
            observed_sharpe,
            n_obs=n_obs,
            n_trials=n_trials,
            variance_of_trial_sharpes=variance_of_trial_sharpes,
            skew=skew,
            kurtosis=kurtosis,
        )
    except _QuantCoreValidationError as exc:
        raise ValidationError(str(exc)) from exc
