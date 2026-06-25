"""Probabilistic and Deflated Sharpe ratios (Bailey & Lopez de Prado, 2014).

These overfitting guards adjust a realized Sharpe ratio for sample length,
non-normality (skew and kurtosis), and - for the Deflated Sharpe - the number of
configurations tried (multiple-testing / selection bias). The Deflated Sharpe is
the honest yardstick that counts the FULL configuration grid as ``n_trials``.

MIGRATED TO ``quantcore``. The PSR/DSR kernel (and the ``_norm_ppf`` /
``_norm_cdf`` helpers) now live in the shared, torch-free ``quantcore`` package
(:mod:`quantcore.dsr`), the single source of truth for the portfolio's
honest-statistics primitives. This module RE-EXPORTS them under their original
public names so every call site and ``volforecast``'s public API are unchanged.
The kernel is byte-identical (parity verified to 0.0), so the migration is
strictly behavior-preserving.

This module also re-exports the three honest-input helpers
(:func:`variance_of_trial_sharpes`, :func:`expected_sharpe_variance`,
:func:`effective_n_trials`) that the Deflated-Sharpe caller needs to supply a
REAL cross-trial variance ``V`` instead of a hardcoded constant (the overlay's
former ``V = 1.0`` bug; see :mod:`volforecast.backtest.overlay`).

The only adaptation is the exception TYPE: ``quantcore`` raises
:class:`quantcore.ValidationError` (a ``QuantCoreError`` subclass), whereas the
rest of ``volforecast`` - and its test-suite ``pytest.raises(...)`` blocks -
expect :class:`volforecast._exceptions.ValidationError` (a ``VolForecastError``
subclass). The thin wrappers below translate the former to the latter with the
IDENTICAL message so the catch semantics (and the regression ``match=``
patterns) are preserved.

Importing this module has no side effects.
"""

from __future__ import annotations

from quantcore import ValidationError as _QuantCoreValidationError
from quantcore.dsr import _norm_cdf  # noqa: F401  (re-export: kept for parity / callers)
from quantcore.dsr import _norm_ppf as _qc_norm_ppf
from quantcore.dsr import deflated_sharpe_ratio as _qc_deflated_sharpe_ratio
from quantcore.dsr import effective_n_trials as _qc_effective_n_trials
from quantcore.dsr import expected_sharpe_variance as _qc_expected_sharpe_variance
from quantcore.dsr import probabilistic_sharpe_ratio as _qc_probabilistic_sharpe_ratio
from quantcore.dsr import variance_of_trial_sharpes as _qc_variance_of_trial_sharpes

from volforecast._exceptions import ValidationError
from volforecast._typing import FloatArray

__all__ = [
    "deflated_sharpe_ratio",
    "effective_n_trials",
    "expected_sharpe_variance",
    "probabilistic_sharpe_ratio",
    "variance_of_trial_sharpes",
]

# Euler-Mascheroni constant for the expected-maximum order statistic (kept here as
# a module attribute for callers/tests that referenced it; the canonical value now
# lives in :data:`quantcore._constants.EULER_MASCHERONI`).
_EULER_MASCHERONI: float = 0.5772156649015329


def _norm_ppf(p: float) -> float:
    """Standard-normal inverse CDF (re-export of :func:`quantcore.dsr._norm_ppf`).

    Thin wrapper that delegates to the shared quantcore kernel (numerically
    byte-identical) and only translates a domain failure from ``quantcore``'s
    :class:`quantcore.ValidationError` to
    :class:`volforecast._exceptions.ValidationError` so the existing catch
    semantics (and the unit-test ``match=`` pattern) are preserved. Kept
    importable because the test-suite exercises it directly.
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
    failed precondition is surfaced as
    :class:`volforecast._exceptions.ValidationError` (with the identical message)
    rather than ``quantcore``'s own ``ValidationError``.

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
    failed precondition is surfaced as
    :class:`volforecast._exceptions.ValidationError` (with the identical message)
    rather than ``quantcore``'s own ``ValidationError``.

    HONESTY REQUIREMENT: ``n_trials`` must count the FULL explored configuration
    grid; ``variance_of_trial_sharpes`` (``V``) must be the REAL cross-trial
    variance (use :func:`variance_of_trial_sharpes` from a grid of trial Sharpes,
    or :func:`expected_sharpe_variance` as the single-series fallback) - never a
    hardcoded ``0.0`` / ``1.0`` / ``1/n``. The PSR uses the FULL (non-excess)
    kurtosis term. The DSR is non-increasing in ``n_trials``.

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


def variance_of_trial_sharpes(trial_sharpes: FloatArray) -> float:
    r"""Real cross-trial variance ``V`` from a set / matrix of trial Sharpes.

    Thin re-export of :func:`quantcore.dsr.variance_of_trial_sharpes` - the honest
    ``V`` the Deflated-Sharpe benchmark needs: the sample variance (``ddof=1``) of
    the **per-observation** Sharpe ratios of the trials that were actually run
    (one per configuration on the swept grid). Passing a hardcoded constant here
    is the most common DSR bug across the portfolio (``V = 0.0`` silently disables
    the deflation; ``V = 1.0`` over-deflates and pins the DSR low). A degenerate
    input (fewer than two finite trial Sharpes) returns ``0.0``.

    Parameters
    ----------
    trial_sharpes:
        The per-observation Sharpe ratios of the trials (1-D; one per swept
        configuration). NaN/inf entries are dropped before the variance.

    Returns
    -------
    float
        The sample cross-trial variance ``V >= 0`` (``0.0`` if fewer than two
        finite trial Sharpes survive).
    """
    return _qc_variance_of_trial_sharpes(trial_sharpes)


def expected_sharpe_variance(observed_sharpe: float, n_obs: int) -> float:
    r"""Analytic single-series proxy for the trial-Sharpe variance ``V``.

    Thin re-export of :func:`quantcore.dsr.expected_sharpe_variance`. When only
    ONE return series is available (no grid of trial Sharpes to take a cross-trial
    variance over), the asymptotic sampling variance of an estimated
    per-observation Sharpe, :math:`V \approx (1 + \tfrac12 \widehat{SR}^2)/n`, is a
    defensible, non-degenerate stand-in for the Deflated-Sharpe benchmark's ``V``
    (the documented single-series fallback). Prefer the real cross-trial
    :func:`variance_of_trial_sharpes` whenever a grid of trial Sharpes exists.

    Parameters
    ----------
    observed_sharpe:
        The observed **per-observation** Sharpe ratio.
    n_obs:
        The number of return observations.

    Returns
    -------
    float
        The analytic variance proxy ``V > 0``.

    Raises
    ------
    ValidationError
        If ``n_obs < 1``.
    """
    try:
        return _qc_expected_sharpe_variance(observed_sharpe, n_obs)
    except _QuantCoreValidationError as exc:
        raise ValidationError(str(exc)) from exc


def effective_n_trials(*grid_axis_sizes: int) -> int:
    """Honest multiplicity count = product of every swept-axis size.

    Thin re-export of :func:`quantcore.dsr.effective_n_trials`. The
    Deflated-Sharpe ``n_trials`` must count the FULL explored configuration grid:
    the product of the size of every swept axis. Each axis size must be ``>= 1``;
    the product is never silently collapsed to 1 (under-counting manufactures
    false significance - the same failure mode as ``V = 0``).

    Parameters
    ----------
    *grid_axis_sizes:
        The size of each swept axis (one positional argument per axis), each
        ``>= 1``. At least one axis is required.

    Returns
    -------
    int
        The product of the axis sizes (the honest "FULL grid" trial count).

    Raises
    ------
    ValidationError
        If no axes are given or any axis size is ``< 1``.
    """
    try:
        return _qc_effective_n_trials(*grid_axis_sizes)
    except _QuantCoreValidationError as exc:
        raise ValidationError(str(exc)) from exc
