"""Probabilistic and Deflated Sharpe ratios (Bailey & Lopez de Prado, 2014).

These overfitting guards adjust a realized Sharpe ratio for sample length,
non-normality (skew and kurtosis), and - for the Deflated Sharpe - the number of
configurations tried (multiple-testing / selection bias). The Deflated Sharpe is
the honest yardstick that counts the FULL configuration grid as ``n_trials``.

MIGRATED to the shared ``quantcore`` package: the PSR/DSR kernels here are
byte-identical to ``quantcore.probabilistic_sharpe_ratio`` /
``quantcore.deflated_sharpe_ratio`` (the Acklam ``_norm_ppf`` + ``erf`` form,
validated to 1e-8 against ``scipy.stats.norm`` in the parity suite). Rather than
maintain a second copy that can drift, these public names are thin wrappers over
quantcore that translate ``quantcore.ValidationError`` into this package's own
:class:`stockclusters._exceptions.ValidationError` with IDENTICAL messages (the
two packages have no shared exception ancestry, so callers that ``except
stockclusters ... ValidationError`` keep working unchanged).

Importing this module has no side effects.
"""

from __future__ import annotations

import quantcore as _qc

from stockclusters._exceptions import ValidationError

__all__ = ["deflated_sharpe_ratio", "probabilistic_sharpe_ratio"]

# quantcore-candidate: DONE — re-exported from quantcore (kernels byte-identical;
# cross-checked to pairs-trading:evaluation/dsr.py and ma-crossover-backtest for
# the (k+2)/4 term). The two honest-input helpers
# (``quantcore.variance_of_trial_sharpes`` / ``quantcore.effective_n_trials``)
# fix the V=0.0 / n_trials=1 footguns at the call sites.

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

    Returns

    .. math::

        \text{PSR} = \Phi\!\left(
            \frac{(\widehat{SR} - SR^\*)\sqrt{n - 1}}
                 {\sqrt{1 - \gamma_3\,\widehat{SR} + \frac{\gamma_4 - 1}{4}\widehat{SR}^2}}
        \right),

    where :math:`\widehat{SR}` is the (non-annualized, per-observation) observed
    Sharpe, :math:`SR^\*` the benchmark Sharpe, :math:`\gamma_3` the skewness,
    :math:`\gamma_4` the kurtosis, and :math:`\Phi` the standard-normal CDF.

    HONESTY REQUIREMENT: ``kurtosis`` here is the **full** (non-excess) kurtosis,
    so a Gaussian has ``kurtosis=3`` and the bracket uses :math:`(\gamma_4 - 1)/4`.
    The excess-vs-full-kurtosis mix-up is a known PSR footgun and is rejected.

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
        If ``n_obs < 2``.
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
        # Translate to this package's ValidationError (no shared ancestry) keeping
        # the IDENTICAL message so callers' catch semantics and messages are intact.
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

    The DSR is the PSR evaluated against an *expected-maximum* benchmark Sharpe
    that grows with the number of independent trials :math:`N`:

    .. math::

        SR^\*_0 = \sqrt{V}\left[(1 - \gamma)\,\Phi^{-1}\!\left(1 - \tfrac{1}{N}\right)
                  + \gamma\,\Phi^{-1}\!\left(1 - \tfrac{1}{N}e^{-1}\right)\right],

    where :math:`V` is the variance of the trial Sharpe ratios, :math:`\gamma`
    the Euler-Mascheroni constant, and :math:`N` = ``n_trials``. The DSR is then
    ``probabilistic_sharpe_ratio(observed_sharpe, ..., benchmark_sharpe=SR*_0)``.

    HONESTY REQUIREMENT: ``n_trials`` must count the FULL explored configuration
    grid (#allocators x #linkages x #covariance-estimators x #rmt(on/off) x
    #rebalance-freqs x #cost-levels x #lookback-windows). The PSR uses the FULL
    ``(\gamma_4)`` kurtosis term. The DSR is non-increasing in ``n_trials``
    (monotonicity asserted in the property suite).

    ``variance_of_trial_sharpes`` (``V``) must be the REAL cross-trial variance of
    the per-observation trial Sharpes (use ``quantcore.variance_of_trial_sharpes``
    or ``quantcore.expected_sharpe_variance``), never a hardcoded ``0.0`` -- with
    ``V == 0`` or ``N == 1`` the benchmark collapses to ``0`` and the DSR
    degenerates to the plain PSR-against-zero, i.e. the multiplicity correction is
    silently disabled.

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
        return _qc.deflated_sharpe_ratio(
            observed_sharpe,
            n_obs=n_obs,
            n_trials=n_trials,
            variance_of_trial_sharpes=variance_of_trial_sharpes,
            skew=skew,
            kurtosis=kurtosis,
        )
    except _qc.ValidationError as exc:
        # Translate to this package's ValidationError (no shared ancestry) keeping
        # the IDENTICAL message so callers' catch semantics and messages are intact.
        raise ValidationError(str(exc)) from exc
