"""Probabilistic and Deflated Sharpe ratios (Bailey & Lopez de Prado, 2014).

These overfitting guards adjust a realized Sharpe ratio for sample length,
non-normality (skew and kurtosis), and — for the Deflated Sharpe — the number of
configurations tried (multiple-testing / selection bias). The Deflated Sharpe is
the honest yardstick that counts the FULL configuration grid as ``n_trials``.

MIGRATED to the shared ``quantcore`` package (``quantcore.dsr``): the
Probabilistic/Deflated Sharpe kernels here are byte-identical to quantcore's
(proven exact ``==`` on a random-input grid), so this module re-exports them
behind the original ``rlallocator`` public names. The only adaptation is the
exception boundary — quantcore raises its OWN ``quantcore.ValidationError`` (no
shared ancestry with rlallocator's), so each call is wrapped to translate it to
:class:`rlallocator._exceptions.ValidationError` with the IDENTICAL message
string, preserving every caller's ``except ValidationError`` semantics.

Importing this module has no side effects.
"""

from __future__ import annotations

from quantcore import ValidationError as _QuantCoreValidationError
from quantcore.dsr import deflated_sharpe_ratio as _quantcore_deflated_sharpe_ratio
from quantcore.dsr import probabilistic_sharpe_ratio as _quantcore_probabilistic_sharpe_ratio

from rlallocator._exceptions import ValidationError

# quantcore-candidate (MIGRATED): PSR/DSR kernels mirror rl-trader:evaluation/dsr.py +
# hrp-portfolio:evaluation/dsr.py, now sourced from quantcore.dsr (byte-identical).

__all__ = ["deflated_sharpe_ratio", "probabilistic_sharpe_ratio"]


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
        return _quantcore_probabilistic_sharpe_ratio(
            observed_sharpe,
            n_obs=n_obs,
            skew=skew,
            kurtosis=kurtosis,
            benchmark_sharpe=benchmark_sharpe,
        )
    except _QuantCoreValidationError as exc:
        # quantcore's ValidationError shares no ancestry with rlallocator's; translate
        # it (preserving the IDENTICAL message) so callers' ``except ValidationError``
        # keep catching it.
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
    grid (#training-seeds x #HP configs). The PSR uses the FULL ``(\gamma_4)``
    kurtosis term. The DSR is non-increasing in ``n_trials`` (monotonicity asserted
    in the property suite). It is a PROBABILITY in ``[0, 1]``, NOT a Sharpe-units
    quantity — the verdict gate is ``> 1 - alpha`` (a confidence level), never
    ``> 0``.

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
        return _quantcore_deflated_sharpe_ratio(
            observed_sharpe,
            n_obs=n_obs,
            n_trials=n_trials,
            variance_of_trial_sharpes=variance_of_trial_sharpes,
            skew=skew,
            kurtosis=kurtosis,
        )
    except _QuantCoreValidationError as exc:
        # See probabilistic_sharpe_ratio: translate quantcore's ValidationError to
        # rlallocator's with the IDENTICAL message string.
        raise ValidationError(str(exc)) from exc
