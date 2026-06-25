"""Probabilistic and Deflated Sharpe ratios (Bailey & Lopez de Prado, 2014).

These overfitting guards adjust a realized Sharpe ratio for sample length,
non-normality (skew and kurtosis), and — for the Deflated Sharpe — the number of
configurations tried (multiple-testing / selection bias). The Deflated Sharpe is
the honest yardstick that counts the FULL configuration grid as ``n_trials``.

MIGRATED TO ``quantcore``. The PSR/DSR kernel and the honest-input ``V`` helper
(:func:`variance_of_trial_sharpes`) are now the single-source-of-truth
implementations in :mod:`quantcore.dsr` (byte-identical to the previously-vendored
copy, validated to 1e-8 against an independent ``scipy.stats.norm`` reference).
This module re-exports them under the original local public names, translating
:class:`quantcore.ValidationError` into :class:`mvtsforecast.ValidationError`
(the two have no shared ancestry) with IDENTICAL message strings so existing
``except ValidationError`` clauses and ``pytest.raises(ValidationError)`` tests are
unchanged. The private ``_norm_cdf`` / ``_norm_ppf`` helpers (imported by the
parity suite) and the ``_EULER_MASCHERONI`` constant are re-exported verbatim.

Importing this module has no side effects.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from quantcore import ValidationError as _QuantCoreValidationError
from quantcore.dsr import _norm_cdf as _norm_cdf  # re-export for the parity suite
from quantcore.dsr import _norm_ppf as _norm_ppf  # re-export for the parity suite
from quantcore.dsr import deflated_sharpe_ratio as _qc_deflated_sharpe_ratio
from quantcore.dsr import expected_sharpe_variance as _qc_expected_sharpe_variance
from quantcore.dsr import probabilistic_sharpe_ratio as _qc_probabilistic_sharpe_ratio
from quantcore.dsr import variance_of_trial_sharpes as _qc_variance_of_trial_sharpes

from mvtsforecast._exceptions import ValidationError

# Euler-Mascheroni constant for the expected-maximum order statistic. Kept as a
# module-level name for any caller/test that referenced it on the old vendored
# implementation; the live value now lives in ``quantcore._constants``.
_EULER_MASCHERONI: float = 0.5772156649015329

_F = TypeVar("_F", bound=Callable[..., Any])


def _translate_validation_error(func: _F) -> _F:
    """Wrap a quantcore callable, re-raising its ValidationError as mvts's.

    ``quantcore.ValidationError`` and ``mvtsforecast.ValidationError`` share no
    ancestry, so a bare quantcore call would leak an exception type the mvts
    callers (and tests) do not catch. This decorator re-raises with the SAME
    message string under mvts's own ``ValidationError`` so the public contract is
    byte-for-byte preserved.
    """

    @wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except _QuantCoreValidationError as exc:  # pragma: no cover - thin shim
            raise ValidationError(str(exc)) from exc

    return _wrapper  # type: ignore[return-value]


# Re-exported under the original local public names (signatures + docstrings live
# in quantcore.dsr; the wrapper only translates the exception type).
probabilistic_sharpe_ratio = _translate_validation_error(_qc_probabilistic_sharpe_ratio)
deflated_sharpe_ratio = _translate_validation_error(_qc_deflated_sharpe_ratio)
variance_of_trial_sharpes = _translate_validation_error(_qc_variance_of_trial_sharpes)
expected_sharpe_variance = _translate_validation_error(_qc_expected_sharpe_variance)

__all__ = [
    "_norm_cdf",
    "_norm_ppf",
    "deflated_sharpe_ratio",
    "expected_sharpe_variance",
    "probabilistic_sharpe_ratio",
    "variance_of_trial_sharpes",
]
