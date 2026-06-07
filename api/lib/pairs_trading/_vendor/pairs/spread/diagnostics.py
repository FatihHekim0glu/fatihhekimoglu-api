"""Battery of diagnostics for an OU spread fit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

from pairs._exceptions import InputError
from pairs.spread.results import OUDiagnostics, OUResult

__all__ = ["ou_diagnostics"]

_ALPHA: float = 0.05
_LJUNG_LAG: int = 10
_PHI_NEAR_UNITY: float = 0.999


def _ar1_slope_pvalue(spread: pd.Series) -> float:
    """Two-sided p-value for the AR(1) slope coefficient."""

    values = spread.dropna().to_numpy(dtype=np.float64)
    design = sm.add_constant(values[:-1], has_constant="add")
    fit = sm.OLS(values[1:], design).fit()
    return float(fit.pvalues[1])


def _adf_pvalue(spread: pd.Series) -> float:
    """ADF MacKinnon p-value at default lag selection."""

    values = spread.dropna().to_numpy(dtype=np.float64)
    result = adfuller(values, autolag="AIC")
    return float(result[1])


def _ljung_box_pvalue(residuals: pd.Series, lag: int) -> float:
    """Ljung-Box p-value at the configured lag."""

    n = residuals.shape[0]
    effective = max(1, min(int(lag), max(1, n // 4)))
    table = acorr_ljungbox(residuals.dropna(), lags=[effective], return_df=True)
    return float(table["lb_pvalue"].iloc[-1])


def ou_diagnostics(spread: pd.Series, ou_result: OUResult) -> OUDiagnostics:
    """Run a panel of sanity checks against an OU fit.

    Checks (in priority order):

    1. **phi significance** -- AR(1) slope p-value < 0.05 and ``phi < 0.999``.
    2. **ADF** -- spread is stationary at the 5% level.
    3. **Half-life range** -- ``1 <= H <= n / 3``.
    4. **Residual whiteness** -- Ljung-Box at lag 10 fails to reject at 5%.

    Parameters
    ----------
    spread : pandas.Series
        Original spread.
    ou_result : OUResult
        Fitted OU dynamics.

    Returns
    -------
    OUDiagnostics
        First failing check is recorded in ``reject_reason``; otherwise ``None``.
    """

    if not isinstance(spread, pd.Series):
        msg = "spread must be a pandas Series"
        raise InputError(msg)
    n = int(spread.dropna().shape[0])
    phi_pvalue = _ar1_slope_pvalue(spread)
    adf_pvalue = _adf_pvalue(spread)
    lb_pvalue = _ljung_box_pvalue(ou_result.residuals, _LJUNG_LAG)
    ratio = float(ou_result.half_life) / max(n, 1)
    reject: str | None = None
    if phi_pvalue >= _ALPHA or float(ou_result.phi) >= _PHI_NEAR_UNITY:
        reject = "phi_not_significant"
    elif adf_pvalue > _ALPHA:
        reject = "adf_nonstationary"
    elif float(ou_result.half_life) > n / 3.0:
        reject = "half_life_too_long"
    elif float(ou_result.half_life) < 1.0:
        reject = "half_life_too_short"
    elif lb_pvalue < _ALPHA:
        reject = "residual_autocorrelation"
    return OUDiagnostics(
        phi_significance_pvalue=float(phi_pvalue),
        adf_pvalue=float(adf_pvalue),
        ljung_box_pvalue=float(lb_pvalue),
        half_life_to_sample_ratio=float(ratio),
        reject_reason=reject,
    )
