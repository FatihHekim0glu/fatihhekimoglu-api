"""Side-by-side regression tables and tidy coefficient summaries."""

from __future__ import annotations

import pandas as pd
from statsmodels.iolib.summary2 import summary_col

_DEFAULT_ORDER: list[str] = ["const", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]
_VALID_FMTS: tuple[str, ...] = ("text", "latex", "html")


def significance_stars(p: float) -> str:
    """Return ``***``/``**``/``*``/``''`` for the usual 0.01/0.05/0.10 cutoffs."""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def _info_dict() -> dict[str, callable]:
    """Footer rows for summary_col with N, Adj R-squared, and F-statistic."""
    return {
        "N": lambda r: f"{int(r.nobs)}",
        "Adj R2": lambda r: f"{float(r.rsquared_adj):.4f}",
        "F-stat": lambda r: f"{float(r.fvalue):.2f}",
    }


def regression_table(
    results_list: list,
    model_names: list[str],
    regressor_order: list[str] | None = None,
    fmt: str = "text",
) -> str:
    """Render a side-by-side comparison table for multiple regressions."""
    if fmt not in _VALID_FMTS:
        raise ValueError(f"fmt must be one of {_VALID_FMTS}, got {fmt!r}")
    if len(results_list) != len(model_names):
        raise ValueError(
            f"results_list and model_names length mismatch: "
            f"{len(results_list)} vs {len(model_names)}"
        )

    order = list(regressor_order) if regressor_order is not None else list(_DEFAULT_ORDER)

    tbl = summary_col(
        results_list,
        model_names=list(model_names),
        stars=True,
        float_format="%.4f",
        regressor_order=order,
        info_dict=_info_dict(),
    )

    if fmt == "latex":
        return tbl.as_latex()
    if fmt == "html":
        return tbl.as_html()
    return str(tbl)


def coef_table(reg) -> pd.DataFrame:
    """Return a tidy [Factor, Coef, Std Err, t, p, Sig] table for a FactorRegression."""
    results = reg.results
    names = list(results.model.exog_names)
    params = [float(v) for v in results.params]
    bse = [float(v) for v in results.bse]
    tvals = [float(v) for v in results.tvalues]
    pvals = [float(v) for v in results.pvalues]
    sig = [significance_stars(p) for p in pvals]
    return pd.DataFrame(
        {
            "Factor": names,
            "Coef": params,
            "Std Err": bse,
            "t": tvals,
            "p": pvals,
            "Sig": sig,
        }
    )
