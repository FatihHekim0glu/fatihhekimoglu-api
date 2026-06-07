"""Rolling-window factor exposures and rolling alpha."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t as student_t
from statsmodels.regression.rolling import RollingOLS

from factorlab.models.factor_sets import factors_for


def _nw_maxlags(window: int) -> int:
    """Newey-West lag rule: max(1, int(4*(window/100)**(2/9)))."""
    return max(1, int(4 * (window / 100.0) ** (2.0 / 9.0)))


def _validate_inputs(
    excess_returns: pd.Series,
    factors: pd.DataFrame,
    window: int,
    model: str,
) -> list[str]:
    """Return the factor columns for the named model after sanity-checking inputs."""
    cols = factors_for(model)
    missing = [c for c in cols if c not in factors.columns]
    if missing:
        raise ValueError(f"factors frame missing columns for {model}: {missing}")
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if len(excess_returns) < window:
        raise ValueError(
            f"need at least {window} observations to fit window={window}, "
            f"got {len(excess_returns)}"
        )
    return cols


def rolling_factor_exposure(
    excess_returns: pd.Series,
    factors: pd.DataFrame,
    window: int = 36,
    model: str = "FF3",
    hac: bool = False,
    hac_maxlags: int | None = None,
) -> pd.DataFrame:
    """Rolling factor betas in long form with columns ['date','factor','beta','se']."""
    cols = _validate_inputs(excess_returns, factors, window, model)
    y = excess_returns.astype(float)
    x = factors[cols].astype(float).loc[y.index]
    x_const = sm.add_constant(x, has_constant="add")

    if not hac:
        res = RollingOLS(y, x_const, window=window).fit()
        params = res.params.drop(columns=["const"])
        bse = res.bse.drop(columns=["const"])
        params.index.name = "date"
        bse.index.name = "date"
        beta_long = params.stack(future_stack=True).rename("beta").reset_index()
        se_long = bse.stack(future_stack=True).rename("se").reset_index()
        beta_long.columns = ["date", "factor", "beta"]
        se_long.columns = ["date", "factor", "se"]
        merged = beta_long.merge(se_long, on=["date", "factor"], how="inner")
        return merged.dropna(subset=["beta", "se"]).reset_index(drop=True)

    maxlags = hac_maxlags if hac_maxlags is not None else _nw_maxlags(window)
    rows: list[dict[str, object]] = []
    idx = y.index
    y_arr = y.to_numpy()
    x_arr = x_const.to_numpy()
    names = list(x_const.columns)
    for t in range(window, len(y_arr) + 1):
        y_win = y_arr[t - window : t]
        x_win = x_arr[t - window : t]
        ols = sm.OLS(y_win, x_win, missing="drop").fit(
            cov_type="HAC", cov_kwds={"maxlags": maxlags}
        )
        end_date = idx[t - 1]
        for j, name in enumerate(names):
            if name == "const":
                continue
            rows.append(
                {
                    "date": end_date,
                    "factor": name,
                    "beta": float(ols.params[j]),
                    "se": float(ols.bse[j]),
                }
            )
    return pd.DataFrame(rows, columns=["date", "factor", "beta", "se"])


def rolling_alpha(
    excess_returns: pd.Series,
    factors: pd.DataFrame,
    window: int = 36,
    model: str = "FF3",
    hac: bool = False,
) -> pd.DataFrame:
    """Rolling intercept (alpha) with ['date','factor','beta','se','t','p'] columns."""
    cols = _validate_inputs(excess_returns, factors, window, model)
    y = excess_returns.astype(float)
    x = factors[cols].astype(float).loc[y.index]
    x_const = sm.add_constant(x, has_constant="add")
    k = len(cols)
    df_resid = window - k - 1
    if df_resid <= 0:
        raise ValueError(
            f"window={window} too small for {k} factors; need window > {k + 1}"
        )

    if not hac:
        res = RollingOLS(y, x_const, window=window).fit()
        alpha = res.params["const"].rename("beta")
        se = res.bse["const"].rename("se")
        out = pd.concat([alpha, se], axis=1).dropna()
        out["t"] = out["beta"] / out["se"]
        out["p"] = 2.0 * student_t.sf(np.abs(out["t"].to_numpy()), df_resid)
        out.index.name = "date"
        out = out.reset_index()
        out["factor"] = "const"
        return out[["date", "factor", "beta", "se", "t", "p"]]

    maxlags = _nw_maxlags(window)
    rows: list[dict[str, object]] = []
    idx = y.index
    y_arr = y.to_numpy()
    x_arr = x_const.to_numpy()
    const_pos = list(x_const.columns).index("const")
    for t in range(window, len(y_arr) + 1):
        y_win = y_arr[t - window : t]
        x_win = x_arr[t - window : t]
        ols = sm.OLS(y_win, x_win, missing="drop").fit(
            cov_type="HAC", cov_kwds={"maxlags": maxlags}
        )
        a = float(ols.params[const_pos])
        s = float(ols.bse[const_pos])
        tval = a / s if s != 0 else np.nan
        pval = float(2.0 * student_t.sf(np.abs(tval), df_resid)) if s != 0 else np.nan
        rows.append(
            {
                "date": idx[t - 1],
                "factor": "const",
                "beta": a,
                "se": s,
                "t": tval,
                "p": pval,
            }
        )
    return pd.DataFrame(rows, columns=["date", "factor", "beta", "se", "t", "p"])
