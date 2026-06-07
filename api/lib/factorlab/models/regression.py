"""Fama-French style factor regression with optional HAC standard errors."""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm

from factorlab.models.factor_sets import factors_for

_STAR_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (0.01, "***"),
    (0.05, "**"),
    (0.10, "*"),
)

_FACTOR_PHRASES: dict[str, tuple[str, str]] = {
    "Mkt-RF": ("market exposure", "defensive market exposure"),
    "SMB": ("small-cap tilt", "large-cap tilt"),
    "HML": ("value tilt", "growth tilt"),
    "MOM": ("momentum tilt", "contrarian tilt"),
    "RMW": ("profitability tilt", "weak-profitability tilt"),
    "CMA": ("conservative-investment tilt", "aggressive-investment tilt"),
}


def _stars(p: float) -> str:
    """Return significance stars for a p-value."""
    for threshold, mark in _STAR_THRESHOLDS:
        if p < threshold:
            return mark
    return ""


def _periods_per_year(index: pd.Index) -> int:
    """Infer 12 for monthly indices, 252 otherwise."""
    if isinstance(index, pd.PeriodIndex):
        freq = (index.freqstr or "").upper()
        return 12 if freq.startswith("M") else 252
    if isinstance(index, pd.DatetimeIndex):
        freq = (index.freqstr or "").upper() if index.freq is not None else ""
        if freq.startswith("M"):
            return 12
        inferred = pd.infer_freq(index) if len(index) >= 3 else None
        if inferred and inferred.upper().startswith("M"):
            return 12
        return 252
    return 12


class FactorRegression:
    """OLS regression of excess returns on a named Fama-French factor set."""

    def __init__(
        self,
        excess_returns: pd.Series,
        factors: pd.DataFrame,
        model: str = "FF3",
        hac: bool = True,
        hac_maxlags: int | None = None,
    ) -> None:
        cols = factors_for(model)
        missing = [c for c in cols if c not in factors.columns]
        if missing:
            raise ValueError(f"factors missing required columns: {missing}")

        y = excess_returns.rename("y")
        df = pd.concat([y, factors[cols]], axis=1, join="inner").dropna()
        if len(df) < len(cols) + 5:
            raise ValueError(
                f"insufficient observations: {len(df)} (need >= {len(cols) + 5})"
            )

        self.model_name = model
        self._cols = cols
        self._df = df
        self._hac = hac

        exog = sm.add_constant(df[cols], has_constant="add")
        self._exog_names: list[str] = list(exog.columns)
        assert self._exog_names == ["const", *cols]

        self._raw = sm.OLS(df["y"], exog).fit()
        if hac:
            n = int(self._raw.nobs)
            lags = (
                hac_maxlags
                if hac_maxlags is not None
                else max(1, int(4 * (n / 100) ** (2 / 9)))
            )
            self._res = self._raw.get_robustcov_results(
                "HAC", maxlags=lags, use_correction=True
            )
            self._hac_lags: int | None = lags
        else:
            self._res = self._raw
            self._hac_lags = None

    def _coef_series(self, values) -> pd.Series:
        """Wrap a positional coefficient array as a named Series."""
        return pd.Series(list(values), index=self._exog_names)

    @property
    def alpha(self) -> float:
        return float(self._res.params[0])

    @property
    def alpha_annualized(self) -> float:
        periods = _periods_per_year(self._df.index)
        return (1.0 + self.alpha) ** periods - 1.0

    @property
    def alpha_t(self) -> float:
        return float(self._res.tvalues[0])

    @property
    def alpha_p(self) -> float:
        return float(self._res.pvalues[0])

    @property
    def betas(self) -> pd.Series:
        return pd.Series(list(self._res.params[1:]), index=self._cols, name="beta")

    @property
    def se(self) -> pd.Series:
        return self._coef_series(self._res.bse).rename("se")

    @property
    def tvalues(self) -> pd.Series:
        return self._coef_series(self._res.tvalues).rename("t")

    @property
    def pvalues(self) -> pd.Series:
        return self._coef_series(self._res.pvalues).rename("p")

    @property
    def r2(self) -> float:
        return float(self._raw.rsquared)

    @property
    def r2_adj(self) -> float:
        return float(self._raw.rsquared_adj)

    @property
    def nobs(self) -> int:
        return int(self._raw.nobs)

    @property
    def residuals(self) -> pd.Series:
        return pd.Series(
            self._raw.resid.to_numpy(),
            index=self._df.index,
            name="residuals",
        )

    @property
    def fitted(self) -> pd.Series:
        return pd.Series(
            self._raw.fittedvalues.to_numpy(),
            index=self._df.index,
            name="fitted",
        )

    @property
    def results(self):
        """Statsmodels results wrapper, HAC-adjusted when hac=True."""
        return self._res

    @property
    def results_raw(self):
        """Unadjusted OLS results, used for R-squared and residual access."""
        return self._raw

    def summary(self) -> str:
        """Return the statsmodels summary as text."""
        return self._res.summary().as_text()

    def interpret(self, alpha_threshold: float = 0.05) -> str:
        """Plain-English markdown narrative of the regression result."""
        periods = _periods_per_year(self._df.index)
        ann = self.alpha_annualized
        alpha_p = self.alpha_p
        stars = _stars(alpha_p)
        verdict = (
            "statistically distinguishable from zero"
            if alpha_p < alpha_threshold
            else "not statistically distinguishable from zero"
        )
        se_series = self.se
        t_series = self.tvalues
        p_series = self.pvalues

        lines: list[str] = []
        lines.append(f"## {self.model_name} regression interpretation")
        lines.append("")
        lines.append(
            f"- Sample size: {self.nobs} observations "
            f"({periods} periods per year assumed)."
        )
        lines.append(
            f"- R-squared: {self.r2:.4f} (adjusted {self.r2_adj:.4f}); "
            f"the factor set explains {self.r2 * 100:.1f}% of return variation."
        )
        lines.append(
            f"- Alpha (per period): {self.alpha:.6f}, "
            f"annualized {ann:.4f}; t = {self.alpha_t:.2f}, "
            f"p = {alpha_p:.4f}{stars}. Alpha is {verdict} "
            f"at the {alpha_threshold:.2f} level."
        )
        if self._hac:
            lines.append(
                f"- Standard errors are HAC (Newey-West) with "
                f"{self._hac_lags} lag(s)."
            )
        else:
            lines.append("- Standard errors are classical OLS (no HAC correction).")

        lines.append("")
        lines.append("### Factor loadings")
        for name in self._cols:
            beta = float(self.betas[name])
            se_v = float(se_series[name])
            t_v = float(t_series[name])
            p_v = float(p_series[name])
            sig_stars = _stars(p_v)
            phrases = _FACTOR_PHRASES.get(name)
            if phrases is not None:
                direction = phrases[0] if beta >= 0 else phrases[1]
            else:
                direction = "positive loading" if beta >= 0 else "negative loading"
            if p_v < alpha_threshold:
                sig_text = "significant"
            else:
                sig_text = "not significant"
            lines.append(
                f"- **{name}**: beta = {beta:+.4f} (se {se_v:.4f}, "
                f"t = {t_v:+.2f}, p = {p_v:.4f}{sig_stars}); "
                f"{direction}, {sig_text} at the {alpha_threshold:.2f} level."
            )

        return "\n".join(lines)
