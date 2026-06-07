"""Align asset returns with Fama-French factors for regression."""

from __future__ import annotations

import pandas as pd

from factorlab.data.prices import compute_returns, load_prices


def _to_month_period(df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Convert a DatetimeIndex to a monthly PeriodIndex; pass through PeriodIndex."""
    out = df.copy()
    idx = out.index
    if isinstance(idx, pd.PeriodIndex):
        if idx.freqstr != "M":
            out.index = idx.asfreq("M")
        return out
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        out.index = idx.to_period("M")
        return out
    raise TypeError(f"unsupported index type: {type(idx).__name__}")


def align_to_factor_index(
    returns: pd.Series,
    factors: pd.DataFrame,
    freq: str = "M",
) -> pd.DataFrame:
    """Inner-join via PeriodIndex (monthly) or NYSE session reindex (daily)."""
    rets = returns.copy()
    facs = factors.copy()
    if isinstance(rets.index, pd.DatetimeIndex) and rets.index.tz is not None:
        rets.index = rets.index.tz_localize(None)
    if isinstance(facs.index, pd.DatetimeIndex) and facs.index.tz is not None:
        facs.index = facs.index.tz_localize(None)

    if freq == "M":
        rets_p = _to_month_period(rets)
        facs_p = _to_month_period(facs)
        ret_name = rets_p.name if rets_p.name is not None else "ret"
        rets_p = rets_p.rename(ret_name)
        joined = facs_p.join(rets_p.rename("ret"), how="inner")
        return joined
    if freq == "D":
        if not isinstance(rets.index, pd.DatetimeIndex):
            raise TypeError("daily alignment requires DatetimeIndex on returns")
        reindexed = facs.reindex(rets.index).dropna(how="all")
        common = rets.index.intersection(reindexed.index)
        out = reindexed.loc[common].copy()
        out["ret"] = rets.loc[common]
        return out
    raise ValueError(f"unsupported freq: {freq}")


def prepare_regression_data(
    ticker: str,
    factors: pd.DataFrame,
    start,
    end,
    freq: str = "M",
    factor_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Build aligned regression frame with excess_ret, RF, and chosen factor columns."""
    if "RF" not in factors.columns:
        raise ValueError("factors must contain an 'RF' column")
    if factor_cols is None:
        factor_cols = [c for c in factors.columns if c not in ("RF", "ret")]
    missing = [c for c in factor_cols if c not in factors.columns]
    if missing:
        raise ValueError(f"missing factor columns: {missing}")

    prices = load_prices(ticker, start=start, end=end)
    rets = compute_returns(prices, freq=freq)
    rets.name = "ret"

    joined = align_to_factor_index(rets, factors[[*factor_cols, "RF"]], freq=freq)
    before = len(joined)
    joined = joined.dropna()
    after = len(joined)
    if before > 0 and (before - after) > 0.05 * before:
        raise ValueError(
            f"alignment dropped {before - after} of {before} rows (>5%)"
        )
    if after < 24:
        raise ValueError(f"insufficient observations after alignment: {after} < 24")

    joined["excess_ret"] = joined["ret"] - joined["RF"]
    joined = joined.drop(columns=["ret"])
    return joined[["excess_ret", "RF", *factor_cols]]
