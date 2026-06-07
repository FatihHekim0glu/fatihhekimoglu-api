"""Factorlab tool — wraps factorlab's compute layer unchanged.

Endpoint:
  POST /tools/factorlab/run — fit a Fama-French factor regression for one
  ticker and return alpha (per-period + annualized), factor betas with
  HAC/OLS standard errors, t-stats, p-values, R-squared, observation count,
  the coefficient table, optional rolling-beta Plotly figures, and a
  plain-English markdown interpretation.

v1 is single-shot synchronous. Two network round-trips per request:

  - yfinance for prices (sub-second with a warm cache)
  - Dartmouth Ken French zip + AQR xlsx for factor data (slow on cold cache,
    ~instant after — the vendored loaders persist parquet under
    ``platformdirs.user_cache_dir('factorlab')`` for 25 days)

If long-running becomes a real concern we can hoist this onto a background
worker; the recon explicitly accepts a single-shot v1.
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import date
from typing import Any, Literal

import pandas as pd
import plotly.io as pio
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from ..deps import get_supabase
from ..lib.factorlab.data.aqr import load_aqr_factors
from ..lib.factorlab.data.align import prepare_regression_data
from ..lib.factorlab.data.famafrench import load_ff_factors
from ..lib.factorlab.models.factor_sets import factors_for
from ..lib.factorlab.models.regression import FactorRegression
from ..lib.factorlab.plots import rolling_beta_chart
from ..lib.factorlab.rolling import rolling_factor_exposure
from ..lib.factorlab.tables import coef_table

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools/factorlab", tags=["factorlab"])


# ---------------------------------------------------------------------------
# Constants & validation
# ---------------------------------------------------------------------------

_TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,15}$")

ModelName = Literal["CAPM", "FF3", "FF4", "FF5", "FF6"]
Frequency = Literal["M", "D"]


# CAPM/FF3/FF4 reuse the FF3 CSV; FF5/FF6 reuse the FF5 CSV. FF4/FF6 also
# require the MOM column, which load_ff_factors merges in automatically.
_MODEL_TO_FF_BASE: dict[str, str] = {
    "CAPM": "FF3",
    "FF3": "FF3",
    "FF4": "FF3",
    "FF5": "FF5",
    "FF6": "FF5",
}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class FactorlabRequest(BaseModel):
    ticker: str = Field("BRK-B", max_length=15)
    model: ModelName = "FF3"
    frequency: Frequency = "M"
    start: date = Field(default=date(2014, 1, 1))
    end: date = Field(default=date(2024, 12, 31))
    hac: bool = True

    @field_validator("ticker")
    @classmethod
    def _ticker_shape(cls, v: str) -> str:
        v = v.strip().upper()
        if not v:
            raise ValueError("ticker cannot be empty")
        if not _TICKER_RE.match(v):
            raise ValueError(
                "ticker must contain only A-Z, 0-9, '.', '-' (1-15 chars); "
                "examples: AAPL, BRK-B, BP.L"
            )
        return v

    @field_validator("end")
    @classmethod
    def _end_after_start(cls, v: date, info: Any) -> date:
        start = info.data.get("start")
        if start is not None and v <= start:
            raise ValueError("end must be strictly after start")
        return v


class CoefficientRow(BaseModel):
    factor: str
    beta: float | None
    se: float | None
    t: float | None
    p: float | None
    stars: str


class FactorlabSummary(BaseModel):
    model_config = {"protected_namespaces": ()}

    model: ModelName
    frequency: Frequency
    ticker: str
    start: date
    end: date
    nobs: int
    alpha: float | None
    alpha_annualized: float | None
    alpha_t: float | None
    alpha_p: float | None
    r_squared: float | None
    r_squared_adj: float | None
    hac: bool
    hac_lags: int | None


class RollingBetaFigure(BaseModel):
    factor: str
    figure: dict[str, Any] = Field(..., description="Plotly figure JSON: {data, layout}.")


class FactorlabResponse(BaseModel):
    summary: FactorlabSummary
    coefficients: list[CoefficientRow]
    rolling_beta_charts: list[RollingBetaFigure]
    interpretation_markdown: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value: Any) -> float | None:
    """Convert NaN / ±Inf / None to JSON-safe None."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _load_factors_for(model: str, frequency: str) -> pd.DataFrame:
    """Load the FF base + (optionally) AQR-MOM for the named model."""
    ff_base = _MODEL_TO_FF_BASE[model]
    factors = load_ff_factors(model=ff_base, freq=frequency)

    needed = set(factors_for(model))
    missing = [c for c in needed if c not in factors.columns]
    if not missing:
        return factors

    # AQR loader only returns monthly BAB/QMJ; the source streamlit app
    # actually relied on Ken French's MOM (which load_ff_factors already
    # appends). If MOM is still missing here something's wrong upstream.
    if "MOM" in missing:
        try:
            aqr = load_aqr_factors()
            if not aqr.empty and "MOM" in aqr.columns:
                factors = factors.join(aqr[["MOM"]], how="inner")
        except Exception:  # noqa: BLE001
            logger.exception("AQR fallback failed (non-fatal)")

    missing = [c for c in needed if c not in factors.columns]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"factor columns missing after load: {missing}",
        )
    return factors


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/run", response_model=FactorlabResponse)
def run(
    req: FactorlabRequest,
    supabase=Depends(get_supabase),
) -> FactorlabResponse:
    """Execute the factorlab regression pipeline.

    Pipeline mirrors the source dashboard:
      load_ff_factors → (optional) load_aqr_factors → prepare_regression_data
      → FactorRegression(...) → coef_table + rolling_beta_chart + interpret
    """
    cols = factors_for(req.model)

    # ----- factor data --------------------------------------------------
    try:
        factors = _load_factors_for(req.model, req.frequency)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("factor data load failed for %s/%s", req.model, req.frequency)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"factor data load failed: {exc}",
        ) from exc

    # ----- aligned regression frame -------------------------------------
    try:
        data = prepare_regression_data(
            ticker=req.ticker,
            factors=factors,
            start=str(req.start),
            end=str(req.end),
            freq=req.frequency,
            factor_cols=cols,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("prepare_regression_data failed for %s", req.ticker)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"data alignment failed for {req.ticker}: {exc}",
        ) from exc

    if data.empty or "excess_ret" not in data.columns:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"no usable regression data for {req.ticker}",
        )

    # ----- fit ----------------------------------------------------------
    try:
        reg = FactorRegression(
            excess_returns=data["excess_ret"],
            factors=data[cols],
            model=req.model,
            hac=req.hac,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("FactorRegression fit failed for %s", req.ticker)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"regression fit failed for {req.ticker}: {exc}",
        ) from exc

    # ----- coefficient table -------------------------------------------
    coefficients: list[CoefficientRow] = []
    try:
        coef_df = coef_table(reg)
        for _, row in coef_df.iterrows():
            coefficients.append(
                CoefficientRow(
                    factor=str(row["Factor"]),
                    beta=_safe_float(row["Coef"]),
                    se=_safe_float(row["Std Err"]),
                    t=_safe_float(row["t"]),
                    p=_safe_float(row["p"]),
                    stars=str(row["Sig"]),
                )
            )
    except Exception:  # noqa: BLE001
        logger.exception("coef_table failed (non-fatal)")

    # ----- rolling betas (best-effort) ---------------------------------
    rolling_charts: list[RollingBetaFigure] = []
    # Default window: 36 monthly / 252 daily — matches the source rolling.py
    # defaults and is short enough to fit inside typical sample sizes.
    window = 36 if req.frequency == "M" else 252
    if reg.nobs > window:
        try:
            rolling_df = rolling_factor_exposure(
                excess_returns=data["excess_ret"],
                factors=data[cols],
                window=window,
                model=req.model,
                hac=False,  # HAC inside a rolling loop is slow; keep classical
            )
            # rolling_beta_chart expects a 'date'-indexed frame
            if not rolling_df.empty:
                rolling_df = rolling_df.set_index("date")
                for factor in cols:
                    try:
                        fig = rolling_beta_chart(rolling_df, factor=factor)
                        figure_json = json.loads(pio.to_json(fig, validate=False))
                        rolling_charts.append(
                            RollingBetaFigure(factor=factor, figure=figure_json)
                        )
                    except Exception:  # noqa: BLE001
                        logger.exception("rolling_beta_chart failed for %s (non-fatal)", factor)
        except Exception:  # noqa: BLE001
            logger.exception("rolling_factor_exposure failed (non-fatal)")

    # ----- interpretation markdown -------------------------------------
    try:
        interpretation = reg.interpret()
    except Exception as exc:  # noqa: BLE001
        logger.exception("interpret() failed (non-fatal)")
        interpretation = f"Interpretation unavailable: {exc}"

    # ----- HAC lags (private attr; tolerate absence) -------------------
    hac_lags = getattr(reg, "_hac_lags", None)
    try:
        hac_lags = int(hac_lags) if hac_lags is not None else None
    except (TypeError, ValueError):
        hac_lags = None

    response = FactorlabResponse(
        summary=FactorlabSummary(
            model=req.model,
            frequency=req.frequency,
            ticker=req.ticker,
            start=req.start,
            end=req.end,
            nobs=int(reg.nobs),
            alpha=_safe_float(reg.alpha),
            alpha_annualized=_safe_float(reg.alpha_annualized),
            alpha_t=_safe_float(reg.alpha_t),
            alpha_p=_safe_float(reg.alpha_p),
            r_squared=_safe_float(reg.r2),
            r_squared_adj=_safe_float(reg.r2_adj),
            hac=req.hac,
            hac_lags=hac_lags,
        ),
        coefficients=coefficients,
        rolling_beta_charts=rolling_charts,
        interpretation_markdown=interpretation,
    )

    _maybe_log_run(supabase, req, response)
    return response


# ---------------------------------------------------------------------------
# Supabase helpers (best-effort)
# ---------------------------------------------------------------------------


def _maybe_log_run(supabase, req: FactorlabRequest, resp: FactorlabResponse) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "factorlab",
                "params": req.model_dump(mode="json"),
                "result": {
                    "summary": resp.summary.model_dump(mode="json"),
                    "n_coefficients": len(resp.coefficients),
                    "n_rolling_charts": len(resp.rolling_beta_charts),
                },
                "status": "ok",
            }
        ).execute()
    except Exception:  # noqa: BLE001
        logger.exception("tool_runs insert failed (non-fatal)")
