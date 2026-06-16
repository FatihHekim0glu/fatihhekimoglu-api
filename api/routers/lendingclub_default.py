"""LendingClub default classifier tool — wraps the vendored ``lendingclub_default`` library.

Endpoint:
  POST /tools/lendingclub-default/run — score ONE loan application's probability
  of default at origination with a leakage-free, calibrated XGBoost model. Returns
  the calibrated PD, a risk decile, container-safe reason codes, and two Plotly
  figures (predicted-PD score distribution + a reliability/calibration curve).

Mirrors the honest-null discipline backing the ``lendingclub_default`` library: the
headline is ROC-AUC / PR-AUC / Brier — NEVER accuracy and NEVER profit/ROI. The
model *ranks* default risk; it does NOT predict which individuals default. Two
limits stand up front: (1) accepted-loans-only selection bias; (2) ``int_rate`` /
``grade`` are LendingClub's own risk model, so their importance is partly circular.
The shipped demo artifact is trained on a SYNTHETIC LC-schema panel so the tool is
reproducible without the proprietary Kaggle dump.

Lazy-load:
  - A module-level ``_BOOSTER`` sentinel holds the assembled scoring bundle. It is
    populated LAZILY on first request (never at import, never per request after the
    first). The two reference figures are computed once from a small seeded
    synthetic evaluation panel and cached alongside the bundle.

Caching:
  - Best-effort row in ``platform.tool_runs`` for run accounting; failures are
    logged but never propagate to the client.

Errors:
  - 422 on request validation problems surfaced by the library.
  - 502 on artifact-load / scoring failures.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Literal

import plotly.io as pio
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from ..deps import get_supabase

# Importing the vendor package side-effects ``sys.path`` so that
# ``import lendingclub_default`` resolves to the vendored copy. Keep this import
# even though it is only used for its side effect.
from ..lib import lendingclub_default as _vendor_marker  # noqa: F401

from lendingclub_default import (
    VALID_GRADES,
    VALID_TERMS,
    ValidationError,
    calibration_figure,
    load_booster,
    reliability_curve,
    score_distribution_figure,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools/lendingclub-default", tags=["lendingclub-default"])


# ---------------------------------------------------------------------------
# Vendored artifacts + lazy scoring bundle
# ---------------------------------------------------------------------------

#: Absolute path to the vendored ``<2MB`` booster + fitted pipeline + calibration.
_ARTIFACTS_DIR: Path = (
    Path(__file__).resolve().parent.parent
    / "lib"
    / "lendingclub_default"
    / "lendingclub_default"
    / "artifacts"
)

#: Module-level lazy-load sentinel. Populated on first request; never at import.
_BOOSTER: Any | None = None

#: Cached reference figures (built once from a seeded synthetic eval panel).
_SCORE_FIGURE: dict[str, Any] | None = None
_CALIBRATION_FIGURE: dict[str, Any] | None = None

#: Reason codes returned per scored application.
_TOP_K_REASON_CODES = 6

#: Size of the seeded synthetic evaluation panel used to build the reference
#: figures (kept small so the one-time build stays well inside the request budget).
_EVAL_PANEL_LOANS = 4_000
_EVAL_PANEL_SEED = 20260616

DataSource = Literal["synthetic", "cache"]
Term = Literal["36 months", "60 months"]


def _figure_json(fig: Any) -> dict[str, Any]:
    """Serialize a Plotly figure to a ``{data, layout}`` dict (API contract)."""
    payload: dict[str, Any] = json.loads(pio.to_json(fig, validate=False))
    return payload


def _build_reference_figures(bundle: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the score-distribution + calibration figures from a seeded eval panel.

    Scores a small, deterministic synthetic LC-schema panel through the shipped
    bundle and renders the two population-context figures the frontend tabs over:
    the predicted-PD histogram split by realised outcome, and the reliability
    curve. Built ONCE and cached; identical across requests for a given artifact.
    """
    from lendingclub_default import (
        SyntheticConfig,
        build_labels,
        drop_leakage,
        generate_synthetic_panel,
    )

    cfg = SyntheticConfig(n_loans=_EVAL_PANEL_LOANS, seed=_EVAL_PANEL_SEED)
    raw_panel = generate_synthetic_panel(cfg)
    labelled = build_labels(raw_panel)
    panel = labelled.panel
    y_true = labelled.y.to_numpy(dtype="float64")

    # The bundle drops leakage internally on ``score``; pass the labelled panel
    # (already free of in-progress loans) straight through.
    features_panel = drop_leakage(panel)
    pd_scores = bundle.score(features_panel)

    score_fig = score_distribution_figure(
        y_true,
        pd_scores,
        title="Predicted PD by realised outcome (synthetic eval panel)",
    )
    curve = reliability_curve(y_true, pd_scores, n_bins=10, strategy="quantile")
    calib_fig = calibration_figure(
        curve,
        title="Calibration (reliability) curve (synthetic eval panel)",
    )
    return _figure_json(score_fig), _figure_json(calib_fig)


def _ensure_loaded() -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Lazily load (and cache) the scoring bundle + reference figures.

    Returns ``(bundle, score_distribution_figure, calibration_figure)``. Raises a
    502 ``HTTPException`` if the committed artifacts cannot be loaded or the
    reference figures cannot be built.
    """
    global _BOOSTER, _SCORE_FIGURE, _CALIBRATION_FIGURE
    if _BOOSTER is not None and _SCORE_FIGURE is not None and _CALIBRATION_FIGURE is not None:
        return _BOOSTER, _SCORE_FIGURE, _CALIBRATION_FIGURE
    try:
        bundle = load_booster(_ARTIFACTS_DIR, use_cache=True)
        score_fig, calib_fig = _build_reference_figures(bundle)
    except Exception as exc:  # noqa: BLE001 — surfaced to the client as a 502
        logger.exception("lendingclub-default artifact load / figure build failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "LendingClub default model artifacts could not be loaded: "
                f"{exc.__class__.__name__}: {exc}"
            ),
        ) from exc
    _BOOSTER = bundle
    _SCORE_FIGURE = score_fig
    _CALIBRATION_FIGURE = calib_fig
    return _BOOSTER, _SCORE_FIGURE, _CALIBRATION_FIGURE


def _reset_cache() -> None:
    """Clear the lazy bundle + figure caches (used by tests for isolation)."""
    global _BOOSTER, _SCORE_FIGURE, _CALIBRATION_FIGURE
    _BOOSTER = None
    _SCORE_FIGURE = None
    _CALIBRATION_FIGURE = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value: float | None) -> float | None:
    """Coerce to a finite ``float`` or ``None`` (drops NaN/inf for JSON safety)."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class LendingClubRequest(BaseModel):
    """Wire contract for ONE loan application scored at origination.

    Every field is an APPLICATION-TIME (origination) feature — no post-funding
    column is accepted, so the wire contract cannot smuggle leakage past the
    leakage-free model.
    """

    loan_amnt: float = Field(15_000.0, gt=0.0, description="Requested loan amount (USD).")
    term: Term = Field("36 months", description="Loan term: '36 months' or '60 months'.")
    int_rate: float = Field(13.5, ge=0.0, le=100.0, description="Interest rate (annual %).")
    grade: str = Field("C", description="LendingClub credit grade A-G.")
    sub_grade: str = Field("C2", description="LendingClub sub-grade, e.g. 'C2'.")
    emp_length: float = Field(5.0, ge=0.0, le=50.0, description="Employment length (years).")
    home_ownership: str = Field(
        "MORTGAGE", description="Home ownership: RENT/OWN/MORTGAGE/OTHER/NONE/ANY."
    )
    annual_inc: float = Field(65_000.0, ge=0.0, description="Self-reported annual income (USD).")
    dti: float = Field(18.4, ge=0.0, le=1000.0, description="Debt-to-income ratio (%).")
    fico_range_low: float = Field(690.0, ge=300.0, le=850.0, description="FICO range low.")
    fico_range_high: float = Field(694.0, ge=300.0, le=850.0, description="FICO range high.")
    revol_util: float = Field(42.7, ge=0.0, le=1000.0, description="Revolving-line utilization (%).")
    open_acc: float = Field(11.0, ge=0.0, le=200.0, description="Number of open credit lines.")
    pub_rec: float = Field(0.0, ge=0.0, le=100.0, description="Number of derogatory public records.")
    purpose: str = Field(
        "debt_consolidation", description="Loan purpose, e.g. 'debt_consolidation'."
    )
    installment: float = Field(509.0, gt=0.0, description="Monthly installment (USD).")
    verification_status: str = Field(
        "Source Verified",
        description="Income verification: 'Verified'/'Source Verified'/'Not Verified'.",
    )
    addr_state: str = Field("CA", description="Two-letter US state code.")

    @field_validator("grade")
    @classmethod
    def _validate_grade(cls, v: str) -> str:
        grade = v.strip().upper()
        if grade not in VALID_GRADES:
            raise ValueError(f"grade must be one of {VALID_GRADES}, got {v!r}")
        return grade

    @field_validator("sub_grade")
    @classmethod
    def _validate_sub_grade(cls, v: str) -> str:
        sub = v.strip().upper()
        # Sub-grade is '<grade><1-5>' e.g. 'C2'. Validate shape leniently so new
        # LC sub-grades never hard-fail a legitimate application.
        if len(sub) == 2 and sub[0] in VALID_GRADES and sub[1] in "12345":
            return sub
        raise ValueError(f"sub_grade must look like 'C2' (grade A-G + digit 1-5), got {v!r}")

    @field_validator("term")
    @classmethod
    def _validate_term(cls, v: str) -> str:
        term = v.strip()
        if term not in VALID_TERMS:
            raise ValueError(f"term must be one of {VALID_TERMS}, got {v!r}")
        return term

    @field_validator("addr_state")
    @classmethod
    def _validate_state(cls, v: str) -> str:
        state = v.strip().upper()
        if len(state) != 2 or not state.isalpha():
            raise ValueError(f"addr_state must be a two-letter US state code, got {v!r}")
        return state

    @field_validator("home_ownership")
    @classmethod
    def _normalise_home_ownership(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("verification_status")
    @classmethod
    def _normalise_verification(cls, v: str) -> str:
        return v.strip()

    @field_validator("purpose")
    @classmethod
    def _normalise_purpose(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("fico_range_high")
    @classmethod
    def _high_ge_low(cls, v: float, info: Any) -> float:
        low = info.data.get("fico_range_low")
        if low is not None and v < low:
            raise ValueError("fico_range_high must be >= fico_range_low")
        return v


class ReasonCodeOut(BaseModel):
    feature: str = Field(..., description="Engineered feature name.")
    direction: Literal["increases", "decreases"] = Field(
        ..., description="Whether the feature pushes the predicted PD up or down."
    )
    contribution: float = Field(..., description="Signed log-odds contribution (coef * x).")


class LendingClubResponse(BaseModel):
    summary: dict[str, Any] = Field(
        ...,
        description=(
            "Scalar summary: calibrated_pd, risk_decile, model_auc, base_rate, "
            "predicted_label, threshold."
        ),
    )
    reason_codes: list[ReasonCodeOut] = Field(
        ..., description="Top container-safe (SHAP-free) reason codes for this application."
    )
    score_distribution_figure: dict[str, Any] = Field(
        ..., description="Plotly figure JSON: {data, layout} — predicted-PD histogram by outcome."
    )
    calibration_figure: dict[str, Any] = Field(
        ..., description="Plotly figure JSON: {data, layout} — reliability curve."
    )
    data_source: DataSource = Field(
        ..., description="'synthetic' (artifact was synthetic-trained) or 'cache'."
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/run", response_model=LendingClubResponse)
def run(
    req: LendingClubRequest,
    supabase: Any = Depends(get_supabase),
) -> LendingClubResponse:
    """Score one loan application into a calibrated PD + risk decile + reason codes.

    Pipeline:
      lazy-load bundle (booster + fitted pipeline + calibration) → drop leakage →
      transform → score → calibrate PD ∈ [0, 1] → risk decile → honest base-rate
      threshold label → container-safe reason codes; the two population figures are
      the cached reference distribution + reliability curve.
    """
    bundle, score_fig, calib_fig = _ensure_loaded()

    application = req.model_dump()
    try:
        scored = bundle.score_one(application, top_k=_TOP_K_REASON_CODES)
    except ValidationError as exc:
        logger.warning("lendingclub-default scoring rejected the application: %s", exc)
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Application could not be scored: {exc.__class__.__name__}: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("lendingclub-default scoring failed")
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Scoring failed: {exc.__class__.__name__}: {exc}",
        ) from exc

    summary: dict[str, Any] = {
        "calibrated_pd": _safe_float(scored.get("pd")),
        "risk_decile": int(scored["decile"]),
        "model_auc": _safe_float(scored.get("model_auc")),
        "base_rate": _safe_float(scored.get("base_rate")),
        "predicted_label": str(scored.get("predicted_label")),
        "threshold": _safe_float(scored.get("threshold")),
    }

    reason_codes: list[ReasonCodeOut] = []
    for code in scored.get("reason_codes", []):
        contribution = _safe_float(code.get("contribution"))
        if contribution is None:
            continue
        direction = code.get("direction")
        if direction not in ("increases", "decreases"):
            continue
        reason_codes.append(
            ReasonCodeOut(
                feature=str(code.get("feature", "")),
                direction=direction,
                contribution=contribution,
            )
        )

    # The shipped artifact is synthetic-trained; the bundle reports 'synthetic'.
    # If the bundle ever reports a different provenance, surface 'cache'.
    data_source: DataSource = "synthetic" if scored.get("data_source") == "synthetic" else "cache"

    response = LendingClubResponse(
        summary=summary,
        reason_codes=reason_codes,
        score_distribution_figure=score_fig,
        calibration_figure=calib_fig,
        data_source=data_source,
    )

    _maybe_log_run(supabase, req, response)
    return response


# ---------------------------------------------------------------------------
# Supabase helpers (best-effort)
# ---------------------------------------------------------------------------


def _maybe_log_run(supabase: Any, req: LendingClubRequest, resp: LendingClubResponse) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "lendingclub-default",
                "params": req.model_dump(mode="json"),
                "result": {
                    "data_source": resp.data_source,
                    "summary": resp.summary,
                },
                "status": "ok",
            }
        ).execute()
    except Exception:
        logger.exception("tool_runs insert failed (non-fatal)")


def _maybe_log_failure(supabase: Any, req: LendingClubRequest, exc: BaseException) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "lendingclub-default",
                "params": req.model_dump(mode="json"),
                "result": None,
                "status": "error",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        ).execute()
    except Exception:
        logger.exception("tool_runs failure insert failed (non-fatal)")
