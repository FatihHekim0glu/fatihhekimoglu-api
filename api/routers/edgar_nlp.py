"""10-K filing NLP tool — wraps the vendored ``edgar_nlp`` library.

Endpoint:
  POST /tools/edgar-nlp/run — pull ONE company's 10-K from SEC EDGAR (keyed on the
  EDGAR ACCEPTANCE DATETIME — the public-dissemination clock, never the period-of-
  report), extract the Risk-Factors (Item 1A) and MD&A (Item 7) sections, score
  Loughran-McDonald tone + hand-rolled readability (Gunning-Fog / Flesch-Kincaid /
  SMOG), and attach the HONEST precomputed point-in-time tone-spread panel verdict.

Mirrors the live ``hrp-portfolio`` / ``finbert-sentiment`` tools: the heavy lifting
lives in the vendored library, the router is a thin, leakage-aware wrapper that
SCORES ONE FILING per request and degrades gracefully.

HONESTY (baked into the library, never overridden here):
  * Negative tone / low readability tilt intuitively, BUT a point-in-time, purged
    walk-forward tone-tertile spread is NOT statistically distinguishable from zero
    after a Deflated-Sharpe correction (DSR ~0, p > 0.3). ``tone_spread_has_edge``
    is a PURE function of OOS spread Sharpe + DSR + HAC significance — it reads
    ``False`` on the seeded synthetic panel (the deployed default). NO alpha is
    claimed; this is a DESCRIPTIVE scorer (Loughran-McDonald 2011).
  * The panel statistics are a fixed, reproducible property of the seeded synthetic
    panel — NEVER a per-request backtest.

DATA PATH — lazy httpx + a <=10 req/s token bucket, NEVER torch/spaCy/transformers.
  * ``data_source_pref="synthetic"`` short-circuits straight to the shipped cached/
    synthetic 10-K fixture (no network).
  * ``"auto"`` / ``"edgar"`` attempt a live EDGAR fetch (lazy httpx, descriptive
    User-Agent, fair-access token bucket) and transparently fall back to the cached/
    synthetic fixture on ANY failure — the endpoint never hard-fails on EDGAR being
    unreachable. The served ``data_source`` reports exactly which path produced the
    scored text (``"edgar"`` | ``"cache"`` | ``"synthetic"``).

Robustness:
  - Best-effort row in ``platform.tool_runs`` for run accounting; failures are
    logged but never propagate to the client.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from ..deps import get_supabase

# Importing the vendor package side-effects ``sys.path`` so that
# ``import edgar_nlp`` resolves to the vendored copy. This MUST run before the
# ``from edgar_nlp import ...`` below, so the import block is hand-ordered
# (matching the live hrp-portfolio / finbert-sentiment convention).
from ..lib import edgar_nlp as _vendor_marker  # noqa: F401

from edgar_nlp import (  # import resolves via the sys.path shim above
    EdgarNlpError,
    Filing,
    ParseError,
    ValidationError,
    filing_figures,
    score_filing,
)
from edgar_nlp._constants import DEFAULT_SECTIONS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools/edgar-nlp", tags=["edgar-nlp"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

DataSourcePref = Literal["auto", "edgar", "synthetic"]
DataSource = Literal["edgar", "cache", "synthetic"]
Section = Literal["risk_factors", "mda"]

#: Anchored Item sections the scorer understands (subset toggle on the request).
_VALID_SECTIONS: tuple[str, ...] = tuple(DEFAULT_SECTIONS)


class EdgarNlpRequest(BaseModel):
    """Wire contract for a single-filing 10-K NLP scoring run.

    Provide a ``ticker`` (default ``AAPL``) OR a ``cik``; optionally pin the
    calendar ``year`` of the EDGAR acceptance datetime (default: latest available /
    cached). ``sections`` is the subset of anchored Item sections to score; both
    are scored by default. ``data_source_pref`` selects the fetch path:
    ``"synthetic"`` short-circuits to the cached fixture (no network), while
    ``"auto"``/``"edgar"`` attempt EDGAR and fall back to the fixture on failure.
    """

    ticker: str = Field(
        default="AAPL",
        description="Issuer ticker symbol (used when no CIK is given).",
    )
    cik: str | None = Field(
        default=None,
        description="Issuer SEC Central Index Key (overrides ticker when present).",
    )
    year: int | None = Field(
        default=None,
        ge=1994,
        le=2100,
        description=(
            "Calendar year of the EDGAR acceptance datetime to fetch. None = the "
            "latest available (or the cached fixture)."
        ),
    )
    sections: list[Section] = Field(
        default=["risk_factors", "mda"],
        description="Subset of ['risk_factors','mda'] to score (default: both).",
    )
    data_source_pref: DataSourcePref = Field(
        "auto",
        description=(
            "Fetch path. 'auto'/'edgar' try a live EDGAR fetch then fall back to the "
            "cached/synthetic fixture; 'synthetic' uses the fixture directly (offline)."
        ),
    )

    @field_validator("ticker")
    @classmethod
    def _normalise_ticker(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("cik")
    @classmethod
    def _normalise_cik(cls, v: str | None) -> str | None:
        if v is None:
            return None
        digits = "".join(ch for ch in str(v) if ch.isdigit())
        # Empty/blank CIK is treated as "unset" so the ticker path is used.
        return digits or None

    @field_validator("sections")
    @classmethod
    def _normalise_sections(cls, v: list[str]) -> list[str]:
        seen: dict[str, None] = {}
        for raw in v:
            name = str(raw).strip().lower()
            if name in _VALID_SECTIONS:
                seen.setdefault(name, None)
        if not seen:
            # Never score zero sections — default to the full set.
            for name in _VALID_SECTIONS:
                seen.setdefault(name, None)
        # Canonical order so the combined-document scalars are deterministic.
        return [name for name in _VALID_SECTIONS if name in seen]


class EdgarNlpResponse(BaseModel):
    summary: dict[str, Any] = Field(
        ...,
        description=(
            "Scalar summary: tone, neg_pct, pos_pct, uncertainty_pct, litigious_pct, "
            "fog, flesch_kincaid, smog, word_count, n_sentences, "
            "panel_tone_spread_sharpe, panel_deflated_sharpe, panel_hac_pvalue, "
            "tone_spread_has_edge (bool), data_source."
        ),
    )
    tone_by_section_figure: dict[str, Any] = Field(
        ..., description="Plotly figure JSON: {data, layout} — LM tone by section."
    )
    readability_figure: dict[str, Any] = Field(
        ..., description="Plotly figure JSON: {data, layout} — readability grades."
    )
    data_source: DataSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value: float | None) -> float | None:
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
# Filing resolution — EDGAR (lazy) → cache/synthetic fallback (never hard-fail)
# ---------------------------------------------------------------------------


def _load_fixture(req: EdgarNlpRequest, *, source: str) -> Filing:
    """Return the shipped cached/synthetic 10-K fixture as a :class:`Filing`."""
    from edgar_nlp.ingest.fixtures import load_fixture_filing

    return load_fixture_filing(ticker=req.ticker or "AAPL", source=source)


def _fetch_from_edgar(req: EdgarNlpRequest) -> Filing:
    """Attempt a live EDGAR fetch for the requested issuer/year.

    Constructs the lazy :class:`~edgar_nlp.ingest.client.EdgarClient` (httpx +
    token bucket are built on first use, never at import). Raises
    :class:`~edgar_nlp.EdgarNlpError` on any failure so the caller can degrade to
    the cached/synthetic fixture with a single ``except`` clause.
    """
    from edgar_nlp.ingest.client import EdgarClient

    client = EdgarClient()
    ticker = req.cik is None and req.ticker or None
    cik = req.cik

    if req.year is None:
        return client.fetch_latest_10k(ticker=ticker, cik=cik)
    return client.fetch_10k_for_year(year=int(req.year), ticker=ticker, cik=cik)


def _resolve_filing(req: EdgarNlpRequest) -> Filing:
    """Resolve the request to a scored-text :class:`Filing`, never hard-failing.

    * ``data_source_pref="synthetic"`` → the cached/synthetic fixture directly.
    * ``"auto"`` / ``"edgar"`` → a live EDGAR fetch, transparently falling back to
      the cached fixture (``data_source="cache"``) on ANY EDGAR failure, so the
      endpoint degrades gracefully when EDGAR is unreachable.
    """
    if req.data_source_pref == "synthetic":
        return _load_fixture(req, source="synthetic")

    try:
        return _fetch_from_edgar(req)
    except EdgarNlpError as exc:
        logger.warning(
            "edgar-nlp EDGAR fetch failed (%s); falling back to cached fixture",
            exc.__class__.__name__,
        )
    except Exception as exc:  # any transport error degrades to the fixture, never 500s
        logger.warning(
            "edgar-nlp EDGAR fetch raised %s; falling back to cached fixture",
            exc.__class__.__name__,
        )
    return _load_fixture(req, source="cache")


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/run", response_model=EdgarNlpResponse)
def run(
    req: EdgarNlpRequest,
    supabase=Depends(get_supabase),
) -> EdgarNlpResponse:
    """Execute the edgar-nlp single-filing scoring pipeline (single-shot, sync).

    Pipeline (all inside the vendored library; lazy httpx + token bucket, NO
    torch/spaCy/transformers):
      resolve filing (EDGAR → cache → synthetic) → extract requested Item sections
      → score Loughran-McDonald tone + hand-rolled readability over the combined
      document → attach the precomputed honest panel verdict → build the
      tone-by-section + readability Plotly figures. SCORES ONE FILING per request.
    """
    try:
        filing = _resolve_filing(req)
    except Exception as exc:
        logger.exception("edgar-nlp filing resolution failed")
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"edgar-nlp filing resolution failed: {exc.__class__.__name__}: {exc}",
        ) from exc

    try:
        score = score_filing(filing, sections=tuple(req.sections))
    except (ValidationError, ParseError) as exc:
        # The requested sections could not be located / validated — a clean 422.
        logger.warning("edgar-nlp run rejected: %s", exc)
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"edgar-nlp run rejected: {exc.__class__.__name__}: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("edgar-nlp run failed")
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"edgar-nlp run failed: {exc.__class__.__name__}: {exc}",
        ) from exc

    raw = score.summary
    data_source = str(raw["data_source"])

    # Re-coerce every scalar through the router's _safe_float so no NaN/Inf leaks
    # across the JSON boundary. tone_spread_has_edge stays a verbatim bool.
    summary: dict[str, Any] = {
        "tone": _safe_float(raw["tone"]),
        "neg_pct": _safe_float(raw["neg_pct"]),
        "pos_pct": _safe_float(raw["pos_pct"]),
        "uncertainty_pct": _safe_float(raw["uncertainty_pct"]),
        "litigious_pct": _safe_float(raw["litigious_pct"]),
        "fog": _safe_float(raw["fog"]),
        "flesch_kincaid": _safe_float(raw["flesch_kincaid"]),
        "smog": _safe_float(raw["smog"]),
        "word_count": int(raw["word_count"]),
        "n_sentences": int(raw["n_sentences"]),
        "panel_tone_spread_sharpe": _safe_float(raw["panel_tone_spread_sharpe"]),
        "panel_deflated_sharpe": _safe_float(raw["panel_deflated_sharpe"]),
        "panel_hac_pvalue": _safe_float(raw["panel_hac_pvalue"]),
        "tone_spread_has_edge": bool(raw["tone_spread_has_edge"]),
        "data_source": data_source,
    }

    try:
        figures = filing_figures(score)
    except Exception as exc:
        logger.exception("edgar-nlp figure build failed")
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"edgar-nlp figure build failed: {exc.__class__.__name__}: {exc}",
        ) from exc

    response = EdgarNlpResponse(
        summary=summary,
        tone_by_section_figure=figures["tone_by_section_figure"],
        readability_figure=figures["readability_figure"],
        data_source=data_source,  # type: ignore[arg-type]
    )

    _maybe_log_run(supabase, req, response)
    return response


# ---------------------------------------------------------------------------
# Supabase helpers (best-effort)
# ---------------------------------------------------------------------------


def _run_params(req: EdgarNlpRequest) -> dict[str, Any]:
    return {
        "ticker": req.ticker,
        "cik": req.cik,
        "year": req.year,
        "sections": list(req.sections),
        "data_source_pref": req.data_source_pref,
    }


def _maybe_log_run(supabase, req: EdgarNlpRequest, resp: EdgarNlpResponse) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "edgar-nlp",
                "params": _run_params(req),
                "result": {
                    "data_source": resp.data_source,
                    "summary": resp.summary,
                },
                "status": "ok",
            }
        ).execute()
    except Exception:
        logger.exception("tool_runs insert failed (non-fatal)")


def _maybe_log_failure(supabase, req: EdgarNlpRequest, exc: BaseException) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "edgar-nlp",
                "params": _run_params(req),
                "result": None,
                "status": "error",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        ).execute()
    except Exception:
        logger.exception("tool_runs failure insert failed (non-fatal)")
