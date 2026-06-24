"""FinBERT sentiment classifier — wraps the vendored ``finbert_sentiment`` library.

Endpoint:
  POST /tools/finbert-sentiment/run — classify 1..64 financial sentences into a
  3-way sentiment label (negative / neutral / positive) and return the per-
  sentence predictions alongside an HONEST evaluation summary: the served model's
  OFFLINE-measured macro-F1 (NOT accuracy alone — the PhraseBank neutral class is
  ~60%) plus the class-prior and lexicon baselines as the honest floor, the
  test-set confusion matrix, and the per-class F1 bar.

Mirrors the live ``lstm-forecast`` tool (the ONNX-serve analog): inference runs
through onnxruntime + tokenizers ONLY when the committed DistilBERT ONNX artifact
is present, else it transparently falls back to the torch-free lexicon baseline.
This router imports NO torch and NO transformers on any path.

HONESTY (baked into the library, never overridden here):
  * Sentiment is a TEXT LABEL, not a tradable signal — NO alpha is claimed.
  * The headline metric is macro-F1 (with a bootstrap CI), never accuracy alone.
  * ``eval_macro_f1`` / ``lexicon_macro_f1`` / ``class_prior_macro_f1`` /
    ``beats_lexicon`` are the OFFLINE-measured values loaded verbatim from the
    committed ``artifacts/metrics.json`` — they are NEVER recomputed on the
    request's (unlabeled) input texts.
  * Walk-forward / purge / Deflated-Sharpe DO NOT apply (there is no return
    series); using them would be a category error.

SERVE PATH — onnxruntime + tokenizers ONLY, NEVER torch/transformers. The shipped
DistilBERT fine-tune is a committed ONNX (int8) artifact + tokenizer.json inside
the vendored package (``finbert_sentiment/artifacts/``). The predictor is loaded
LAZILY at module level (``_MODEL = None``) on the first request; a missing /
unreadable artifact for the requested ``distilbert`` model degrades to the
lexicon, and a hard artifact failure surfaces as a 502.

Robustness:
  - Best-effort row in ``platform.tool_runs`` for run accounting; failures are
    logged but never propagate to the client.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from ..deps import get_supabase

# Importing the vendor package side-effects ``sys.path`` so that
# ``import finbert_sentiment`` resolves to the vendored copy. This MUST run before
# the ``from finbert_sentiment import ...`` below, so the import block is hand-
# ordered (matching the live hrp-portfolio / lstm-forecast convention).
from ..lib import finbert_sentiment as _vendor_marker  # noqa: F401

from finbert_sentiment import (  # import resolves via the sys.path shim above
    LABELS,
    ArtifactError,
    Predictor,
    SentimentResult,
    SentimentSummary,
    ValidationError,
    build_evaluation_figures,
    load_committed_metrics,
    load_predictor,
    run_sentiment,
)
from finbert_sentiment.service import _ci_tuple, _safe_float as _lib_safe_float

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools/finbert-sentiment", tags=["finbert-sentiment"])


# ---------------------------------------------------------------------------
# Lazy served model (serve path; loaded on first call, NEVER imports torch/TF)
# ---------------------------------------------------------------------------

#: The process-wide served :class:`~finbert_sentiment.Predictor`. Typed loosely so
#: importing this module never imports onnxruntime/tokenizers. Created lazily on
#: the first request via :func:`_get_model`. ``None`` until then.
_MODEL: Predictor | None = None
_MODEL_LOCK = threading.Lock()


def _get_model() -> Predictor:
    """Return the process-wide served predictor, loading it lazily once.

    onnxruntime + tokenizers are imported INSIDE the library's
    :func:`load_predictor` (and only when the DistilBERT ONNX artifact is
    present) — importing this router module never pulls in an inference engine,
    and torch/transformers are never imported on this path. When the committed
    ONNX artifact is present the transformer backend is served; otherwise the
    predictor transparently falls back to the torch-free lexicon baseline.

    The default ``"distilbert"`` preference is used for the warm load; a request
    that explicitly asks for ``"lexicon"`` is served by a per-request lexicon
    predictor (cheap, pure-Python) rather than this cached one.

    Raises
    ------
    ArtifactError
        If the resolved backend's artifacts fail to load; the endpoint maps this
        to a 502.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:  # pragma: no cover - double-checked locking
            return _MODEL
        _MODEL = load_predictor("distilbert")
        logger.info("finbert-sentiment served model initialized: %s", _MODEL.backend)
        return _MODEL


def _run_sentiment_with_predictor(
    texts: list[str], predictor: Predictor
) -> SentimentResult:
    """Assemble a :class:`SentimentResult` reusing an already-built predictor.

    Numerically identical to :func:`finbert_sentiment.run_sentiment`: it runs the
    same committed-metrics load, the same ``predictor.predict`` forward pass, the
    same per-class score mapping over the canonical ``LABELS`` order, and the same
    confusion / per-class-F1 figures built from the committed metrics. The ONLY
    difference is that the (expensive) ONNX-backed predictor is supplied by the
    caller — i.e. the process-wide cached :data:`_MODEL` — instead of being
    rebuilt (which would re-create the ~67MB onnxruntime session) on every
    request. Inference is deterministic, so reusing the same session yields
    byte-for-byte the same predictions as a freshly loaded one.
    """
    metrics = load_committed_metrics()
    predictions = predictor.predict(texts)

    labels = list(LABELS)
    prediction_records = [
        {
            "text": p.text,
            "label": p.label,
            "scores": {name: _lib_safe_float(p.scores.get(name, 0.0)) for name in labels},
        }
        for p in predictions
    ]

    confusion_fig, f1_fig = build_evaluation_figures(metrics)

    summary = SentimentSummary(
        served_model=predictor.backend,
        eval_macro_f1=_lib_safe_float(metrics.get("eval_macro_f1")),
        eval_macro_f1_ci=_ci_tuple(metrics.get("eval_macro_f1_ci")),
        eval_accuracy=_lib_safe_float(metrics.get("eval_accuracy")),
        lexicon_macro_f1=_lib_safe_float(metrics.get("lexicon_macro_f1")),
        class_prior_macro_f1=_lib_safe_float(metrics.get("class_prior_macro_f1")),
        beats_lexicon=metrics.get("beats_lexicon"),
        n_texts=len(prediction_records),
        data_source=str(metrics.get("data_source", "unknown")),
        transformer_measured=bool(
            metrics.get("transformer_published_macro_f1", {}).get(
                "measured_in_this_build", False
            )
        ),
        notes=str(
            metrics.get(
                "notes",
                "Sentiment is a text label, not a tradable signal — no alpha is claimed.",
            )
        ),
    )
    return SentimentResult(
        summary=summary,
        predictions=prediction_records,
        confusion_figure=confusion_fig,
        per_class_f1_figure=f1_fig,
    )


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

RequestedModel = Literal["distilbert", "lexicon"]

#: Mirror of finbert_sentiment._validation.{MAX_BATCH, MAX_TEXT_CHARS} so the
#: request validator and the library agree on one bound (a single string is the
#: source of truth in the library; these are restated for the 422 path).
_MAX_TEXTS = 64
_MAX_TEXT_CHARS = 4000


class SentimentRequest(BaseModel):
    """Wire contract for a finbert-sentiment classification run.

    ``texts`` is a list of 1..64 non-empty sentences (each capped in length). The
    ``model`` toggle requests the transformer; the library falls back to the
    lexicon when no ONNX artifact is present.
    """

    texts: list[str] = Field(
        ...,
        description="1..64 non-empty financial sentences to classify (one label each).",
    )
    model: RequestedModel = Field(
        "distilbert",
        description=(
            "Served model preference. 'distilbert' uses the committed ONNX fine-tune "
            "when present (else falls back to 'lexicon'); 'lexicon' always uses the "
            "torch-free Loughran-McDonald-style baseline."
        ),
    )
    seed: int = Field(
        0,
        ge=0,
        le=999_999,
        description="Provenance only; inference is deterministic (no per-request randomness).",
    )

    @field_validator("texts")
    @classmethod
    def _validate_texts(cls, v: list[str]) -> list[str]:
        if not isinstance(v, list):  # pragma: no cover - pydantic coerces first
            raise ValueError("texts must be a list of strings")
        if len(v) == 0:
            raise ValueError("texts must contain at least 1 sentence")
        if len(v) > _MAX_TEXTS:
            raise ValueError(f"texts has {len(v)} items but the per-batch cap is {_MAX_TEXTS}")
        cleaned: list[str] = []
        for i, item in enumerate(v):
            if not isinstance(item, str):
                raise ValueError(f"texts[{i}] must be a string")
            if not item.strip():
                raise ValueError(f"texts[{i}] must not be blank")
            if len(item) > _MAX_TEXT_CHARS:
                raise ValueError(
                    f"texts[{i}] has {len(item)} chars but the per-text cap is {_MAX_TEXT_CHARS}"
                )
            cleaned.append(item)
        return cleaned


class SentimentResponse(BaseModel):
    summary: dict[str, Any] = Field(
        ...,
        description=(
            "Honest summary: served_model ('distilbert-onnx'|'lexicon'), eval_macro_f1, "
            "eval_macro_f1_ci, eval_accuracy, lexicon_macro_f1, class_prior_macro_f1, "
            "beats_lexicon (bool|null), n_texts, data_source, transformer_measured, notes. "
            "The eval scalars are OFFLINE-measured (loaded from the committed metrics), "
            "never recomputed on the request's unlabeled input."
        ),
    )
    predictions: list[dict[str, Any]] = Field(
        ...,
        description=(
            "One {text, label, scores{negative, neutral, positive}} record per input sentence."
        ),
    )
    confusion_figure: dict[str, Any] = Field(
        ..., description="Plotly figure JSON: {data, layout} — the locked test-set confusion matrix."
    )
    per_class_f1_figure: dict[str, Any] = Field(
        ..., description="Plotly figure JSON: {data, layout} — the per-class F1 bar."
    )


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
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/run", response_model=SentimentResponse)
def run(
    req: SentimentRequest,
    supabase=Depends(get_supabase),
) -> SentimentResponse:
    """Execute the finbert-sentiment classification pipeline (single-shot, sync).

    Pipeline (all inside the vendored library; serve via onnxruntime + tokenizers
    when the ONNX artifact is present, else the torch-free lexicon — NO torch/TF):
      load committed metrics.json → serve per-sentence predictions via the lazy
      predictor → assemble the honest summary (served model + offline-measured
      macro-F1 / lexicon / class-prior baselines + beats_lexicon) → build the
      test-set confusion + per-class-F1 figures from the committed metrics. The
      eval scalars are NEVER recomputed on the request's unlabeled input.
    """
    # Eagerly warm the lazy served model so a missing/unreadable artifact surfaces
    # as a clean 502 before we run the (cheap) classification. A request that
    # explicitly asks for the lexicon does not need the cached distilbert model.
    #
    # The warmed predictor is captured and THREADED into the run below so the
    # ~67MB onnxruntime session is built once (process-wide cache) instead of
    # being rebuilt per request — the previous code warmed _MODEL but then
    # discarded it, because run_sentiment() builds its own predictor internally.
    cached_predictor: Predictor | None = None
    if req.model == "distilbert":
        try:
            cached_predictor = _get_model()
        except ArtifactError as exc:
            logger.exception("finbert-sentiment served model artifact load failed")
            _maybe_log_failure(supabase, req, exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"finbert-sentiment artifact load failed: {exc.__class__.__name__}: {exc}",
            ) from exc
        except Exception as exc:  # noqa: BLE001 — normalize any session init error to 502
            logger.exception("finbert-sentiment served model init failed")
            _maybe_log_failure(supabase, req, exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"finbert-sentiment model init failed: {exc.__class__.__name__}: {exc}",
            ) from exc

    try:
        if cached_predictor is not None:
            # distilbert path: reuse the cached ONNX-backed predictor. Numerically
            # identical to run_sentiment (same metrics, same forward pass, same
            # figures) but without rebuilding the onnxruntime session.
            result = _run_sentiment_with_predictor(req.texts, cached_predictor)
        else:
            # lexicon path: cheap, pure-Python, no ONNX session — build per request
            # exactly as before (the cached predictor is the distilbert one).
            result = run_sentiment(req.texts, model_pref=req.model, seed=int(req.seed))
    except ValidationError as exc:
        # Should be caught by the request validator, but the library is the
        # authoritative guard — surface a clean 422.
        logger.warning("finbert-sentiment run rejected: %s", exc)
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"finbert-sentiment run rejected: {exc.__class__.__name__}: {exc}",
        ) from exc
    except ArtifactError as exc:
        logger.exception("finbert-sentiment artifact failure during run")
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"finbert-sentiment artifact failure: {exc.__class__.__name__}: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("finbert-sentiment run failed")
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"finbert-sentiment run failed: {exc.__class__.__name__}: {exc}",
        ) from exc

    raw_summary = result.summary
    # Re-coerce every scalar through the router's _safe_float so no NaN/Inf leaks
    # across the JSON boundary (the library already coerces, but the contract
    # mandates _safe_float here too). beats_lexicon stays bool|null verbatim.
    ci = raw_summary.eval_macro_f1_ci
    summary: dict[str, Any] = {
        "served_model": str(raw_summary.served_model),
        "eval_macro_f1": _safe_float(raw_summary.eval_macro_f1),
        "eval_macro_f1_ci": (
            [_safe_float(ci[0]), _safe_float(ci[1])] if ci is not None else None
        ),
        "eval_accuracy": _safe_float(raw_summary.eval_accuracy),
        "lexicon_macro_f1": _safe_float(raw_summary.lexicon_macro_f1),
        "class_prior_macro_f1": _safe_float(raw_summary.class_prior_macro_f1),
        "beats_lexicon": raw_summary.beats_lexicon,
        "n_texts": int(raw_summary.n_texts),
        "data_source": str(raw_summary.data_source),
        "transformer_measured": bool(raw_summary.transformer_measured),
        "notes": str(raw_summary.notes),
    }

    # Predictions: re-coerce each score through _safe_float defensively.
    predictions: list[dict[str, Any]] = [
        {
            "text": str(p["text"]),
            "label": str(p["label"]),
            "scores": {k: _safe_float(v) for k, v in dict(p["scores"]).items()},
        }
        for p in result.predictions
    ]

    response = SentimentResponse(
        summary=summary,
        predictions=predictions,
        confusion_figure=result.confusion_figure,
        per_class_f1_figure=result.per_class_f1_figure,
    )

    _maybe_log_run(supabase, req, response)
    return response


# ---------------------------------------------------------------------------
# Supabase helpers (best-effort)
# ---------------------------------------------------------------------------


def _maybe_log_run(supabase, req: SentimentRequest, resp: SentimentResponse) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "finbert-sentiment",
                "params": {"model": req.model, "n_texts": len(req.texts), "seed": req.seed},
                "result": {"summary": resp.summary},
                "status": "ok",
            }
        ).execute()
    except Exception:
        logger.exception("tool_runs insert failed (non-fatal)")


def _maybe_log_failure(supabase, req: SentimentRequest, exc: BaseException) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "finbert-sentiment",
                "params": {"model": req.model, "n_texts": len(req.texts), "seed": req.seed},
                "result": None,
                "status": "error",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        ).execute()
    except Exception:
        logger.exception("tool_runs failure insert failed (non-fatal)")
