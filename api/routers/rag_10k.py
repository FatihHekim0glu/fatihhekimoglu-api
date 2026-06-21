"""LLM agent reading 10-Ks (RAG) — wraps the vendored ``rag10k`` library.

Endpoint:
  POST /tools/rag-10k/run — ask ONE question about a company's 10-K and get a
  GROUNDED, CITED answer. Retrieval runs over an ONNX-embedded read-only index of
  the filing's chunks (committed, torch-free), an answer is composed (generative
  iff an LLM key is present, else the KEY-FREE extractive fallback), every cited
  chunk is validated to EXIST and be retrievable (NO hallucinated citations), and
  the system ABSTAINS ("not found in the filing") when retrieval is below the
  similarity threshold rather than fabricating. Each response carries the PURE
  ``answer_is_grounded`` verdict and the FROZEN 150-question harness ``eval_summary``
  (retrieval recall@k, citation-soundness, abstention rate).

Mirrors the live ``edgar-nlp`` (EDGAR-ingest analog) + ``mvts-forecast`` (ONNX-serve
analog) tools: the heavy lifting lives in the vendored library; the router is a
thin, leakage-aware wrapper that ANSWERS ONE QUESTION per request and degrades
gracefully.

KEY-FREE BY DEFAULT (the headline honesty contract):
  * Retrieval + the eval harness are entirely key-free: the int8-quantized
    sentence-transformer is served via onnxruntime + tokenizers (NO torch) over the
    committed read-only index, so the retrieval/eval half is demoable with no key.
  * The LLM generation step is gated behind an ``ANTHROPIC_API_KEY`` /
    ``OPENAI_API_KEY`` env check (set as a Fly secret). ABSENT a key the endpoint
    serves the EXTRACTIVE cited answer and NEVER hard-fails — ``mode_used`` and
    ``llm_key_present`` report honestly which path ran. This router NEVER adds a key.

SERVE PATH — onnxruntime + tokenizers ONLY, NEVER torch. The committed serve
artifacts (``artifacts/embedder.int8.onnx`` + ``tokenizer.json`` + the read-only
``corpus_index.npz`` index + the frozen ``eval.json`` / ``eval_set.json``) ship
inside the vendored package. This router imports NO torch and NO provider SDK at
module load; the query embedding runs through onnxruntime, and the LLM client (when
a key is present) is constructed lazily inside the library.

DATA REALITY: the deployed default answers over the SMALL committed cached corpus
(``data_source_pref="cache"``), so the tool works with NO live EDGAR fetch. A
``data_source_pref="edgar"`` request attempts a lazy live fetch and transparently
falls back to the committed cache on ANY failure — the endpoint never hard-fails on
EDGAR being unreachable. The served ``data_source`` reports which path produced the
answered filing (``"cache"`` | ``"edgar"``).

Robustness:
  - The onnxruntime embedder session is warmed LAZILY on the first request; a
    missing / unreadable serve artifact surfaces as a clean 502 before retrieval.
  - A request :class:`ValidationError` maps to a 422; any other failure maps to a
    502.
  - Best-effort row in ``platform.tool_runs`` for run accounting; failures are
    logged but never propagate to the client.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from typing import Any, Literal

import plotly.io as pio
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from ..deps import get_supabase

# Importing the vendor package side-effects ``sys.path`` so that ``import rag10k``
# resolves to the vendored copy. This MUST run before the ``from rag10k import ...``
# below, so the import block is hand-ordered and isort is suppressed (matching the
# live mvts-forecast / edgar-nlp convention).
from ..lib import rag_10k as _vendor_marker  # noqa: F401

from rag10k import (  # import resolves via the sys.path shim above
    ArtifactError,
    Rag10kError,
    ValidationError,
    run_query,
)
from rag10k.embed.onnx_embedder import OnnxEmbedder, embedder_artifacts_present

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools/rag-10k", tags=["rag-10k"])


# ---------------------------------------------------------------------------
# Lazy onnxruntime embedder (serve path; loaded on first call, NEVER imports torch)
# ---------------------------------------------------------------------------

#: The process-wide :class:`~rag10k.embed.onnx_embedder.OnnxEmbedder` serving the
#: committed int8 graph + tokenizer. Typed loosely (``object``) so importing this
#: module never imports onnxruntime / tokenizers; it is created lazily on the first
#: request via :func:`_get_embedder`. The corpus index and the LLM client are loaded
#: lazily inside the library on the serve path.
_EMBEDDER: object | None = None
_EMBEDDER_LOCK = threading.Lock()


def _get_embedder() -> OnnxEmbedder:
    """Return the process-wide ONNX embedder, loading it lazily once (torch-free).

    onnxruntime + tokenizers are imported INSIDE the library's
    :meth:`OnnxEmbedder.load` — importing the router module never pulls in an
    inference engine, and torch is never imported on this path. A missing /
    unreadable serve artifact raises :class:`ArtifactError`, which the endpoint maps
    to a 502.
    """
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    with _EMBEDDER_LOCK:
        if _EMBEDDER is not None:  # pragma: no cover - double-checked locking
            return _EMBEDDER
        embedder = OnnxEmbedder().load()
        _EMBEDDER = embedder
        logger.info("rag-10k onnxruntime embedder initialized (torch-free serve path)")
        return embedder


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

Mode = Literal["auto", "extractive", "generative"]
DataSourcePref = Literal["cache", "edgar"]
DataSource = Literal["cache", "edgar"]

#: The answer modes the request may select from.
_VALID_MODES: tuple[str, ...] = ("auto", "extractive", "generative")
#: The data-source preferences the request may select from.
_VALID_DATA_SOURCES: tuple[str, ...] = ("cache", "edgar")
#: Hard cap on ``top_k`` so a request can never ask for an unbounded retrieval.
_MAX_TOP_K: int = 20


class Rag10kRequest(BaseModel):
    """Wire contract for a single grounded-RAG query over a company's 10-K.

    Provide a ``question`` (required) and a ``ticker`` selecting which committed
    filing's index to answer over (default ``AAPL``). ``top_k`` bounds the
    retrieval (capped at :data:`_MAX_TOP_K`). ``mode`` selects the answer path:
    ``"auto"`` is generative iff an LLM key is present else extractive,
    ``"extractive"`` is always the key-free cited answer, ``"generative"`` requests
    the LLM but degrades to extractive when no key is set (NEVER a hard failure).
    ``data_source_pref`` selects ``"cache"`` (the committed offline corpus, the
    default) or ``"edgar"`` (a lazy live fetch that falls back to the cache).
    """

    question: str = Field(
        ...,
        description="The question to answer about the company's 10-K (required, non-empty).",
    )
    ticker: str = Field(
        default="AAPL",
        description="Issuer ticker symbol selecting which committed filing's index to answer over.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=_MAX_TOP_K,
        description=f"Number of chunks to retrieve (1..{_MAX_TOP_K}; capped server-side).",
    )
    mode: Mode = Field(
        "auto",
        description=(
            "Answer path. 'auto' = generative iff an LLM key is present else extractive; "
            "'extractive' = always the key-free cited answer; 'generative' = LLM when a "
            "key is set, else degrades to extractive (never a hard failure)."
        ),
    )
    data_source_pref: DataSourcePref = Field(
        "cache",
        description=(
            "Filing source. 'cache' answers over the committed offline corpus (default); "
            "'edgar' attempts a lazy live fetch and falls back to the cache on any failure."
        ),
    )
    seed: int = Field(
        7,
        ge=0,
        le=999_999,
        description="Master seed (accepted for a stable contract; retrieval is deterministic).",
    )

    @field_validator("question")
    @classmethod
    def _normalise_question(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("question must be a non-empty string")
        return cleaned

    @field_validator("ticker")
    @classmethod
    def _normalise_ticker(cls, v: str) -> str:
        cleaned = v.strip().upper()
        if not cleaned:
            raise ValueError("ticker must be a non-empty string")
        return cleaned


class Rag10kResponse(BaseModel):
    summary: dict[str, Any] = Field(
        ...,
        description=(
            "Scalar/bool summary: answer (str), abstained (bool), answer_is_grounded "
            "(bool, the PURE verdict), n_citations (int), retrieval_top_score (float), "
            "mode_used ('extractive'|'generative'), llm_key_present (bool), data_source "
            "('cache'|'edgar'). answer_is_grounded is TRUE only when every emitted claim "
            "maps to a retrieved supporting chunk above the similarity threshold; "
            "otherwise the system ABSTAINS."
        ),
    )
    citations: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Validated citations (NO hallucinated citations): each "
            "{chunk_id, item, char_span, score, text_snippet}."
        ),
    )
    eval_summary: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Frozen 150-Q harness IR metrics: {recall_at_k, citation_soundness, "
            "abstention_rate, n_questions}. No accuracy or alpha claim — this is an "
            "information-retrieval tool."
        ),
    )
    retrieval_figure: dict[str, Any] = Field(
        default_factory=dict,
        description="Plotly figure JSON: the top-k retrieval-score bar with the threshold line.",
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


def _figure_json(fig: Any) -> dict[str, Any]:
    """Normalise a library figure into a plain ``{data, layout}`` JSON dict.

    The vendored figure builder already returns a JSON-serializable ``{data,
    layout}`` mapping, but we round-trip through ``pio.to_json`` defensively so any
    stray numpy scalar is coerced exactly like the mvts-forecast / hrp-portfolio
    tools do.
    """
    payload: dict[str, Any] = json.loads(pio.to_json(fig, validate=False))
    return payload


def _empty_figure(title: str) -> dict[str, Any]:
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.update_layout(title=title, height=360)
    payload: dict[str, Any] = json.loads(pio.to_json(fig, validate=False))
    return payload


def _normalise_summary(raw: dict[str, Any]) -> dict[str, Any]:
    """Coerce the library summary into a JSON-safe dict (scalars via :func:`_safe_float`).

    ``answer_is_grounded`` / ``abstained`` / ``llm_key_present`` are PURE library
    booleans; the router NEVER recomputes or overrides them. The cosine
    ``retrieval_top_score`` is passed through ``_safe_float`` so a NaN/inf can never
    reach the client.
    """
    return {
        "answer": str(raw.get("answer", "")),
        "abstained": bool(raw.get("abstained", False)),
        "answer_is_grounded": bool(raw.get("answer_is_grounded", False)),
        "n_citations": int(raw.get("n_citations", 0)),
        "retrieval_top_score": _safe_float(raw.get("retrieval_top_score")),
        "mode_used": str(raw.get("mode_used", "extractive")),
        "llm_key_present": bool(raw.get("llm_key_present", False)),
        "data_source": str(raw.get("data_source", "cache")),
    }


def _normalise_eval(raw: dict[str, Any]) -> dict[str, Any]:
    """Coerce the frozen-harness eval block to JSON-safe scalars (or ``{}``)."""
    if not raw:
        return {}
    return {
        "recall_at_k": _safe_float(raw.get("recall_at_k")),
        "citation_soundness": _safe_float(raw.get("citation_soundness")),
        "abstention_rate": _safe_float(raw.get("abstention_rate")),
        "n_questions": int(raw.get("n_questions", 0)),
    }


def _normalise_citation(raw: dict[str, Any]) -> dict[str, Any]:
    """Coerce one citation dict to the wire shape with a JSON-safe score."""
    char_span = raw.get("char_span")
    span: Any = list(char_span) if isinstance(char_span, (list, tuple)) else char_span
    return {
        "chunk_id": str(raw.get("chunk_id", "")),
        "item": str(raw.get("item", "")),
        "char_span": span,
        "score": _safe_float(raw.get("score")),
        "text_snippet": str(raw.get("text_snippet", "")),
    }


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/run", response_model=Rag10kResponse)
def run(
    req: Rag10kRequest,
    supabase=Depends(get_supabase),
) -> Rag10kResponse:
    """Execute the grounded-RAG query pipeline (single-shot, sync).

    Pipeline (all inside the vendored library; retrieval served torch-free via
    onnxruntime + tokenizers):
      load the committed index for ``ticker`` (or chunk+embed a fresh EDGAR fetch
      with a cache fallback) -> embed the query (ONNX, no torch) -> retrieve top-k
      -> apply the PURE abstention gate -> abstain OR compose (generative iff a key
      is present, else the key-free extractive answer) -> validate every citation
      (NO hallucinated citations) -> derive the PURE ``answer_is_grounded`` verdict
      -> attach the frozen-harness ``eval_summary`` + the retrieval-score figure.
      NEVER trains, NEVER requires a key, NEVER fetches without the cache fallback.
    """
    # Eagerly warm the lazy onnxruntime embedder so a missing/unreadable serve
    # artifact surfaces as a clean 502 before retrieval — but only when the
    # artifacts are expected on disk (a fresh checkout without them surfaces the
    # same 502 from the library on the run below).
    if embedder_artifacts_present():
        try:
            _get_embedder()
        except ArtifactError as exc:
            logger.exception("rag-10k ONNX embedder load failed")
            _maybe_log_failure(supabase, req, exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"rag-10k artifact load failed: {exc.__class__.__name__}: {exc}",
            ) from exc
        except Exception as exc:  # normalize any onnxruntime init error to 502
            logger.exception("rag-10k onnxruntime embedder init failed")
            _maybe_log_failure(supabase, req, exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"rag-10k embedder init failed: {exc.__class__.__name__}: {exc}",
            ) from exc

    # The LLM client is lazy + key-gated INSIDE the library; this router NEVER
    # injects a client and NEVER adds a key. Absent a key, ``run_query`` resolves to
    # the extractive cited answer and never hard-fails.
    try:
        result = run_query(
            req.question,
            ticker=req.ticker,
            top_k=int(req.top_k),
            mode=req.mode,
            data_source_pref=req.data_source_pref,
            seed=int(req.seed),
        )
    except ValidationError as exc:
        logger.warning("rag-10k run rejected: %s", exc)
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"rag-10k run rejected: {exc.__class__.__name__}: {exc}",
        ) from exc
    except ArtifactError as exc:
        logger.exception("rag-10k serve artifact failed")
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"rag-10k artifact load failed: {exc.__class__.__name__}: {exc}",
        ) from exc
    except Rag10kError as exc:
        logger.exception("rag-10k run failed")
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"rag-10k run failed: {exc.__class__.__name__}: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("rag-10k run failed (unexpected)")
        _maybe_log_failure(supabase, req, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"rag-10k run failed: {exc.__class__.__name__}: {exc}",
        ) from exc

    payload = result.to_dict()
    summary = _normalise_summary(payload.get("summary", {}))
    citations = [_normalise_citation(c) for c in payload.get("citations", [])]
    eval_summary = _normalise_eval(payload.get("eval_summary", {}))

    # --- Figure (best-effort; never fatal) ----------------------------------
    raw_figure = payload.get("retrieval_figure") or {}
    if raw_figure:
        try:
            figure_json = _figure_json(raw_figure)
        except Exception:
            logger.exception("rag-10k retrieval figure failed (non-fatal)")
            figure_json = _empty_figure("Retrieval scores (top-k)")
    else:
        figure_json = {}

    data_source = str(summary.get("data_source", "cache"))
    response = Rag10kResponse(
        summary=summary,
        citations=citations,
        eval_summary=eval_summary,
        retrieval_figure=figure_json,
        data_source=data_source,  # type: ignore[arg-type]
    )

    _maybe_log_run(supabase, req, response)
    return response


# ---------------------------------------------------------------------------
# Supabase helpers (best-effort)
# ---------------------------------------------------------------------------


def _maybe_log_run(supabase, req: Rag10kRequest, resp: Rag10kResponse) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "rag-10k",
                "params": req.model_dump(mode="json"),
                "result": {
                    "data_source": resp.data_source,
                    "summary": resp.summary,
                    "eval_summary": resp.eval_summary,
                },
                "status": "ok",
            }
        ).execute()
    except Exception:
        logger.exception("tool_runs insert failed (non-fatal)")


def _maybe_log_failure(supabase, req: Rag10kRequest, exc: BaseException) -> None:
    if supabase is None:
        return
    try:
        supabase.schema("platform").table("tool_runs").insert(
            {
                "tool_slug": "rag-10k",
                "params": req.model_dump(mode="json"),
                "result": None,
                "status": "error",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        ).execute()
    except Exception:
        logger.exception("tool_runs failure insert failed (non-fatal)")
