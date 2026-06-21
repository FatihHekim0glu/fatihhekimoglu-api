"""Serve entrypoint the backend calls (onnxruntime + tokenizers + numpy, NO torch).

The FastAPI router calls :func:`run_query` to answer one question about a
company's 10-K end-to-end:

    ingest/cache -> parse -> chunk -> embed(ONNX) -> retrieve -> abstain?
        -> (generate | extractive) -> validate citations -> verdict -> response

The committed read-only index is used by default (``data_source="cache"``); a
fresh EDGAR fetch (lazy) chunks + embeds on the fly when requested. Generation is
key-gated: ``mode="auto"`` uses the LLM iff a key is present, else the KEY-FREE
extractive answer. Citations are ALWAYS validated (no hallucinated citations) and
the response carries the PURE ``answer_is_grounded`` verdict + the frozen-harness
``eval_summary``. NEVER fetches without the cache fallback; NEVER requires a key.

Importing this module has no side effects (onnxruntime / tokenizers / the LLM
client / plotly are imported lazily inside the call).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from rag10k._constants import ABSTENTION_THRESHOLD, DEFAULT_TOP_K, ENTAILMENT_THRESHOLD, MAX_TOP_K

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from rag10k.embed.onnx_embedder import OnnxEmbedder
    from rag10k.eval.harness import EvalSummary
    from rag10k.retrieve.index import CorpusIndex
    from rag10k.retrieve.search import RetrievalResult

#: The answer modes the serve path supports.
_MODES: tuple[str, ...] = ("auto", "extractive", "generative")
#: The data-source preferences the serve path supports.
_DATA_SOURCES: tuple[str, ...] = ("cache", "edgar")

#: Per-process cache of the frozen-harness ``eval_summary`` keyed by artifact dir,
#: so the (deterministic) 150-Q harness runs at most once per server, not per query.
_EVAL_SUMMARY_CACHE: dict[str, dict[str, Any]] = {}


@dataclass(frozen=True, slots=True)
class QuerySummary:
    """Immutable, JSON-safe summary of a single grounded-RAG query.

    Mirrors the backend ``summary`` block.

    Attributes
    ----------
    answer:
        The answer text (or the abstention message).
    abstained:
        Whether the system abstained (retrieval below threshold).
    answer_is_grounded:
        The PURE verdict (every claim cites a retrieved, entailing chunk).
    n_citations:
        Number of (validated) citations.
    retrieval_top_score:
        Cosine score of the top retrieved chunk.
    mode_used:
        ``"extractive"`` or ``"generative"`` (what actually ran).
    llm_key_present:
        Whether an LLM key was detected (honest about generation availability).
    data_source:
        ``"cache"`` or ``"edgar"`` (provenance of the answered filing).
    """

    answer: str
    abstained: bool
    answer_is_grounded: bool
    n_citations: int
    retrieval_top_score: float
    mode_used: str
    llm_key_present: bool
    data_source: str

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this summary."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class QueryRun:
    """Immutable bundle returned to the backend: summary + citations + eval + figure.

    Attributes
    ----------
    summary:
        The :class:`QuerySummary`.
    citations:
        List of ``{chunk_id, item, char_span, score, text_snippet}`` dicts.
    eval_summary:
        The frozen-harness ``{recall_at_k, citation_soundness, abstention_rate,
        n_questions}`` block.
    retrieval_figure:
        A Plotly ``{data, layout}`` dict: the top-k retrieval-score bar.
    """

    summary: QuerySummary
    citations: list[dict[str, Any]] = field(default_factory=list)
    eval_summary: dict[str, Any] = field(default_factory=dict)
    retrieval_figure: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this run."""
        return {
            "summary": self.summary.to_dict(),
            "citations": list(self.citations),
            "eval_summary": dict(self.eval_summary),
            "retrieval_figure": self.retrieval_figure,
        }


def run_query(
    question: str,
    *,
    ticker: str = "AAPL",
    top_k: int = DEFAULT_TOP_K,
    mode: str = "auto",
    data_source_pref: str = "cache",
    seed: int = 7,
    threshold: float = ABSTENTION_THRESHOLD,
    llm_client: Callable[[str, str], str] | None = None,
    index: CorpusIndex | None = None,
    embedder: OnnxEmbedder | None = None,
    artifact_dir: str | Path | None = None,
    include_figure: bool = True,
    include_eval: bool = True,
) -> QueryRun:
    """Answer one 10-K question end-to-end and return a JSON-safe summary + figures.

    Loads the committed index for ``ticker`` (or chunks+embeds a fresh EDGAR fetch
    when ``data_source_pref="edgar"``, with a cache fallback), embeds the query
    (ONNX, torch-free), retrieves top-k, and applies the abstention gate. If
    abstaining, returns the abstention message with no citations. Otherwise answers
    GENERATIVELY (LLM, when ``mode`` resolves to generative AND a key/``llm_client``
    is available) or EXTRACTIVELY (key-free fallback), validates every citation (no
    hallucinated citations), derives the PURE ``answer_is_grounded`` verdict, reads
    the frozen ``eval_summary``, and builds the retrieval-score bar. NEVER trains,
    NEVER requires a key, NEVER fetches without the cache fallback.

    Parameters
    ----------
    question:
        The user question (required, non-empty).
    ticker:
        Which committed filing's index to answer over.
    top_k:
        Number of chunks to retrieve (capped at ``MAX_TOP_K``).
    mode:
        ``"auto"`` (generative iff key present else extractive), ``"extractive"``,
        or ``"generative"``.
    data_source_pref:
        ``"cache"`` (default) or ``"edgar"`` (lazy live fetch, cache fallback).
    seed:
        Reserved seed for any deterministic sampling (retrieval is already
        deterministic; accepted so the backend contract is stable).
    threshold:
        The cosine abstention floor.
    llm_client:
        Optional injected ``(system, user) -> str`` LLM callable (mocked in tests);
        when ``None`` a real client is built lazily only if a key is present.
    index:
        Optional pre-loaded :class:`~rag10k.retrieve.index.CorpusIndex` (tests
        inject a fixture index; the deployed path loads the committed artifact).
    embedder:
        Optional pre-loaded embedder (anything with ``embed_one``); the deployed
        path constructs the committed :class:`~rag10k.embed.onnx_embedder.OnnxEmbedder`.
    artifact_dir:
        Directory holding the committed embedder + index (defaults to the package's
        shipped ``artifacts/``).
    include_figure:
        Whether to build the (lazy plotly) retrieval-score figure.
    include_eval:
        Whether to attach the frozen-harness ``eval_summary`` (cached per process).

    Returns
    -------
    QueryRun
        The summary, citations, eval summary, and retrieval figure.

    Raises
    ------
    ValidationError
        If the request is invalid (empty question, unknown mode/data-source, bad
        ``top_k``).
    ArtifactError
        If the committed index / embedder artifacts cannot be loaded.
    """
    from rag10k.generate.answer import llm_key_present as _llm_key_present
    from rag10k.retrieve.search import search

    question, ticker, top_k, mode, data_source_pref = _validate_request(
        question, ticker, top_k, mode, data_source_pref
    )

    key_present = bool(_llm_key_present())

    # 1) Load the corpus + the (torch-free) ONNX embedder. ``data_source_pref="edgar"``
    #    attempts a lazy live fetch and degrades to the committed cache on any failure.
    corpus_index, used_embedder, data_source = _load_corpus(
        ticker=ticker,
        data_source_pref=data_source_pref,
        index=index,
        embedder=embedder,
        artifact_dir=artifact_dir,
    )

    # 2) Retrieve top-k + apply the PURE abstention gate.
    retrieval = search(question, corpus_index, used_embedder, top_k=top_k, threshold=threshold)

    eval_summary = (
        _eval_summary_dict(used_embedder, artifact_dir=artifact_dir) if include_eval else {}
    )
    figure = _retrieval_figure(retrieval, threshold=threshold) if include_figure else {}

    # 3) Abstention dominates: a below-threshold retrieval NEVER answers; it returns
    #    the canonical "not found in the filing" message with no citations.
    if retrieval.abstain:
        summary = QuerySummary(
            answer=_abstention_message(),
            abstained=True,
            answer_is_grounded=False,
            n_citations=0,
            retrieval_top_score=float(retrieval.top_score),
            mode_used="extractive",
            llm_key_present=key_present,
            data_source=data_source,
        )
        return QueryRun(
            summary=summary,
            citations=[],
            eval_summary=eval_summary,
            retrieval_figure=figure,
        )

    # 4) Restrict the answerable context to the chunks AT/ABOVE the similarity
    #    threshold — the genuinely-supported ones. The grounding contract requires
    #    every cited chunk to be above threshold, so the answer (and its citations)
    #    are drawn only from supported chunks; the figure below still shows the full
    #    top-k so the threshold line stays meaningful.
    supported = _supported_retrieval(retrieval, threshold=threshold)

    # 5) Compose the answer (generative iff key/client available, else extractive).
    answer, mode_used = _compose(
        question, supported, mode=mode, llm_client=llm_client, key_present=key_present
    )

    # 6) Validate every citation against the retrieved set + the index (no hallucinated
    #    citations). Reject (strict) so a fabricated citation can never reach the
    #    response; a valid generative/extractive answer cites only retrieved chunks.
    citation_report = _validate(answer, supported, corpus_index)

    # 7) Derive the PURE grounding verdict + the validated citation list.
    verdict, citations = _verdict_and_citations(
        answer, supported, citation_report, threshold=threshold
    )

    summary = QuerySummary(
        answer=answer.answer,
        abstained=bool(answer.abstained),
        answer_is_grounded=bool(verdict.answer_is_grounded),
        n_citations=int(citation_report.n_citations),
        retrieval_top_score=float(retrieval.top_score),
        mode_used=mode_used,
        llm_key_present=key_present,
        data_source=data_source,
    )
    return QueryRun(
        summary=summary,
        citations=citations,
        eval_summary=eval_summary,
        retrieval_figure=figure,
    )


def _validate_request(
    question: str, ticker: str, top_k: int, mode: str, data_source_pref: str
) -> tuple[str, str, int, str, str]:
    """Validate + normalize the request; return the cleaned ``(question, ...)`` tuple.

    Raises
    ------
    ValidationError
        On an empty question, unknown mode/data-source, or ``top_k <= 0``.
    """
    from rag10k._exceptions import ValidationError

    if not isinstance(question, str) or not question.strip():
        raise ValidationError("run_query: question must be a non-empty string.")
    if mode not in _MODES:
        raise ValidationError(f"run_query: mode must be one of {_MODES}, got {mode!r}.")
    if data_source_pref not in _DATA_SOURCES:
        raise ValidationError(
            f"run_query: data_source_pref must be one of {_DATA_SOURCES}, got {data_source_pref!r}."
        )
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValidationError(f"run_query: top_k must be a positive int, got {top_k!r}.")

    clean_ticker = ticker.strip().upper() if isinstance(ticker, str) and ticker.strip() else "AAPL"
    capped_top_k = min(top_k, MAX_TOP_K)
    return question.strip(), clean_ticker, capped_top_k, mode, data_source_pref


def _load_corpus(
    *,
    ticker: str,
    data_source_pref: str,
    index: CorpusIndex | None,
    embedder: OnnxEmbedder | None,
    artifact_dir: str | Path | None,
) -> tuple[CorpusIndex, Any, str]:
    """Resolve the corpus index + embedder + the answered ``data_source``.

    Injected ``index``/``embedder`` (tests) short-circuit artifact loading. The
    ``data_source_pref="edgar"`` path attempts a lazy live fetch, builds an
    in-memory index from the freshly chunked filing, and degrades to the committed
    cache on ANY failure (never hard-fails the endpoint).
    """
    from rag10k.embed.onnx_embedder import OnnxEmbedder

    used_embedder: Any = embedder if embedder is not None else OnnxEmbedder(artifact_dir).load()

    if index is not None:
        # A caller-supplied index is used verbatim; its provenance is "cache" unless
        # an EDGAR fetch was explicitly requested (handled below only for the
        # artifact-loading path).
        return index, used_embedder, "cache"

    if data_source_pref == "edgar":
        fresh = _try_fetch_and_index(ticker, used_embedder)
        if fresh is not None:
            return fresh, used_embedder, "edgar"
        # Fall through to the committed cache on any live-fetch/index failure.

    corpus_index = _load_committed_index(ticker, artifact_dir=artifact_dir)
    return corpus_index, used_embedder, "cache"


def _load_committed_index(ticker: str, *, artifact_dir: str | Path | None) -> CorpusIndex:
    """Load the committed index for ``ticker``, SCOPED to that issuer's filing.

    Retrieval is per-company (no cross-filing leakage): a per-ticker
    ``corpus_index_<TICKER>.npz`` is preferred when present (already single-issuer);
    otherwise the default ``corpus_index.npz`` bundle (which hosts several filings)
    is loaded and FILTERED to the requested ticker's CIK via
    :meth:`~rag10k.retrieve.index.CorpusIndex.for_cik`, so the mixed bundle never
    answers a ticker A query with a ticker B chunk. If the ticker cannot be resolved
    to a CIK (an unknown issuer), the full bundle is returned unscoped so the
    endpoint still answers rather than hard-failing. Raises
    :class:`~rag10k._exceptions.ArtifactError` only when no index artifact exists.
    """
    from rag10k.ingest.cached_corpus import resolve_ticker_to_cik
    from rag10k.retrieve.index import CorpusIndex, index_present

    if index_present(ticker, artifact_dir=artifact_dir):
        return CorpusIndex.load(ticker, artifact_dir=artifact_dir)

    bundle = CorpusIndex.load(None, artifact_dir=artifact_dir)
    cik = resolve_ticker_to_cik(ticker)
    if cik is not None and cik in bundle.ciks:
        # Scope the mixed bundle to the requested issuer's chunks (no leakage).
        return bundle.for_cik(cik)
    return bundle


def _try_fetch_and_index(ticker: str, embedder: Any) -> CorpusIndex | None:
    """Lazily fetch a fresh 10-K, chunk + embed it, and return an in-memory index.

    Returns ``None`` on ANY failure so the caller degrades to the committed cache.
    The fetched filing is keyed by its EDGAR acceptance datetime (no look-ahead);
    chunking + embedding reuse the deterministic serve path.
    """
    from rag10k.chunk.splitter import chunk_filing
    from rag10k.ingest.client import EdgarClient
    from rag10k.parse.sections import extract_sections
    from rag10k.retrieve.index import CorpusIndex

    try:
        filing = EdgarClient().fetch_latest_10k(ticker=ticker)
        parsed = extract_sections(filing.raw_text)
        chunks = chunk_filing(filing, parsed)
        vectors = embedder.embed([chunk.text for chunk in chunks])
        return CorpusIndex.from_chunks(chunks, vectors)
    except Exception:  # degrade to the committed cache on ANY live-fetch/index failure
        return None


def _supported_retrieval(retrieval: RetrievalResult, *, threshold: float) -> RetrievalResult:
    """Return ``retrieval`` restricted to chunks AT/ABOVE the abstention threshold.

    The genuinely-supported chunks are the only legitimate citation source under the
    grounding contract (every cited chunk must clear the floor). The caller only
    reaches this on a NON-abstaining retrieval, so at least one chunk is retained.
    Ranks are re-based to stay contiguous from 0.
    """
    from rag10k.retrieve.search import RetrievalResult, RetrievedChunk

    kept = [
        RetrievedChunk(chunk=hit.chunk, score=hit.score, rank=rank)
        for rank, hit in enumerate(h for h in retrieval.retrieved if float(h.score) >= threshold)
    ]
    return RetrievalResult(
        query=retrieval.query,
        retrieved=kept,
        top_score=retrieval.top_score,
        abstain=retrieval.abstain,
        threshold=retrieval.threshold,
        extra=dict(retrieval.extra),
    )


def _compose(
    question: str,
    retrieval: RetrievalResult,
    *,
    mode: str,
    llm_client: Callable[[str, str], str] | None,
    key_present: bool,
) -> tuple[Any, str]:
    """Compose the answer via the generate dispatcher; return ``(answer, mode_used)``.

    ``mode_used`` is reported HONESTLY: ``"generative"`` only when the LLM path
    actually ran (the model id is not the extractive sentinel), else ``"extractive"``.
    """
    from rag10k.generate.answer import compose_answer
    from rag10k.generate.extractive import EXTRACTIVE_MODEL

    answer = compose_answer(question, retrieval, mode=mode, client=llm_client)
    mode_used = "extractive" if answer.model == EXTRACTIVE_MODEL else "generative"
    return answer, mode_used


def _validate(answer: Any, retrieval: RetrievalResult, corpus_index: CorpusIndex) -> Any:
    """Validate the answer's citations against the retrieved set + the index (strict).

    Raises
    ------
    CitationError
        If the answer cites a fabricated / non-retrieved chunk id.
    """
    from rag10k.eval.citations import validate_citations

    return validate_citations(
        answer.cited_chunk_ids, retrieval.retrieved, corpus_index, strict=True
    )


def _verdict_and_citations(
    answer: Any,
    retrieval: RetrievalResult,
    citation_report: Any,
    *,
    threshold: float,
) -> tuple[Any, list[dict[str, Any]]]:
    """Derive the PURE grounding verdict + the validated citation dicts.

    Each cited chunk is treated as one claim unit: it is "entailed" iff it was
    retrieved at or above the abstention threshold AND its content appears in the
    answer under the deterministic entailment proxy (the fraction of the cited
    chunk's content tokens present in the answer clears the floor). For an
    EXTRACTIVE answer the chunk text is reproduced verbatim, so this is ~1.0 by
    construction; for a GENERATIVE answer it requires the cited chunk's content to
    actually surface in the answer. ``answer_is_grounded`` is then the pure
    conjunction (did-not-abstain AND citations-valid AND every claim entailed).
    """
    from rag10k.eval.harness import entailment_proxy
    from rag10k.eval.verdict import answer_is_grounded

    by_id = {hit.chunk.chunk_id: hit for hit in retrieval.retrieved}
    claim_entailments: list[bool] = []
    citations: list[dict[str, Any]] = []
    for chunk_id in answer.cited_chunk_ids:
        hit = by_id.get(chunk_id)
        if hit is None:  # pragma: no cover - validate_citations already rejected this
            claim_entailments.append(False)
            continue
        above_threshold = float(hit.score) >= threshold
        # Proxy direction: fraction of the CITED CHUNK's content tokens that appear
        # in the answer — so a chunk whose content the answer actually reflects is
        # "entailed" (verbatim for extractive; surfaced for generative).
        entailed = above_threshold and entailment_proxy(
            hit.chunk.text, answer.answer, threshold=ENTAILMENT_THRESHOLD
        )
        claim_entailments.append(bool(entailed))
        citations.append(hit.to_citation())

    verdict = answer_is_grounded(
        abstained=bool(answer.abstained),
        citation_report=citation_report,
        claim_entailments=claim_entailments,
    )
    return verdict, citations


def _retrieval_figure(retrieval: RetrievalResult, *, threshold: float) -> dict[str, Any]:
    """Build the top-k retrieval-score bar (lazy plotly); ``{}`` if nothing retrieved."""
    if not retrieval.retrieved:  # pragma: no cover - only reached on a non-abstaining path
        return {}
    from rag10k.plots import retrieval_score_figure

    chunk_ids = [hit.chunk.chunk_id for hit in retrieval.retrieved]
    scores = [float(hit.score) for hit in retrieval.retrieved]
    return retrieval_score_figure(chunk_ids, scores, threshold=threshold)


def _eval_summary_dict(
    embedder: Any,
    *,
    artifact_dir: str | Path | None,
) -> dict[str, Any]:
    """Return the frozen-harness ``eval_summary`` block (cached per process).

    The eval summary is a property of the WHOLE committed 150-Q harness over the
    FULL committed corpus — it is independent of which ticker the request asked
    about. It is read from the committed ``eval.json`` (the canonical frozen number,
    built offline over the full corpus); if that artifact is unavailable, it falls
    back to running the deterministic harness over the FULL combined index (NOT the
    per-ticker-scoped one, since the harness spans all three issuers and scopes each
    question internally). Memoized per artifact dir so it is computed at most once.
    Degrades to an empty block if neither source is available.
    """
    cache_key = str(artifact_dir) if artifact_dir is not None else "<default>"
    cached = _EVAL_SUMMARY_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    committed = _read_committed_eval_summary(artifact_dir=artifact_dir)
    if committed is not None:
        _EVAL_SUMMARY_CACHE[cache_key] = committed
        return dict(committed)

    block = _harness_eval_summary(embedder, artifact_dir=artifact_dir)
    if block:
        _EVAL_SUMMARY_CACHE[cache_key] = block
    return dict(block)


def _read_committed_eval_summary(*, artifact_dir: str | Path | None) -> dict[str, Any] | None:
    """Read the committed ``eval.json`` honest IR summary block, or ``None`` if absent.

    The deployed response quotes the committed frozen-harness summary built offline
    over the full corpus; reading it avoids re-running 150 retrievals per server and
    keeps the served numbers byte-identical to the committed ones.
    """
    import json
    from pathlib import Path as _Path

    from rag10k.embed.build_index import EVAL_SUMMARY_NAME
    from rag10k.embed.onnx_embedder import default_artifact_dir

    directory = _Path(artifact_dir) if artifact_dir is not None else default_artifact_dir()
    path = directory / EVAL_SUMMARY_NAME
    if not path.is_file():
        return None
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))["summary"]
    except (OSError, ValueError, KeyError, TypeError):  # pragma: no cover - defensive
        return None
    return {
        "recall_at_k": float(summary["recall_at_k"]),
        "citation_soundness": float(summary["citation_soundness"]),
        "abstention_rate": float(summary["abstention_rate"]),
        "n_questions": int(summary["n_questions"]),
    }


def _harness_eval_summary(embedder: Any, *, artifact_dir: str | Path | None) -> dict[str, Any]:
    """Run the frozen harness over the FULL committed bundle (fallback for ``eval.json``).

    Loads the FULL combined corpus index (the harness scopes each question to its
    own issuer internally) and runs the deterministic 150-Q harness. Returns an empty
    block if the committed eval set / index cannot be loaded.
    """
    from rag10k._exceptions import Rag10kError
    from rag10k.eval.harness import run_harness
    from rag10k.retrieve.index import CorpusIndex

    try:
        full_index = CorpusIndex.load(None, artifact_dir=artifact_dir)
        summary: EvalSummary = run_harness(full_index, embedder)
    except Rag10kError:  # pragma: no cover - the committed eval set ships with the package
        return {}

    return {
        "recall_at_k": float(summary.recall_at_k),
        "citation_soundness": float(summary.citation_soundness),
        "abstention_rate": float(summary.abstention_rate),
        "n_questions": int(summary.n_questions),
    }


def _abstention_message() -> str:
    """Return the canonical abstention message ("not found in the filing")."""
    from rag10k._constants import ABSTENTION_MESSAGE

    return ABSTENTION_MESSAGE
