"""rag10k — an LLM agent that reads 10-Ks: grounded, cited, abstaining RAG.

Ask a question about a company's 10-K and get a GROUNDED, CITED answer: retrieval
over an ONNX-embedded index of the filing's chunks, an LLM that must cite its
sources and ABSTAIN ("not found in the filing") when the answer isn't in the
document, and a FROZEN 150-question eval harness measuring retrieval recall@k,
citation-soundness, and abstention rate. There are NO hallucinated citations
(every cited chunk exists and is retrievable) and NO alpha/DSR claim — this is an
information-retrieval tool, not a trading signal.

Retrieval is KEY-FREE: a small sentence-transformer exported to int8 ONNX is
served via onnxruntime + tokenizers (NO torch) over a prebuilt, committed,
read-only index, so the retrieval/eval half is fully demoable without a key. Only
the GENERATIVE answer needs an ``ANTHROPIC_API_KEY``/``OPENAI_API_KEY``; absent a
key the deployed default returns a KEY-FREE EXTRACTIVE cited answer.

IMPORT PURITY (enforced): the package has ZERO import-time side effects. No
network, no EDGAR fetch, no onnxruntime / tokenizers / torch / anthropic / openai
/ httpx import happens at package load — every heavy client is constructed lazily
on first use. ``python -c "import rag10k"`` is safe and fast.

Public API is curated below; see :data:`__all__`.
"""

from __future__ import annotations

from rag10k._constants import (
    ABSTENTION_MESSAGE,
    ABSTENTION_THRESHOLD,
    DEFAULT_SECTIONS,
    DEFAULT_TOP_K,
    DEFAULT_USER_AGENT,
    EDGAR_MAX_REQUESTS_PER_SECOND,
    EMBEDDING_DIM,
    EPS,
    MAX_TOP_K,
    SECTION_LABELS,
)
from rag10k._exceptions import (
    ArtifactError,
    CitationError,
    GenerationError,
    IngestError,
    InsufficientDataError,
    ParseError,
    Rag10kError,
    RetrievalError,
    ValidationError,
)
from rag10k._manifest import RunManifest, config_hash
from rag10k._rng import make_rng, spawn_substreams
from rag10k._validation import (
    ensure_matrix,
    ensure_text,
    ensure_vector,
    l2_normalize,
    validate_char_span,
)
from rag10k.chunk.splitter import Chunk, chunk_filing, chunk_section
from rag10k.embed.onnx_embedder import (
    OnnxEmbedder,
    default_artifact_dir,
    embedder_artifacts_present,
)
from rag10k.eval.citations import CitationReport, validate_citations
from rag10k.eval.harness import (
    EvalQuestion,
    EvalSummary,
    entailment_proxy,
    recall_at_k,
    run_harness,
)
from rag10k.eval.verdict import GroundingVerdict, answer_is_grounded
from rag10k.generate.answer import (
    GeneratedAnswer,
    build_grounded_prompt,
    generate_answer,
    llm_key_present,
)
from rag10k.generate.extractive import extractive_answer
from rag10k.ingest.client import EdgarClient, Filing, TokenBucket
from rag10k.ingest.fixtures import SYNTHETIC_TICKERS, load_fixture_filing
from rag10k.parse.sections import ParsedFiling, extract_sections, normalize_text
from rag10k.retrieve.index import CorpusIndex, default_index_path, index_present
from rag10k.retrieve.search import RetrievalResult, RetrievedChunk, search
from rag10k.serve import QueryRun, QuerySummary, run_query

__version__ = "0.1.0"

__all__ = [
    # constants
    "ABSTENTION_MESSAGE",
    "ABSTENTION_THRESHOLD",
    "DEFAULT_SECTIONS",
    "DEFAULT_TOP_K",
    "DEFAULT_USER_AGENT",
    "EDGAR_MAX_REQUESTS_PER_SECOND",
    "EMBEDDING_DIM",
    "EPS",
    "MAX_TOP_K",
    "SECTION_LABELS",
    "SYNTHETIC_TICKERS",
    # exceptions
    "ArtifactError",
    # chunk
    "Chunk",
    "CitationError",
    # eval
    "CitationReport",
    # retrieve
    "CorpusIndex",
    # ingest
    "EdgarClient",
    "EvalQuestion",
    "EvalSummary",
    "Filing",
    # generate
    "GeneratedAnswer",
    "GenerationError",
    "GroundingVerdict",
    "IngestError",
    "InsufficientDataError",
    # embed (ONNX serve)
    "OnnxEmbedder",
    "ParseError",
    # parse
    "ParsedFiling",
    # serve
    "QueryRun",
    "QuerySummary",
    "Rag10kError",
    "RetrievalError",
    "RetrievalResult",
    "RetrievedChunk",
    # reproducibility
    "RunManifest",
    "TokenBucket",
    "ValidationError",
    # version
    "__version__",
    "answer_is_grounded",
    "build_grounded_prompt",
    "chunk_filing",
    "chunk_section",
    "config_hash",
    "default_artifact_dir",
    "default_index_path",
    "embedder_artifacts_present",
    # validation
    "ensure_matrix",
    "ensure_text",
    "ensure_vector",
    "entailment_proxy",
    "extract_sections",
    "extractive_answer",
    "generate_answer",
    "index_present",
    "l2_normalize",
    "llm_key_present",
    "load_fixture_filing",
    "make_rng",
    "normalize_text",
    "recall_at_k",
    "run_harness",
    "run_query",
    "search",
    "spawn_substreams",
    "validate_char_span",
    "validate_citations",
]
