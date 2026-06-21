"""Project-wide constants for the rag10k retrieval / IR domain.

Single source of truth for numerical tolerances, embedding/index geometry,
retrieval defaults, the abstention threshold, and EDGAR fair-access literals so
that no magic value is duplicated across modules. Importing this module has no
side effects.
"""

from __future__ import annotations

from typing import Final

# quantcore-candidate: mirrors hrp-portfolio:src/hrp/_constants.py

#: Small positive floor used to guard divisions, ``sqrt`` arguments, and
#: near-zero embedding norms (e.g. cosine = dot / (||a|| ||b||) when a norm is 0).
#: Chosen well above float64 round-off but far below any meaningful similarity.
EPS: Final[float] = 1e-12

#: Embedding dimensionality of the committed sentence-transformer (``all-MiniLM-L6-v2``
#: is 384-dim). The committed ONNX embedder, the index, and every query vector
#: MUST agree on this width.
EMBEDDING_DIM: Final[int] = 384

#: Parity tolerance asserted between the FP32 ONNX-exported embedder and the
#: reference torch sentence-transformer (max absolute element-wise difference).
ONNX_PARITY_ATOL: Final[float] = 1e-4

#: Minimum per-probe cosine similarity asserted between the int8-quantized ONNX
#: embedder and the reference torch sentence-transformer. int8 quantization
#: loosens exact (1e-4) equality, so the export validates COSINE parity instead.
#: Per-channel weight quantization keeps the served embeddings at cosine ~0.99 of
#: torch; the floor is set below that with margin so a quantization regression
#: (e.g. per-tensor, ~0.96) fails the build.
ONNX_COSINE_PARITY_MIN: Final[float] = 0.98

#: Default number of chunks retrieved for a query (the response top-k). Capped at
#: :data:`MAX_TOP_K` at the request boundary.
DEFAULT_TOP_K: Final[int] = 5

#: Hard ceiling on the requested ``top_k`` (guards a pathological request from
#: scanning/returning the whole index).
MAX_TOP_K: Final[int] = 20

#: Cosine-similarity ABSTENTION threshold. If the top retrieved chunk's score is
#: below this floor, the system MUST abstain ("not found in the filing") rather
#: than emit a (generated or extractive) answer. Tuned against the frozen harness
#: so adversarial out-of-document questions abstain and in-document questions do not.
ABSTENTION_THRESHOLD: Final[float] = 0.35

#: Token budget for a single chunk (windowed over a section's tokens). Small
#: enough that a chunk is a citable unit, large enough to carry an answer span.
CHUNK_TOKEN_SIZE: Final[int] = 128

#: Token overlap between adjacent chunks so an answer spanning a window boundary
#: is not split out of every chunk.
CHUNK_TOKEN_OVERLAP: Final[int] = 32

#: Citation-soundness entailment-proxy threshold: a cited chunk is treated as
#: "supporting" an answer claim when their deterministic lexical/embedding overlap
#: clears this floor. The proxy is deterministic and is NOT a learned model.
ENTAILMENT_THRESHOLD: Final[float] = 0.30

#: The canonical abstention message emitted whenever retrieval is below threshold.
ABSTENTION_MESSAGE: Final[str] = "not found in the filing"

#: Canonical 10-K section identifiers the chunker draws provenance from. Keys are
#: the stable internal slugs used across the API/CLI; values are human labels.
SECTION_LABELS: Final[dict[str, str]] = {
    "risk_factors": "Item 1A. Risk Factors",
    "mda": "Item 7. Management's Discussion and Analysis",
}

#: The 10-K sections the chunker extracts and indexes by default.
DEFAULT_SECTIONS: Final[tuple[str, ...]] = ("risk_factors", "mda")

#: A descriptive User-Agent is REQUIRED by SEC EDGAR fair-access policy. The
#: ingest client must send a contactable UA on every request.
DEFAULT_USER_AGENT: Final[str] = "rag-10k/0.1.0 (research; contact: fatihhekimoglu.com)"

#: SEC EDGAR fair-access rate ceiling: no more than 10 requests per second.
EDGAR_MAX_REQUESTS_PER_SECOND: Final[float] = 10.0
