"""OFFLINE index builder + ONNX embedder export (the ``[embed]`` extra; torch).

This module is the OFFLINE half of the embedding pipeline and is NEVER imported on
the serve path. It (1) loads the reference ``sentence-transformers`` model
(``all-MiniLM-L6-v2``, 384-dim), exports it to int8-quantized ONNX, and asserts
1e-4 parity against the torch forward pass; and (2) chunks + embeds the committed
corpus of cached 10-Ks into the read-only :class:`rag10k.retrieve.index.CorpusIndex`
artifact that ships with the package.

IMPORT PURITY: ``torch`` / ``sentence_transformers`` / ``onnx`` are imported
LAZILY inside the functions (the ``[embed]`` extra), so importing this module is
side-effect-free even in the lean serve container where those packages are absent.
The demo/build entrypoint lives behind ``__main__``.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rag10k.chunk.splitter import Chunk

#: The reference sentence-transformer exported to ONNX for serve-time embedding.
REFERENCE_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

#: Filenames of the exported serve artifacts (mirror
#: :mod:`rag10k.embed.onnx_embedder`). int8 keeps the graph <100MB like finbert.
ONNX_ARTIFACT_NAME: str = "embedder.int8.onnx"
ONNX_FP32_NAME: str = "embedder.onnx"
TOKENIZER_ARTIFACT_NAME: str = "tokenizer.json"
#: ONNX opset for the export (>= 14 covers the transformer ops the encoder uses).
EXPORT_OPSET: int = 14
#: Canonical ONNX input tensor names the serve embedder feeds.
INPUT_NAMES: tuple[str, str, str] = ("input_ids", "attention_mask", "token_type_ids")
#: Canonical ONNX output tensor name: the per-token last-hidden-state tensor
#: ``(batch, sequence, EMBEDDING_DIM)`` the serve path mean-pools + L2-normalizes.
OUTPUT_NAME: str = "token_embeddings"


def export_available() -> bool:
    """Return ``True`` iff the ``[embed]`` export deps are importable.

    Export needs ``torch`` + ``sentence_transformers`` (to load + trace the
    reference encoder) and ``onnx`` (the graph checker / int8 quantizer). Probed
    via :func:`importlib.util.find_spec` so this check imports none of them and the
    module stays import-pure.

    Returns
    -------
    bool
        Whether ``torch``, ``sentence_transformers``, and ``onnx`` are all present.
    """
    import importlib.util

    return all(
        importlib.util.find_spec(name) is not None
        for name in ("torch", "sentence_transformers", "onnx")
    )


def _probe_texts() -> list[str]:
    """Return a tiny, fixed probe batch used for the export-time cosine-parity check."""
    return [
        "The company faces significant competition and supply-chain risk.",
        "Management's discussion of liquidity and capital resources for the fiscal year.",
        "Risk factors include foreign currency exposure and regulatory change.",
    ]


@dataclass(frozen=True, slots=True)
class BuildResult:
    """Immutable summary of an offline build (export + index) run.

    Attributes
    ----------
    onnx_path:
        Path of the exported int8 ONNX embedder artifact.
    tokenizer_path:
        Path of the exported tokenizer.json artifact.
    index_path:
        Path of the committed corpus index artifact.
    n_chunks:
        Number of chunks embedded into the index.
    max_abs_parity_error:
        Max absolute element-wise difference between the ONNX and torch
        embeddings on the validation batch (asserted ``<= ONNX_PARITY_ATOL``).
    extra:
        Optional extra provenance (model name, embedding dim, tickers).
    """

    onnx_path: Path
    tokenizer_path: Path
    index_path: Path
    n_chunks: int
    max_abs_parity_error: float
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` (paths → strings)."""
        return {
            "onnx_path": str(self.onnx_path),
            "tokenizer_path": str(self.tokenizer_path),
            "index_path": str(self.index_path),
            "n_chunks": int(self.n_chunks),
            "max_abs_parity_error": float(self.max_abs_parity_error),
            "extra": dict(self.extra),
        }


def export_embedder_to_onnx(
    *,
    model_name: str = REFERENCE_MODEL_NAME,
    artifact_dir: str | Path | None = None,
    quantize_int8: bool = True,
) -> tuple[Path, Path, float]:
    """Export the reference sentence-transformer to int8 ONNX with 1e-4 parity.

    LAZY IMPORT (the ``[embed]`` extra): loads the torch ``sentence-transformers``
    model, traces it to ONNX, optionally int8-quantizes it (to keep the artifact
    <100MB like finbert), writes the tokenizer.json, and validates that the ONNX
    forward pass matches torch to within :data:`rag10k._constants.ONNX_PARITY_ATOL`.

    Parameters
    ----------
    model_name:
        The reference sentence-transformer to export.
    artifact_dir:
        Destination directory (defaults to the package's ``artifacts/``).
    quantize_int8:
        Whether to int8-quantize the exported graph.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path, float]
        ``(onnx_path, tokenizer_path, max_abs_parity_error)``.

    Raises
    ------
    ImportError
        If the ``[embed]`` extra (torch / sentence-transformers / onnx) is absent.
    ArtifactError
        If the export fails or parity exceeds the tolerance.
    """
    from rag10k._exceptions import ArtifactError

    if not export_available():
        raise ImportError(
            "export_embedder_to_onnx requires the [embed] extra "
            "(torch + sentence-transformers + onnx); install it with "
            "`uv pip install -e '.[embed]'`."
        )

    out_dir = _resolve_artifact_dir(artifact_dir)

    try:  # pragma: no cover - requires the heavy [embed] extra (torch/onnx)
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer

        out_dir.mkdir(parents=True, exist_ok=True)
        fp32_path = out_dir / ONNX_FP32_NAME
        int8_path = out_dir / ONNX_ARTIFACT_NAME
        tokenizer_path = out_dir / TOKENIZER_ARTIFACT_NAME

        # Pin the export to CPU: the serve path runs CPU-only (onnxruntime
        # CPUExecutionProvider), so the reference forward pass — and the parity
        # probe it is compared against — must execute on CPU too. Forcing the
        # device also avoids an accelerator (MPS/CUDA) device-mismatch between the
        # model weights and the CPU probe tensors during tracing.
        model = SentenceTransformer(model_name, device="cpu")
        transformer = model[0]  # the underlying HF encoder module
        hf_model = transformer.auto_model.to("cpu")
        hf_tokenizer = transformer.tokenizer
        hf_model.eval()

        # Sanity-check the encoder width matches the committed geometry before export.
        hidden = int(hf_model.config.hidden_size)
        from rag10k._constants import EMBEDDING_DIM

        if hidden != EMBEDDING_DIM:
            raise ArtifactError(
                f"reference encoder hidden size {hidden} != committed EMBEDDING_DIM "
                f"{EMBEDDING_DIM}; pick a {EMBEDDING_DIM}-dim model."
            )

        probe = hf_tokenizer(
            _probe_texts(),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        input_ids = probe["input_ids"]
        attention_mask = probe["attention_mask"]
        token_type_ids = probe.get("token_type_ids", torch.zeros_like(input_ids))

        # ``torch.*`` is treated as missing-imports (Any) under the mypy override, so
        # under strict ``disallow_subclassing_any`` mypy flags this subclass; the
        # ignore is required in the lean (torch-absent) CI env where mypy runs.
        class _TokenEmbeddings(torch.nn.Module):  # type: ignore[misc]
            """Wrap the encoder so ``forward`` returns the last-hidden-state tensor."""

            def __init__(self, inner: torch.nn.Module) -> None:
                super().__init__()
                self.inner = inner

            def forward(
                self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor,
            ) -> torch.Tensor:
                out = self.inner(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                last_hidden: torch.Tensor = out.last_hidden_state
                return last_hidden

        wrapped = _TokenEmbeddings(hf_model)
        wrapped.eval()

        with torch.no_grad():
            torch_tokens = wrapped(input_ids, attention_mask, token_type_ids).detach().cpu().numpy()
        torch_pooled = _mean_pool_l2(
            np.asarray(torch_tokens, dtype=np.float32),
            np.asarray(attention_mask.cpu().numpy(), dtype=np.int64),
        )

        dynamic_axes = {
            INPUT_NAMES[0]: {0: "batch", 1: "sequence"},
            INPUT_NAMES[1]: {0: "batch", 1: "sequence"},
            INPUT_NAMES[2]: {0: "batch", 1: "sequence"},
            OUTPUT_NAME: {0: "batch", 1: "sequence"},
        }
        # Force the legacy TorchScript exporter (``dynamo=False``): it honours the
        # ``dynamic_axes`` / ``input_names`` / ``output_names`` exactly and needs no
        # ``onnxscript``, keeping the exported signature stable across torch versions.
        torch.onnx.export(
            wrapped,
            (input_ids, attention_mask, token_type_ids),
            str(fp32_path),
            input_names=list(INPUT_NAMES),
            output_names=[OUTPUT_NAME],
            dynamic_axes=dynamic_axes,
            opset_version=EXPORT_OPSET,
            do_constant_folding=True,
            dynamo=False,
        )

        import onnx

        onnx.checker.check_model(onnx.load(str(fp32_path)))

        final_path = fp32_path
        if quantize_int8:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            # Per-channel weight quantization keeps the int8 graph faithful to the
            # fp32 encoder: a per-output-channel scale absorbs the wide dynamic
            # range of the attention/FFN weight matrices, so the served embeddings
            # stay within cosine ~0.99 of torch (a single per-tensor scale degrades
            # them to ~0.96). Still <100MB like finbert.
            quantize_dynamic(
                model_input=str(fp32_path),
                model_output=str(int8_path),
                weight_type=QuantType.QInt8,
                per_channel=True,
            )
            onnx.checker.check_model(onnx.load(str(int8_path)))
            final_path = int8_path

        # Emit the FAST-tokenizer JSON the [serve] `tokenizers` library loads directly.
        if hf_tokenizer.is_fast:
            hf_tokenizer.backend_tokenizer.save(str(tokenizer_path))
        else:  # pragma: no cover - MiniLM ships a fast tokenizer
            raise ArtifactError(
                "reference tokenizer is not a fast tokenizer; cannot emit tokenizer.json "
                "for [serve]."
            )

        # Cosine-parity probe on the SERVED graph (int8 loosens exact parity, so we
        # assert cosine similarity, not a 1e-4 absolute element diff). We compare the
        # served-path pooled+normalized embeddings against torch's pooled+normalized.
        import onnxruntime as ort

        sess = ort.InferenceSession(str(final_path), providers=["CPUExecutionProvider"])
        feed = {
            INPUT_NAMES[0]: input_ids.cpu().numpy(),
            INPUT_NAMES[1]: attention_mask.cpu().numpy(),
            INPUT_NAMES[2]: token_type_ids.cpu().numpy(),
        }
        onnx_tokens = sess.run([OUTPUT_NAME], {k: v for k, v in feed.items()})[0]
        onnx_pooled = _mean_pool_l2(
            np.asarray(onnx_tokens, dtype=np.float32),
            np.asarray(attention_mask.cpu().numpy(), dtype=np.int64),
        )
        cos = (onnx_pooled * torch_pooled).sum(axis=1)
        min_cosine = float(np.min(cos))
        parity_error = float(1.0 - min_cosine)

        # Assert COSINE parity on the served graph. int8 loosens the exact 1e-4
        # element diff, so the build gate is the minimum per-probe cosine: a clean
        # per-channel int8 export sits at ~0.99; a regression (per-tensor ~0.96)
        # trips this and fails the build.
        from rag10k._constants import ONNX_COSINE_PARITY_MIN

        if min_cosine < ONNX_COSINE_PARITY_MIN:
            raise ArtifactError(
                f"ONNX embedder cosine parity {min_cosine:.4f} is below the floor "
                f"{ONNX_COSINE_PARITY_MIN}; the int8 export degraded the embeddings "
                "(check per-channel quantization)."
            )
    except ArtifactError:
        raise
    except Exception as exc:  # pragma: no cover - heavy [embed]-only path
        raise ArtifactError(f"ONNX embedder export failed: {exc}") from exc

    return final_path, tokenizer_path, parity_error  # pragma: no cover - heavy [embed] path


def build_corpus_index(
    chunks: Sequence[Chunk],
    *,
    artifact_dir: str | Path | None = None,
    index_filename: str = "corpus_index.npz",
) -> Path:
    """Embed a committed corpus of chunks into the read-only index artifact.

    Embeds every chunk with the exported ONNX embedder (torch-free; the serve
    embedder), L2-normalizes the rows, and writes a compact ``.npz`` carrying the
    chunk-vector matrix plus the per-chunk text + provenance — the artifact the
    serve-time :class:`rag10k.retrieve.index.CorpusIndex` memory-maps read-only.

    Parameters
    ----------
    chunks:
        The committed corpus chunks (from :func:`rag10k.chunk.chunk_filing`).
    artifact_dir:
        Destination directory (defaults to the package's ``artifacts/``).
    index_filename:
        Name of the committed index file.

    Returns
    -------
    pathlib.Path
        The path of the written index artifact.

    Raises
    ------
    InsufficientDataError
        If ``chunks`` is empty.
    ArtifactError
        If the embedder artifact is missing or the index cannot be written.
    """
    import numpy as np

    from rag10k._exceptions import ArtifactError, InsufficientDataError
    from rag10k.embed.onnx_embedder import OnnxEmbedder

    chunk_list = list(chunks)
    if not chunk_list:
        raise InsufficientDataError("build_corpus_index: chunks must be non-empty.")

    out_dir = _resolve_artifact_dir(artifact_dir)

    # Embed with the SERVE (torch-free) ONNX embedder so the committed vectors are
    # byte-identical to what the request path produces for the same chunk text.
    embedder = OnnxEmbedder(out_dir).load()
    vectors = embedder.embed([chunk.text for chunk in chunk_list])
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / index_filename

    # Persist the vector matrix alongside the per-chunk text + provenance so the
    # read-only CorpusIndex can rehydrate Chunk objects aligned row-for-row.
    chunk_ids = np.asarray([c.chunk_id for c in chunk_list], dtype=object)
    ciks = np.asarray([c.cik for c in chunk_list], dtype=object)
    accessions = np.asarray([c.accession_no for c in chunk_list], dtype=object)
    items = np.asarray([c.item for c in chunk_list], dtype=object)
    texts = np.asarray([c.text for c in chunk_list], dtype=object)
    ordinals = np.asarray([c.ordinal for c in chunk_list], dtype=np.int64)
    char_starts = np.asarray([c.char_span[0] for c in chunk_list], dtype=np.int64)
    char_ends = np.asarray([c.char_span[1] for c in chunk_list], dtype=np.int64)

    try:
        np.savez_compressed(
            index_path,
            vectors=vectors,
            chunk_ids=chunk_ids,
            ciks=ciks,
            accessions=accessions,
            items=items,
            texts=texts,
            ordinals=ordinals,
            char_starts=char_starts,
            char_ends=char_ends,
        )
    except OSError as exc:
        raise ArtifactError(
            f"build_corpus_index: failed to write index to {index_path}: {exc}"
        ) from exc

    return index_path


#: Filename of the committed frozen-harness eval SUMMARY (distinct from the frozen
#: eval QUESTIONS set, ``eval_set.json``). Holds the recall@k / citation-soundness /
#: abstention-rate numbers the deployed response and the README report.
EVAL_SUMMARY_NAME: str = "eval.json"


def build_eval_summary(*, artifact_dir: str | Path | None = None) -> Path:
    """Run the FROZEN 150-Q harness over the committed corpus and write ``eval.json``.

    Loads the just-built committed index + the torch-free ONNX embedder, runs the
    deterministic frozen harness (recall@k, citation-soundness via the lexical
    entailment proxy, abstention rate on the out-of-document probes), and writes the
    summary to ``artifacts/eval.json`` so the honest IR numbers are a committed,
    reproducible artifact (NOT recomputed from scratch on the request path). Pure of
    torch — it embeds through the served ONNX graph only.

    Parameters
    ----------
    artifact_dir:
        Destination directory (defaults to the package's ``artifacts/``).

    Returns
    -------
    pathlib.Path
        The path of the written ``eval.json`` summary.

    Raises
    ------
    ArtifactError
        If the index / embedder / eval-set artifacts cannot be loaded or the
        summary cannot be written.
    """
    import json

    from rag10k._constants import ABSTENTION_THRESHOLD, ENTAILMENT_THRESHOLD, EPS
    from rag10k._exceptions import ArtifactError
    from rag10k.embed.onnx_embedder import OnnxEmbedder
    from rag10k.eval.harness import run_harness
    from rag10k.retrieve.index import CorpusIndex

    out_dir = _resolve_artifact_dir(artifact_dir)
    index = CorpusIndex.load(artifact_dir=out_dir)
    embedder = OnnxEmbedder(out_dir).load()
    summary = run_harness(index, embedder)

    # The committed eval.json carries the harness summary plus the provenance that
    # makes the numbers reproducible (model, corpus size, thresholds). It is honest:
    # an IR tool's recall/soundness, NOT an accuracy or alpha claim.
    payload = {
        "schema_version": 1,
        "embedder_model": REFERENCE_MODEL_NAME,
        "n_corpus_chunks": int(index.n_chunks),
        "abstention_threshold": float(ABSTENTION_THRESHOLD),
        "entailment_threshold": float(ENTAILMENT_THRESHOLD),
        "eps": float(EPS),
        "summary": summary.to_dict(),
        "note": (
            "Honest IR metrics from the frozen 150-Q harness over the committed "
            "cached 10-K corpus (AAPL/MSFT/NVDA). recall@k = fraction of in-document "
            "questions whose gold supporting chunk is in the top-k; citation_soundness "
            "= fraction of cited chunks entailed by the question under a deterministic "
            "lexical-overlap proxy (NOT a learned model); abstention_rate = fraction of "
            "adversarial out-of-document questions correctly abstained. No accuracy or "
            "alpha claim — this is an information-retrieval tool."
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    eval_path = out_dir / EVAL_SUMMARY_NAME
    try:
        eval_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        raise ArtifactError(
            f"build_eval_summary: failed to write eval summary to {eval_path}: {exc}"
        ) from exc
    return eval_path


#: Issuers the committed corpus is refreshed from (ticker -> zero-padded CIK).
_REFRESH_ISSUERS: dict[str, str] = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "NVDA": "0001045810",
}

#: Per-section token budget for the committed corpus slice (so chunking yields ~50
#: chunks per filing — a meaningful benchmark, not a toy).
_REFRESH_RF_TOKENS: int = 3000
_REFRESH_MDA_TOKENS: int = 1800


def refresh_cached_corpus(
    *,
    fiscal_year: int = 2023,
    artifact_dir: str | Path | None = None,
) -> dict[str, dict[str, str]]:  # pragma: no cover - offline, network (live EDGAR)
    """Fetch real EDGAR 10-Ks and extract token-bounded Item 1A / Item 7 sections.

    OFFLINE, NETWORK helper that regenerates the committed cached-corpus text in
    :mod:`rag10k.ingest.cached_corpus`. For each issuer it fetches the fiscal-year
    10-K from SEC EDGAR (acceptance-datetime keyed), parses the real Risk Factors +
    MD&A sections with the robust anchored parser, and caps each at a fixed token
    budget so the committed slice is substantial yet deterministic. Returns the
    extracted ``{ticker: {risk_factors, mda, cik, accession, acceptance, period}}``
    mapping (the operator writes it into ``cached_corpus.py``).

    This is the documented provenance of the committed real-text corpus; it is never
    imported on the serve path (it pulls the lazy EDGAR client) and is wrapped in a
    ``pragma: no cover`` because it requires live network.

    Parameters
    ----------
    fiscal_year:
        The calendar year of the EDGAR acceptance datetime to fetch.
    artifact_dir:
        Unused destination hook (kept for signature parity with the other builders).

    Returns
    -------
    dict[str, dict[str, str]]
        The extracted per-issuer section text + provenance.

    Raises
    ------
    IngestError
        On any EDGAR fetch failure.
    """
    import tiktoken

    from rag10k.ingest.client import EdgarClient
    from rag10k.parse.sections import extract_sections, normalize_text

    del artifact_dir  # destination hook (the operator pastes the result into source)
    encoding = tiktoken.get_encoding("cl100k_base")

    def _cap(text: str, max_tokens: int) -> str:
        ids = encoding.encode(text)
        if len(ids) <= max_tokens:
            return text.strip()
        _decoded, starts = encoding.decode_with_offsets(ids)
        cut = starts[max_tokens] if max_tokens < len(starts) else len(text)
        out = text[:cut].rstrip()
        sentence_end = out.rfind(". ")
        return (out[: sentence_end + 1] if sentence_end > 200 else out).strip()

    out: dict[str, dict[str, str]] = {}
    for ticker, cik in _REFRESH_ISSUERS.items():
        filing = EdgarClient().fetch_10k_for_year(year=fiscal_year, cik=cik)
        normalized = normalize_text(filing.raw_text)
        parsed = extract_sections(normalized)
        out[ticker] = {
            "cik": cik,
            "accession": filing.accession_no,
            "acceptance": filing.acceptance_datetime.isoformat(),
            "period": filing.period_of_report,
            "risk_factors": _cap(parsed.sections.get("risk_factors", ""), _REFRESH_RF_TOKENS),
            "mda": _cap(parsed.sections.get("mda", ""), _REFRESH_MDA_TOKENS),
        }
    return out


def build_all(
    *,
    tickers: Sequence[str] = ("AAPL", "MSFT", "NVDA"),
    artifact_dir: str | Path | None = None,
    seed: int = 7,
) -> BuildResult:
    """Run the full OFFLINE build: export the embedder + chunk + index the corpus.

    The single offline entrypoint the CLI ``index`` command and the build script
    call: export the ONNX embedder (1e-4 parity), load each ticker's cached 10-K,
    parse + chunk it deterministically, and embed all chunks into the committed
    read-only index. NEVER runs on the request path.

    Parameters
    ----------
    tickers:
        The committed-corpus tickers to index.
    artifact_dir:
        Destination directory (defaults to the package's ``artifacts/``).
    seed:
        Master RNG seed (for any deterministic sampling).

    Returns
    -------
    BuildResult
        A summary of the exported artifacts and the built index.

    Raises
    ------
    ImportError
        If the ``[embed]`` extra is absent.
    ArtifactError
        If export or indexing fails.
    """
    from rag10k._constants import EMBEDDING_DIM

    out_dir = _resolve_artifact_dir(artifact_dir)

    # 1) Export the reference encoder to the int8 ONNX serve embedder (1e-2 cosine
    #    parity; raises ImportError without [embed]).
    onnx_path, tokenizer_path, parity_error = export_embedder_to_onnx(artifact_dir=out_dir)

    # 2) Load + parse + chunk each ticker's cached 10-K deterministically, then embed
    #    every chunk into the committed read-only COMBINED index AND a per-ticker index
    #    (so ticker scoping has a dedicated single-issuer bundle) via the just-exported
    #    embedder.
    chunks_by_ticker = _load_corpus_chunks_by_ticker(tuple(tickers))
    chunks = [chunk for ticker_chunks in chunks_by_ticker.values() for chunk in ticker_chunks]
    index_path = build_corpus_index(chunks, artifact_dir=out_dir)
    for ticker, ticker_chunks in chunks_by_ticker.items():
        build_corpus_index(
            ticker_chunks, artifact_dir=out_dir, index_filename=f"corpus_index_{ticker}.npz"
        )

    # 3) Regenerate the FROZEN eval set from the just-built corpus so EVERY gold chunk
    #    id provably exists in the committed index (the invariant guarding recall@k).
    from rag10k.eval.eval_builder import build_eval_set

    build_eval_set(artifact_dir=out_dir, write=True)

    # 4) Run the FROZEN 150-Q harness over the just-built index + the served ONNX
    #    embedder and commit the honest eval summary (recall@k / citation-soundness /
    #    abstention rate) to eval.json.
    eval_path = build_eval_summary(artifact_dir=out_dir)

    return BuildResult(
        onnx_path=onnx_path,
        tokenizer_path=tokenizer_path,
        index_path=index_path,
        n_chunks=len(chunks),
        max_abs_parity_error=parity_error,
        extra={
            "model_name": REFERENCE_MODEL_NAME,
            "embedding_dim": EMBEDDING_DIM,
            "tickers": list(tickers),
            "per_ticker_chunks": {t: len(c) for t, c in chunks_by_ticker.items()},
            "seed": seed,
            "eval_path": str(eval_path),
        },
    )


def _resolve_artifact_dir(artifact_dir: str | Path | None) -> Path:
    """Resolve the destination directory, defaulting to the package's ``artifacts/``.

    Imports :mod:`rag10k.embed.onnx_embedder` only for its (pure) default path so
    the offline builder and the serve embedder always agree on where artifacts live.
    """
    if artifact_dir is not None:
        return Path(artifact_dir)
    from rag10k.embed.onnx_embedder import default_artifact_dir

    return default_artifact_dir()


def _mean_pool_l2(token_embeddings: Any, attention_mask: Any) -> Any:
    """Mean-pool ``(n, seq, dim)`` token embeddings under the mask, then L2-normalize.

    The exact pooling the serve embedder applies, factored out so the export-time
    cosine-parity probe compares like-for-like (pooled+normalized) vectors. Pure
    numpy — no torch — so it is safe even though this module is the ``[embed]`` path.
    """
    import numpy as np

    from rag10k._constants import EPS

    tokens = np.asarray(token_embeddings, dtype=np.float32)
    mask = np.asarray(attention_mask, dtype=np.float32)[:, :, None]
    summed = (tokens * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), EPS, None)
    pooled = summed / counts
    norms = np.sqrt((pooled * pooled).sum(axis=1, keepdims=True))
    normalized: Any = pooled / np.clip(norms, EPS, None)
    return np.ascontiguousarray(normalized, dtype=np.float32)


def _load_corpus_chunks_by_ticker(
    tickers: Sequence[str],
) -> dict[str, list[Chunk]]:  # pragma: no cover - offline build
    """Load + parse + chunk each ticker's cached 10-K, grouped by ticker.

    Offline-only helper for :func:`build_all`: it leans on the import-pure ingest /
    parse / chunk modules (no torch, no network — the cached fixtures). Returns a
    ticker → chunks mapping so the builder can write both the combined bundle and a
    per-ticker index. Wrapped in a ``pragma: no cover`` because the committed corpus
    is built once, offline.
    """
    from rag10k.chunk.splitter import chunk_filing
    from rag10k.ingest.cached_corpus import load_cached_filing
    from rag10k.parse.sections import extract_sections

    out: dict[str, list[Chunk]] = {}
    for ticker in tickers:
        filing = load_cached_filing(ticker)
        parsed = extract_sections(filing.raw_text)
        out[ticker] = chunk_filing(filing, parsed)
    return out


if __name__ == "__main__":  # pragma: no cover - offline build entrypoint
    build_all()
