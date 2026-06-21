# Committed serve artifacts (`rag10k/artifacts/`)

This directory ships the **read-only, committed** serve artifacts inside the wheel
(via `[tool.hatch.build.targets.wheel].artifacts`). They are built **offline** by
the `[embed]` path (`rag10k.embed.build_index.build_all`, `rag-10k index`) and are
NEVER produced on the request path:

| file | what | built by |
| --- | --- | --- |
| `embedder.int8.onnx` | int8-quantized sentence-transformer (`all-MiniLM-L6-v2`, 384-dim), ~23MB (<100MB like finbert) | `export_embedder_to_onnx` |
| `tokenizer.json` | the matching `tokenizers` (fast) tokenizer | `export_embedder_to_onnx` |
| `corpus_index.npz` | committed read-only index: chunk vectors + text + provenance (AAPL/MSFT/NVDA cached 10-Ks) | `build_corpus_index` |
| `eval_set.json` | the FROZEN 150-question eval set (questions + gold chunk ids + out-of-document flags) | committed by hand |
| `eval.json` | the committed frozen-harness SUMMARY (recall@k, citation-soundness, abstention rate + provenance) | `build_eval_summary` |

The serve path loads `embedder.int8.onnx` + `tokenizer.json` with onnxruntime +
tokenizers (NO torch) and memory-maps `corpus_index.npz` read-only. `eval_set.json`
is frozen — its contents pin the regression metrics — and `eval.json` is the honest
IR summary the deployed response and the README report quote.

The `.gitignore` ignores `*.onnx` / `*.npz` globally but **allow-lists** these
committed artifacts, so they ship while the FP32 export intermediate
(`embedder.onnx`, ~90MB) and other training intermediates do not.

## How the real artifacts are built (offline `[embed]` path)

`embedder.int8.onnx` + `corpus_index.npz` are the production artifacts produced by
exporting the real `sentence-transformers/all-MiniLM-L6-v2` encoder:

1. **Export** — the torch encoder is traced to FP32 ONNX (`embedder.onnx`,
   asserted at cosine 1.0 vs torch), then **per-channel int8-quantized**
   (`embedder.int8.onnx`, ~23MB). The export asserts COSINE parity vs torch on a
   probe set (int8 loosens the exact 1e-4 element diff, so the build gate is the
   minimum per-probe cosine `>= ONNX_COSINE_PARITY_MIN`; a clean per-channel export
   sits at cosine ~0.99).
2. **Index** — every cached-corpus chunk is embedded through the just-exported
   torch-free ONNX graph (so the committed vectors are byte-identical to what the
   request path produces for the same text) and written, row-aligned with its text
   and provenance, to `corpus_index.npz`.
3. **Eval** — the FROZEN 150-Q harness is run over that index + embedder and its
   honest summary is committed to `eval.json`.

Everything runs CPU-only (the serve path is `onnxruntime` CPUExecutionProvider) and
is deterministic. The serve API contract and response shape never depend on which
build produced the artifacts.
