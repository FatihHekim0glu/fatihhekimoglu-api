# Serve artifacts

This directory holds the **committed serve artifacts** that ride inside the wheel
and the vendored API package. The lean serve container loads these via
`onnxruntime` + `tokenizers` only — **never** torch/transformers.

When the offline `[train]` path runs (`finbert-sentiment train` →
`finbert_sentiment.model.export`), it writes here:

- `model.int8.onnx` — the dynamic-int8-quantized DistilBERT graph (inputs:
  `input_ids`, `attention_mask`; output: 3-way logits in
  `negative, neutral, positive` order).
- `tokenizer.json` — the matching fast-tokenizer for `tokenizers`.
- `metrics.json` — the OFFLINE-measured evaluation bundle (macro-F1, per-class
  P/R/F1, confusion matrix, bootstrap CIs, lexicon/class-prior macro-F1, McNemar)
  that the API reports verbatim (never recomputed per request).

If no transformer was trained in this build, these files are absent and the
predictor falls back to the torch-free **lexicon** baseline — which needs no
artifact at all. The `.gitignore` un-ignores `*.onnx`, `tokenizer.json`, and
`metrics.json` here so they are tracked when present.
