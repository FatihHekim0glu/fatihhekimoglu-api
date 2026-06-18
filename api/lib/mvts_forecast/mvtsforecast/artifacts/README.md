# Shipped model artifacts

This directory holds the committed ONNX model artifacts served at inference time
and the precomputed `metrics.json` for the deep models:

- `lstm.onnx`, `patchtst.onnx`, `transformer_vs.onnx` — exported by
  `mvtsforecast.train.export_onnx` from the offline-trained torch models and
  loaded by `mvtsforecast.models.onnx_runtime.OnnxForecaster`;
- `metrics.json` — the precomputed out-of-sample return-space metrics so the
  deployed default returns instantly without re-running the deep forward pass.

The shipped models are trained on the **synthetic multivariate panel** (see
`mvtsforecast.data.synthetic.synthetic_panel`) — a weak common factor plus
dominant idiosyncratic noise and mild autocorrelation, with no real market data
or API key in this repo. By construction the next-step return is dominated by
unforecastable noise, so the honest NULL (the deep models do **not** reliably
beat the naive random-walk baseline) holds and the artifacts are fully
reproducible. Retrain on real data via `mvts-forecast train --data-source yfinance`.

`*.onnx` files and `metrics.json` are **committed** (they ship inside the wheel);
`*.pt`/`*.pth`/`*.pkl` training intermediates are git-ignored. The serve container
runs onnxruntime only — **never torch**.
