# Shipped model artifacts

This directory holds the committed ONNX model artifacts served at inference time
and the precomputed `metrics.json` for the cross-sectional comparison:

- `gcn.onnx`, `graphsage.onnx` — the dense-adjacency Graph Convolutional Network
  and mean-aggregator GraphSAGE, exported by `gnnstocks.train.export_onnx` from the
  offline-trained plain-torch models (NO torch-geometric) and loaded by
  `gnnstocks.models.onnx_runtime.OnnxGraphModel`. The per-rebalance normalized
  adjacency and node features are graph **inputs**; the weights are baked in;
- `ridge.onnx` — the cross-sectional ridge baseline, exported via skl2onnx, served
  through the same onnxruntime path for an exact-parity comparison;
- `metrics.json` — the precomputed out-of-sample rank-IC / long-short metrics so
  the deployed default returns instantly without re-running the GNN forward pass.

The shipped models are trained on the **synthetic block-factor panel** (see
`gnnstocks.data.synthetic.block_factor_panel`) — `K` sector blocks, each a latent
factor plus within-block correlation plus dominant idiosyncratic noise, with no
real market data or API key in this repo. By construction the graph is
**descriptive** (it recovers the blocks) but the next-period cross-sectional return
is near-random, so the honest NULL (the GNN does **not** reliably beat the per-node
ridge / cross-sectional-momentum baselines) holds and the artifacts are fully
reproducible. Retrain on real point-in-time data via `gnn-stocks train` after
loading a Polygon-PIT universe.

`*.onnx` files and `metrics.json` are **committed** (they ship inside the wheel);
`*.pt`/`*.pth`/`*.pkl` training intermediates are git-ignored. The serve container
runs onnxruntime only — **never torch**.
