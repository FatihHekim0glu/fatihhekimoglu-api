# Committed artifacts

The OFFLINE training pipeline (`rl-trader train`, the `[train]` extra) writes the
shipped, deployed inference artifacts here:

- `policy.onnx` — the exported PPO policy MLP (obs -> action), validated to 1e-4
  against the SB3 torch policy. Served torch-free via onnxruntime by the request
  path. Kept `<10MB`.
- `metrics.json` — the precomputed per-seed OOS metrics, the seed-lottery
  dispersion, the Diebold-Mariano-vs-buy-hold p-value, the Deflated Sharpe (with
  the honest `n_trials = #seeds x #HP configs`), and the PURE `rl_beats_baseline`
  verdict.

These are TRACKED in git (they ship in the wheel and back the deployed serve path);
the `.gitignore` ignores stray `*.onnx` / `metrics.json` elsewhere but never these.

The committed `policy.onnx` (a dense `obs → 3` graph, the gnn-stocks ONNX-clean
pattern) + `metrics.json` ship the honest-NULL result: on the GBM-regime synthetic
null the agent does **not** beat buy-and-hold out-of-sample (`rl_beats_baseline =
False`; median-seed OOS Sharpe ≈ −0.77 vs. buy-hold ≈ +0.24; across-seed Sharpe band
≈ [−1.16, −0.38]). They are produced by `rltrader.train.train_pipeline` (the offline
`rl-trader train` path) and reproduced torch-free by the serve path via onnxruntime.
