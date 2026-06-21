# Committed artifacts

This directory ships the offline-trained deliverables consumed by the lean serve path:

- `policy.onnx` — the exported PPO policy MLP (obs → per-asset scores → simplex
  weights), loaded lazily via onnxruntime (NEVER torch). Validated 1e-4 vs torch at
  export time. `< 10 MB`.
- `metrics.json` — the precomputed out-of-sample metrics from the purged walk-forward:
  the per-seed OOS Sharpes (the seed lottery), the median-seed OOS equity curve, the
  RL allocation-over-time weight path, the across-seed band, the Diebold-Mariano
  p-value vs. the best baseline, the Deflated Sharpe, the PBO, and the honest
  `n_effective_trials` (#seeds × #HP configs).

Both are written by `rlallocator.train.train_pipeline` (the `[train]` extra: torch +
stable-baselines3 + gymnasium) and are deliberately tracked in git so they ship in the
wheel and back the deployed serve path. Until the offline training is run, the serve
path degrades to a baselines-only comparison and the honest-NULL placeholder verdict
(`rl_beats_baselines = False`).
