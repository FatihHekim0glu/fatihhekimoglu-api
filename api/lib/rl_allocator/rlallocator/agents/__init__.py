"""Agents subpackage: the PPO allocator (lazy torch/sb3), the numpy baselines, the ONNX serve policy.

- :mod:`rlallocator.agents.baselines` — the pure-numpy allocation baselines
  (equal-weight 1/N, Markowitz mean-variance, risk-parity), computed LIVE on the
  serve path with TRAIN-window-only covariance (no look-ahead);
- :mod:`rlallocator.agents.ppo` — the SB3 PPO allocator (torch + sb3 + gymnasium
  imported LAZILY, the ``[train]`` extra);
- :mod:`rlallocator.agents.onnx_policy` — the committed ONNX policy served via
  onnxruntime (NEVER torch) → simplex weights.

Importing this subpackage has no side effects (torch / sb3 / onnxruntime are
imported lazily inside their functions).
"""

from __future__ import annotations
