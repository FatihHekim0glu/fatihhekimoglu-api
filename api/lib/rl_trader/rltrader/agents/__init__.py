"""Agents subpackage: the SB3 PPO trainer, the live baselines, and the ONNX policy serve.

The PPO agent (:mod:`rltrader.agents.ppo`) trains offline (torch + sb3 + gymnasium,
LAZY, the ``[train]`` extra); the baselines (:mod:`rltrader.agents.baselines`) are
pure numpy and run LIVE; the ONNX policy (:mod:`rltrader.agents.onnx_policy`) serves
the committed exported policy through onnxruntime (NEVER torch). Importing this
subpackage has no side effects and pulls in nothing heavy.
"""

from __future__ import annotations
