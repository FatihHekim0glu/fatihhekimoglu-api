"""SB3 PPO allocator wrapper (torch + stable-baselines3, LAZY, the ``[train]`` extra).

The OFFLINE training agent: wraps a Stable-Baselines3 PPO policy trained in the
gymnasium-wrapped :class:`rlallocator.env.portfolio_env.PortfolioEnv`. torch,
stable-baselines3, and gymnasium are imported LAZILY inside the methods (the
``[train]`` extra), so importing this module pulls in NO torch / sb3 / gymnasium
and has no side effects. This agent is NEVER invoked on the request path — the
served policy is the exported ONNX MLP (:mod:`rlallocator.agents.onnx_policy`).

FALLBACK (documented; the shipped path): the SB3 ``MlpPolicy`` mixes a value head,
an action-distribution layer, and (for the continuous portfolio action) a Gaussian
``log_std`` parameter into its forward pass, none of which the serve path needs.
Rather than wrestle the full SB3 ``ActorCriticPolicy`` graph through
``torch.onnx.export`` (it is not ONNX-clean), this wrapper EXTRACTS only the small
policy MLP — the shared/pi feature extractor plus the per-asset score head — into a
standalone plain-torch :class:`~torch.nn.Module` (the rl-trader / gnn-stocks dense
pattern) and exports THAT to ONNX. The exported graph is a clean dense
``obs -> per-asset scores`` MLP validated 1e-4 against the same extracted torch
module; the serve path projects those scores onto the long-only weight simplex via
:func:`rlallocator._validation.project_to_simplex`. The honest-null deliverable holds
either way; the shipped path (SB3 PPO trained, policy-MLP extracted + exported
ONNX-clean) is documented in the README.

When the ``[train]`` extra (torch + stable-baselines3 + gymnasium) is absent — e.g.
the lean serve container or CI — the offline training / export methods raise
``NotImplementedError`` (the training path is simply not available there); the serve
path never needs them. Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rlallocator._exceptions import ValidationError
from rlallocator._typing import FloatArray, ObservationMatrix

if TYPE_CHECKING:  # pragma: no cover - typing-only import (torch is the [train] extra)
    import torch

#: The ONNX input tensor name (the per-bar observation: look-back window + weights).
OBSERVATION_INPUT: str = "observation"

#: The ONNX output tensor name (the per-asset score head; projected to the simplex).
ACTION_OUTPUT: str = "asset_scores"

#: The ONNX opset for the LEGACY exporter (``dynamo=False``; the dynamo path needs
#: onnxscript, which is NOT a dependency of this project).
ONNX_OPSET: int = 17


@dataclass(frozen=True, slots=True)
class PpoConfig:
    """Immutable PPO hyper-parameter configuration (defines the ONNX export signature).

    Attributes
    ----------
    obs_dim:
        The observation dimension (flattened look-back window + current weights:
        ``lookback * n_assets + n_assets``).
    n_assets:
        The number of assets (the per-asset score head width; softmax/projected to
        the long-only weight simplex).
    hidden_dim:
        The MLP hidden width (small — the policy is a tiny MLP for clean ONNX).
    n_steps:
        PPO rollout length per update.
    total_timesteps:
        Total training timesteps per seed.
    learning_rate:
        The Adam learning rate.
    gamma:
        The reward discount factor.
    """

    obs_dim: int = 390
    n_assets: int = 6
    hidden_dim: int = 64
    n_steps: int = 2048
    total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    gamma: float = 0.99

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this config."""
        return asdict(self)


def _build_policy_mlp(config: PpoConfig) -> torch.nn.Sequential:
    """Construct the standalone policy MLP that the ONNX serve graph is exported from.

    LAZY IMPORT: ``torch`` (the ``[train]`` extra) is imported inside this function.
    The architecture mirrors the SB3 ``MlpPolicy`` pi-branch default (two
    ``Tanh``-activated hidden layers) followed by the linear per-asset score head, so
    the SB3 policy's pi-extractor + action-net weights load straight in. It is a clean
    dense ``obs -> per-asset scores`` graph (no value head, no Gaussian sampler), which
    is exactly what ``torch.onnx.export(..., dynamo=False)`` serializes cleanly; the
    serve path projects the scores onto the long-only weight simplex.
    """
    import torch
    from torch import nn

    return nn.Sequential(
        nn.Linear(config.obs_dim, config.hidden_dim),
        nn.Tanh(),
        nn.Linear(config.hidden_dim, config.hidden_dim),
        nn.Tanh(),
        nn.Linear(config.hidden_dim, config.n_assets),
    ).to(torch.float32)


class PpoAgent:
    """A thin SB3 PPO wrapper that trains in the gym-wrapped portfolio env (LAZY torch/sb3).

    Construction is cheap and import-pure: torch / stable-baselines3 / gymnasium are
    imported only inside :meth:`train` / :meth:`export_onnx` / :meth:`policy_scores`
    (the ``[train]`` extra). The trained policy is exported to a small ONNX MLP and
    served through onnxruntime — torch is NEVER imported on the request path. When the
    ``[train]`` extra is absent the training / export methods raise
    ``NotImplementedError`` (the offline path is unavailable there); the serve path
    never needs them.
    """

    def __init__(self, config: PpoConfig | None = None) -> None:
        """Record the config; defer the SB3 model build to :meth:`train`.

        Parameters
        ----------
        config:
            The PPO hyper-parameter config; ``None`` => :class:`PpoConfig` defaults.
        """
        self._config: PpoConfig = config if config is not None else PpoConfig()
        #: The fitted SB3 ``PPO`` model (set by :meth:`train`).
        self._model: Any | None = None
        #: The extracted standalone policy MLP (lazily built from the SB3 policy).
        self._policy_net: torch.nn.Module | None = None

    @property
    def config(self) -> PpoConfig:
        """Return the immutable PPO configuration this agent was built with."""
        return self._config

    @property
    def is_trained(self) -> bool:
        """Return ``True`` once :meth:`train` has produced a fitted policy."""
        return self._policy_net is not None

    def train(self, env: Any, *, seed: int = 7) -> PpoAgent:
        """Train the PPO policy in the gym-wrapped portfolio env for one seed (LAZY torch/sb3).

        LAZY IMPORT: ``stable_baselines3`` / ``torch`` (the ``[train]`` extra) load
        inside this method. Trains a PPO policy on the gymnasium-wrapped
        :class:`PortfolioEnv` for ``config.total_timesteps`` with the given ``seed``,
        then extracts the policy MLP (pi feature extractor + per-asset score head)
        into a standalone plain-torch module ready for ONNX export. NEVER invoked on
        the request path.

        Parameters
        ----------
        env:
            A gymnasium-API env (e.g. ``PortfolioEnv.as_gym_env()``).
        seed:
            The torch / SB3 RNG seed for this training run (one per seed-lottery draw).

        Returns
        -------
        PpoAgent
            ``self``, with a fitted policy.

        Raises
        ------
        ValueError
            If ``seed`` is negative.
        NotImplementedError
            If the ``[train]`` extra (torch + stable-baselines3 + gymnasium) is not
            installed (e.g. the lean serve container / CI) — the offline training path
            is unavailable there and the serve path never needs it.
        """
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}.")
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:  # pragma: no cover - exercised only without [train]
            raise NotImplementedError(
                "PpoAgent.train requires the [train] extra (torch + stable-baselines3 + "
                "gymnasium), which is not installed; the offline training path is "
                "unavailable here. Install it with `uv pip install -e '.[train]'`."
            ) from exc

        cfg = self._config
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=cfg.n_steps,
            learning_rate=cfg.learning_rate,
            gamma=cfg.gamma,
            seed=seed,
            policy_kwargs={"net_arch": [cfg.hidden_dim, cfg.hidden_dim]},
            verbose=0,
        )
        model.learn(total_timesteps=cfg.total_timesteps)
        self._model = model
        self._policy_net = self._extract_policy_mlp(model)
        return self

    def _extract_policy_mlp(self, model: Any) -> torch.nn.Sequential:
        """Copy the SB3 policy's pi-extractor + per-asset score head into a standalone MLP.

        LAZY IMPORT: ``torch`` is imported inside this method. The SB3
        ``ActorCriticPolicy`` keeps a shared/pi ``mlp_extractor.policy_net`` (the two
        ``Tanh`` hidden layers) and an ``action_net`` (the linear per-asset score
        head). We load both weight sets into the standalone :func:`_build_policy_mlp`
        module so the exported ONNX graph reproduces the policy's per-asset scores
        exactly, with no value head and no Gaussian sampler. The serve path then
        projects those scores onto the long-only weight simplex.
        """
        import torch

        policy_net = _build_policy_mlp(self._config)
        sb3_policy = model.policy
        # The pi branch: mlp_extractor.policy_net is Sequential[Linear,Tanh,Linear,Tanh];
        # action_net is the final Linear score head. Map them onto our [0,2,4] layout.
        pi_net = sb3_policy.mlp_extractor.policy_net
        action_net = sb3_policy.action_net
        with torch.no_grad():
            policy_net[0].weight.copy_(pi_net[0].weight)
            policy_net[0].bias.copy_(pi_net[0].bias)
            policy_net[2].weight.copy_(pi_net[2].weight)
            policy_net[2].bias.copy_(pi_net[2].bias)
            policy_net[4].weight.copy_(action_net.weight)
            policy_net[4].bias.copy_(action_net.bias)
        policy_net.eval()
        return policy_net

    def policy_scores(self, observations: ObservationMatrix) -> FloatArray:
        """Return the fitted policy's per-asset score logits for a batch (LAZY torch).

        LAZY IMPORT: ``torch`` is imported inside this method. Runs the extracted
        policy MLP's forward pass on ``observations`` and returns the raw per-asset
        scores — the reference output the exported ONNX graph is validated against to
        1e-4; the scores are projected onto the weight simplex by the env / serve path.

        Parameters
        ----------
        observations:
            A ``(batch, obs_dim)`` observation matrix.

        Returns
        -------
        FloatArray
            The ``(batch, n_assets)`` per-asset score logits.

        Raises
        ------
        ValidationError
            If the agent has not been trained or the input shape is wrong.
        """
        # Validate preconditions (trained + well-shaped input) BEFORE importing torch
        # so an untrained / malformed call raises a clean ValidationError even on the
        # lean serve image where torch is absent.
        net = self._require_policy_net()
        batch = self._coerce_batch(observations)

        import torch

        with torch.no_grad():
            out = net(torch.from_numpy(batch.astype("float32")))
        scores: FloatArray = np.asarray(out.numpy(), dtype="float64")
        return scores

    def export_onnx(self, out_path: str | Path, *, rtol: float = 1e-4) -> Path:
        """Export ONLY the policy MLP to ONNX with a 1e-4 torch-vs-ONNX parity gate.

        LAZY IMPORT: ``torch`` (the ``[train]`` extra) and ``onnxruntime`` load inside
        this method. Exports the extracted policy network (obs -> per-asset scores)
        with ``torch.onnx.export(..., dynamo=False)`` and a DYNAMIC batch axis, then
        runs the torch forward pass and the exported ONNX forward pass on a probe
        batch and asserts they agree to ``rtol`` (the parity gate). The serve path then
        uses ONLY the ONNX graph (no torch). The env + backtester are pure numpy, so no
        gymnasium is needed at serve.

        Parameters
        ----------
        out_path:
            Destination ``.onnx`` path (the committed ``artifacts/policy.onnx``).
        rtol:
            Relative tolerance for the torch-vs-ONNX parity assertion (default 1e-4).

        Returns
        -------
        pathlib.Path
            The written ``.onnx`` path.

        Raises
        ------
        ValidationError
            If the agent is untrained, ``rtol`` is invalid, or the torch and ONNX
            forward passes disagree beyond ``rtol``.
        """
        if not np.isfinite(rtol) or rtol <= 0.0:
            raise ValidationError(f"rtol must be finite and > 0, got {rtol!r}.")

        # Validate the trained precondition BEFORE importing torch so an untrained
        # call raises a clean ValidationError even where torch is absent.
        net = self._require_policy_net()

        import torch

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # A deterministic probe batch defines the export signature and the parity
        # gate; the batch axis is exported DYNAMIC so any per-bar batch binds at serve.
        probe = np.zeros((4, self._config.obs_dim), dtype="float64")
        probe[1] = 1.0
        probe[2] = np.linspace(-1.0, 1.0, self._config.obs_dim)
        probe[3] = np.linspace(0.5, -0.5, self._config.obs_dim)
        sample = torch.from_numpy(probe.astype("float32"))

        # LEGACY exporter (dynamo=False — the dynamo path needs onnxscript, NOT a
        # dependency). The weights W are baked in; only the batch axis is dynamic.
        torch.onnx.export(
            net,
            (sample,),
            str(out),
            input_names=[OBSERVATION_INPUT],
            output_names=[ACTION_OUTPUT],
            dynamic_axes={
                OBSERVATION_INPUT: {0: "batch"},
                ACTION_OUTPUT: {0: "batch"},
            },
            opset_version=ONNX_OPSET,
            dynamo=False,
        )

        # Parity gate: the served ONNX graph MUST reproduce the torch forward pass to
        # ``rtol`` on the probe batch, else the export is rejected (no silent drift).
        with torch.no_grad():
            torch_scores = np.asarray(net(sample).numpy(), dtype="float64")

        from rlallocator.agents.onnx_policy import OnnxPolicy

        onnx_scores = np.asarray(OnnxPolicy(out).predict_scores(probe), dtype="float64")
        if onnx_scores.shape != torch_scores.shape:  # pragma: no cover - defensive
            raise ValidationError(
                f"export_onnx: ONNX output shape {onnx_scores.shape} does not match torch "
                f"{torch_scores.shape}."
            )
        max_abs_diff = float(np.max(np.abs(torch_scores - onnx_scores)))
        if not np.allclose(torch_scores, onnx_scores, rtol=rtol, atol=rtol):
            raise ValidationError(
                f"export_onnx: torch-vs-ONNX parity failed (max abs diff {max_abs_diff:.3e} "
                f"> rtol {rtol:.1e}); the exported policy graph drifted from torch."
            )
        return out

    def _require_policy_net(self) -> torch.nn.Module:
        """Return the extracted policy MLP or raise if the agent is untrained."""
        if self._policy_net is None:
            raise ValidationError(
                "PpoAgent has no fitted policy; call train(env, seed=...) before "
                "policy_scores / export_onnx."
            )
        return self._policy_net

    def _coerce_batch(self, observations: ObservationMatrix) -> FloatArray:
        """Coerce an observation batch to a finite ``(batch, obs_dim)`` float64 array.

        Flattens a single observation vector to a 1-row batch, checks the dimension
        matches ``config.obs_dim``, and enforces finiteness — the shared input
        boundary for :meth:`policy_scores`.
        """
        arr = np.asarray(observations, dtype="float64")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValidationError(f"observations must be 1-D or 2-D, got ndim={arr.ndim}.")
        if arr.shape[0] == 0:
            raise ValidationError("observations must contain at least one row.")
        if arr.shape[1] != self._config.obs_dim:
            raise ValidationError(
                f"observations have obs_dim={arr.shape[1]} but the policy expects "
                f"{self._config.obs_dim}."
            )
        if not bool(np.isfinite(arr).all()):
            raise ValidationError("observations must be finite (no NaN/inf).")
        return arr
