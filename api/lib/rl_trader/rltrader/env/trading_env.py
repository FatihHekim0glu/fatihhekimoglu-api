"""Causal single-asset trading environment (gymnasium-compatible, strictly next-bar reward).

A minimal single-asset trading env where, at bar ``t``:

- the OBSERVATION is a look-back window of past returns / features PLUS the
  current position — ONLY information available at ``t`` (data <= ``t``);
- the ACTION is a discrete ``{short, flat, long}`` choice (mapped to a target
  position in ``{-1, 0, +1}``) or a continuous target weight in ``[-1, 1]``;
- the REWARD is ``position_t * return_{t -> t+1} - cost_bps*|Δposition| -
  slippage`` — the position set at ``t`` earns the NEXT bar's return, so the
  reward is STRICTLY CAUSAL with no look-ahead.

``reset(seed)`` is deterministic. The env is the step-by-step oracle the
vectorized backtester (:mod:`rltrader.env.backtester`) must match to 1e-10
(:mod:`rltrader.env.parity`), which is the load-bearing look-ahead guard.

gymnasium is an OPTIONAL dependency (the ``[train]`` extra) imported LAZILY inside
:meth:`TradingEnv.as_gym_env`; the core env is pure numpy so it runs on the serve
path without gymnasium. Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

import numpy as np

from rltrader._exceptions import InsufficientDataError, ValidationError
from rltrader._typing import FloatArray, ObservationVector, ReturnSeries
from rltrader._validation import ensure_series

#: The discrete action set: short (-1), flat (0), long (+1).
DISCRETE_ACTIONS: tuple[int, ...] = (-1, 0, 1)


def _coerce_returns(returns: ReturnSeries, *, name: str = "returns") -> FloatArray:
    """Coerce a return path to a finite 1-D float64 ndarray (no NaN, length >= 2).

    Funnels the env's bound return path through :func:`rltrader._validation.ensure_series`
    so the env shares the house coercion/finiteness guarantees, then materializes a
    contiguous float64 numpy view (the env's hot loop is pure numpy).

    Raises
    ------
    InsufficientDataError
        If fewer than two bars are present (no causal ``r_{t -> t+1}`` step exists).
    """
    arr = ensure_series(returns, name=name, allow_nan=False).to_numpy(dtype="float64")
    if arr.size < 2:
        raise InsufficientDataError(
            f"{name} must have at least 2 bars to form one causal reward step, got {arr.size}."
        )
    return arr


def _coerce_action(action: int | float, config: EnvConfig) -> float:
    """Coerce + validate a single action to a target position respecting ``config``.

    Discrete envs accept only the ``{-1, 0, +1}`` members of :data:`DISCRETE_ACTIONS`;
    continuous envs accept any finite target weight in ``[-max_position, max_position]``.

    Raises
    ------
    ValidationError
        If the action is non-finite, or out of the configured action set / range.
    """
    value = float(action)
    if not np.isfinite(value):
        raise ValidationError(f"action must be finite, got {action!r}.")
    bound = float(config.max_position)
    if config.continuous:
        if value < -bound or value > bound:
            raise ValidationError(
                f"continuous action must lie in [{-bound}, {bound}], got {value!r}."
            )
        return value
    # Discrete: the action names short/flat/long; map to the signed unit position.
    if value not in {float(a) for a in DISCRETE_ACTIONS}:
        raise ValidationError(
            f"discrete action must be one of {DISCRETE_ACTIONS}, got {action!r}."
        )
    return value * bound


@dataclass(frozen=True, slots=True)
class EnvConfig:
    """Immutable configuration of the causal trading env.

    Attributes
    ----------
    lookback:
        Number of trailing return bars in the observation window (``>= 1``).
    cost_bps:
        Per-side transaction cost in basis points charged on ``|Δposition|``.
    slippage_bps:
        Per-trade slippage in basis points charged on ``|Δposition|``.
    continuous:
        If ``True``, the action is a continuous target weight in ``[-1, 1]``; if
        ``False``, a discrete ``{short, flat, long}`` choice.
    max_position:
        The position bound (``1.0`` => positions in ``[-1, 1]``).
    """

    lookback: int = 32
    cost_bps: float = 5.0
    slippage_bps: float = 1.0
    continuous: bool = False
    max_position: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this config."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StepResult:
    """Immutable result of one env step (the Gym 5-tuple, frozen).

    Attributes
    ----------
    observation:
        The next-bar observation vector (data <= the new ``t`` only).
    reward:
        The strictly-causal reward ``position_t * return_{t->t+1} - cost - slip``.
    terminated:
        Whether the episode reached the end of the price path.
    truncated:
        Whether the episode hit ``episode_len`` before the path end.
    info:
        Auxiliary diagnostics (realized position, turnover, gross/net return).
    """

    observation: ObservationVector
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this step result."""
        return {
            "observation": [float(x) for x in np.asarray(self.observation).ravel()],
            "reward": float(self.reward),
            "terminated": bool(self.terminated),
            "truncated": bool(self.truncated),
            "info": dict(self.info),
        }


class TradingEnv:
    """A causal single-asset trading environment (gymnasium-compatible API).

    The env wraps a fixed single-asset return path and exposes the standard
    ``reset`` / ``step`` API. The position chosen at bar ``t`` earns the
    ``t -> t+1`` return (strictly causal); the observation at ``t`` uses ONLY data
    ``<= t``. Construction is cheap and import-pure — gymnasium is imported lazily
    only when :meth:`as_gym_env` is called.
    """

    def __init__(
        self,
        returns: ReturnSeries,
        config: EnvConfig | None = None,
        *,
        episode_len: int | None = None,
    ) -> None:
        """Bind the return path + config; defer all RNG to :meth:`reset`.

        Parameters
        ----------
        returns:
            The single-asset per-bar return path the episode walks over.
        config:
            The env configuration; ``None`` => :class:`EnvConfig` defaults.
        episode_len:
            Max bars per episode before truncation; ``None`` => the full path.

        Raises
        ------
        ValidationError
            If ``returns`` is too short for the look-back window or malformed.
        """
        cfg = config if config is not None else EnvConfig()
        if cfg.lookback < 1:
            raise ValidationError(f"EnvConfig.lookback must be >= 1, got {cfg.lookback}.")
        if cfg.cost_bps < 0.0 or not np.isfinite(cfg.cost_bps):
            raise ValidationError(f"EnvConfig.cost_bps must be finite and >= 0, got {cfg.cost_bps}.")
        if cfg.slippage_bps < 0.0 or not np.isfinite(cfg.slippage_bps):
            raise ValidationError(
                f"EnvConfig.slippage_bps must be finite and >= 0, got {cfg.slippage_bps}."
            )
        if cfg.max_position <= 0.0 or not np.isfinite(cfg.max_position):
            raise ValidationError(
                f"EnvConfig.max_position must be finite and > 0, got {cfg.max_position}."
            )
        if episode_len is not None and episode_len < 1:
            raise ValidationError(f"episode_len must be >= 1 when given, got {episode_len}.")

        self._returns: FloatArray = _coerce_returns(returns)
        self._config: EnvConfig = cfg
        # The number of scorable bars: a position can be set at every bar t with a
        # forward return r_{t+1}, i.e. t in [0, N-2]. This is look-back independent
        # (the look-back only restricts which *observations* are well-formed).
        self._n_returns: int = int(self._returns.size)
        self._n_scored: int = self._n_returns - 1
        self._episode_len: int | None = episode_len
        # Mutable per-episode state (set on reset).
        self._t: int = 0
        self._position: float = 0.0
        self._steps_taken: int = 0
        self._done: bool = True

    def reset(self, *, seed: int | None = None) -> tuple[ObservationVector, dict[str, Any]]:
        """Reset to the first decision bar; return ``(observation, info)`` deterministically.

        Parameters
        ----------
        seed:
            Optional seed for a deterministic reset (episode start position).

        Returns
        -------
        tuple[ObservationVector, dict[str, Any]]
            The first observation (data <= the start bar) and an info dict.

        Raises
        ------
        InsufficientDataError
            If the return path is too short for one full look-back observation
            window plus a forward return (``N < lookback + 1``).
        """
        # ``seed`` is accepted for the gymnasium-API contract and determinism; the
        # env is itself deterministic (the return path is fixed), so the seed does
        # not introduce randomness — reset to the first well-formed decision bar.
        _ = seed
        start = self._config.lookback - 1
        if start > self._n_scored - 1:
            raise InsufficientDataError(
                f"return path of length {self._n_returns} is too short for a look-back of "
                f"{self._config.lookback}: need at least {self._config.lookback + 1} bars."
            )
        self._t = start
        self._position = 0.0
        self._steps_taken = 0
        self._done = False
        obs = self._observe(self._t)
        info: dict[str, Any] = {"t": self._t, "position": self._position}
        return obs, info

    def step(self, action: int | float) -> StepResult:
        r"""Advance one bar; return the strictly-causal :class:`StepResult`.

        The ``action`` sets the target position held over the CURRENT bar; the
        realized reward is ``position_t * return_{t -> t+1} - cost_bps*|Δposition|
        - slippage_bps*|Δposition|`` (the position set at ``t`` earns the NEXT
        bar's return). The observation returned is for the NEW bar and uses only
        data ``<= t+1``.

        Parameters
        ----------
        action:
            A discrete ``{-1, 0, +1}`` choice or a continuous target weight in
            ``[-1, 1]`` (per :attr:`EnvConfig.continuous`).

        Returns
        -------
        StepResult
            The next observation, the causal reward, the done flags, and info.

        Raises
        ------
        ValidationError
            If ``action`` is out of range or the episode is already done.
        """
        if self._done:
            raise ValidationError("step called on a finished episode; call reset() first.")

        prev_position = self._position
        new_position = _coerce_action(action, self._config)
        delta = abs(new_position - prev_position)
        # The position set at t earns the NEXT bar's return (strictly causal).
        forward_return = float(self._returns[self._t + 1])
        gross = new_position * forward_return
        friction = self._friction(delta)
        reward = gross - friction

        # Commit the position and advance the clock by one bar.
        self._position = new_position
        self._steps_taken += 1
        self._t += 1

        # Terminated: the new bar has no forward return (we scored the last bar).
        terminated = self._t >= self._n_scored
        # Truncated: hit the episode budget before the path end.
        truncated = (
            self._episode_len is not None
            and self._steps_taken >= self._episode_len
            and not terminated
        )
        self._done = bool(terminated or truncated)

        # The observation is for the NEW bar (data <= t+1); when the episode is over
        # there is no further decision bar, so reuse the last well-formed window.
        obs_index = self._t if not terminated else self._n_scored - 1
        observation = self._observe(obs_index)
        info: dict[str, Any] = {
            "t": self._t,
            "position": new_position,
            "turnover": delta,
            "gross_return": gross,
            "cost": friction,
            "net_return": reward,
        }
        return StepResult(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _friction(self, position_change: float) -> float:
        """Return the per-bar (cost + slippage) charge on an absolute position change.

        Charged IDENTICALLY to the vectorized backtester:
        ``(cost_bps + slippage_bps) / 1e4 * |Δposition|`` (return units).
        """
        rate = (self._config.cost_bps + self._config.slippage_bps) / 10_000.0
        return rate * float(position_change)

    def _observe(self, t: int) -> ObservationVector:
        """Build the observation at decision bar ``t`` (data <= ``t`` only).

        The observation is the trailing look-back window of past returns
        ``[r_{t-lookback+1}, ..., r_t]`` concatenated with the current position. It
        NEVER reads ``r_{t+1}`` or beyond, so it is strictly causal. The public
        ``reset`` / ``step`` path only ever asks for ``t >= lookback - 1`` (the first
        decision bar onward), so the window is always fully populated.
        """
        lookback = self._config.lookback
        lo = t - lookback + 1
        window = self._returns[lo : t + 1]
        return np.concatenate((window.astype("float64"), [float(self._position)]))

    def rollout(self, actions: FloatArray) -> FloatArray:
        """Replay a full action sequence step-by-step and return per-bar net rewards.

        Drives the env from ``reset`` through one action per bar and collects the
        per-bar net reward series. This is the step-by-step ORACLE the vectorized
        backtester must reproduce to 1e-10 (the parity look-ahead guard).

        Parameters
        ----------
        actions:
            The per-bar action / target-position sequence to replay.

        Returns
        -------
        FloatArray
            The per-bar net reward series produced by the step-by-step rollout.

        Raises
        ------
        ValidationError
            If ``actions`` length does not match the episode's bar count.
        """
        acts = ensure_series(actions, name="actions", allow_nan=False).to_numpy(dtype="float64")
        if acts.size != self._n_returns:
            raise ValidationError(
                f"actions length ({acts.size}) must match the return path length "
                f"({self._n_returns}); the position at each bar t earns r_{{t+1}}."
            )
        # Step bar-by-bar over the scorable window t in [0, N-2]. The position set at
        # bar t (acts[t]) earns the NEXT bar's return; friction is charged on the
        # change vs the previous bar's position (the book opens flat). This is the
        # single step-by-step oracle the vectorized backtester must reproduce.
        net = np.empty(self._n_scored, dtype="float64")
        prev_position = 0.0
        for t in range(self._n_scored):
            position = _coerce_action(acts[t], self._config)
            delta = abs(position - prev_position)
            net[t] = position * float(self._returns[t + 1]) - self._friction(delta)
            prev_position = position
        return net

    def as_gym_env(self) -> Any:
        """Return a gymnasium-API wrapper around this env (LAZY ``gymnasium`` import).

        LAZY IMPORT: ``gymnasium`` (the ``[train]`` extra) is imported inside this
        method so importing :mod:`rltrader.env.trading_env` never imports
        gymnasium. The wrapper exposes ``observation_space`` / ``action_space`` and
        delegates ``reset`` / ``step`` to this env for SB3 PPO training.

        Returns
        -------
        Any
            A ``gymnasium.Env`` instance suitable for SB3.

        Raises
        ------
        ImportError
            If the ``[train]`` extra (gymnasium) is not installed.
        """
        # LAZY: importing gymnasium only here keeps the serve path torch/gym-free.
        try:
            import gymnasium as gym
            from gymnasium import spaces
        except ImportError as exc:  # pragma: no cover - exercised only without [train]
            raise ImportError(
                "TradingEnv.as_gym_env requires the [train] extra (gymnasium). "
                "Install it with `uv pip install -e '.[train]'`."
            ) from exc

        env = self

        class _GymTradingEnv(gym.Env):  # type: ignore[misc]
            """Thin gymnasium adapter delegating reset/step to the pure-numpy env."""

            metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

            def __init__(self) -> None:
                super().__init__()
                obs_dim = env._config.lookback + 1
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
                )
                if env._config.continuous:
                    bound = float(env._config.max_position)
                    self.action_space = spaces.Box(
                        low=-bound, high=bound, shape=(1,), dtype=np.float64
                    )
                else:
                    self.action_space = spaces.Discrete(len(DISCRETE_ACTIONS))

            def reset(
                self, *, seed: int | None = None, options: dict[str, Any] | None = None
            ) -> tuple[ObservationVector, dict[str, Any]]:
                super().reset(seed=seed)
                return env.reset(seed=seed)

            def step(
                self, action: Any
            ) -> tuple[ObservationVector, float, bool, bool, dict[str, Any]]:
                resolved = env._resolve_gym_action(action)
                result = env.step(resolved)
                return (
                    result.observation,
                    result.reward,
                    result.terminated,
                    result.truncated,
                    result.info,
                )

        return _GymTradingEnv()

    def _resolve_gym_action(self, action: Any) -> int | float:
        """Map a gymnasium action (Discrete index or Box weight) to an env action.

        For a discrete env, SB3 emits an index into :data:`DISCRETE_ACTIONS`; map it
        back to the signed ``{-1, 0, +1}`` member. For a continuous env, SB3 emits a
        length-1 Box; take its scalar.
        """
        if self._config.continuous:
            return float(np.asarray(action, dtype="float64").ravel()[0])
        return int(DISCRETE_ACTIONS[int(action)])
