"""Causal multi-asset portfolio environment (gymnasium-compatible, strictly next-bar reward).

A multi-asset portfolio env where, at bar ``t``:

- the OBSERVATION is a flattened look-back window of per-asset returns PLUS the
  current portfolio weights — ONLY information available at ``t`` (data <= ``t``);
- the ACTION is a raw per-asset score vector projected onto the long-only weight
  SIMPLEX (sum to one, non-negative) via
  :func:`rlallocator._validation.project_to_simplex`, so the held weights are a
  valid simplex EVERY bar by construction;
- the REWARD is ``w_t · r_{t -> t+1} - cost_bps/1e4 * ||w_t - w_{t-1}||_1`` — the
  weights set at ``t`` earn the NEXT bar's asset returns, so the reward is STRICTLY
  CAUSAL with no look-ahead.

``reset(seed)`` is deterministic. The env is the step-by-step oracle the vectorized
backtester (:mod:`rlallocator.env.backtester`) must match to 1e-10
(:mod:`rlallocator.env.parity`), which is the load-bearing look-ahead guard.

gymnasium is an OPTIONAL dependency (the ``[train]`` extra) imported LAZILY inside
:meth:`PortfolioEnv.as_gym_env`; the core env is pure numpy so it runs on the serve
path without gymnasium. Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

import numpy as np

from rlallocator._exceptions import InsufficientDataError, ValidationError
from rlallocator._typing import FloatArray, ObservationVector, ReturnPanel, WeightVector
from rlallocator._validation import is_simplex, project_to_simplex


def _coerce_panel(returns: ReturnPanel, *, name: str = "returns") -> FloatArray:
    """Coerce a return panel to a finite 2-D ``(n_bars, n_assets)`` float64 array.

    Funnels the env's bound return panel through a finiteness check and materializes
    a contiguous float64 numpy view (the env's hot loop is pure numpy).

    Raises
    ------
    ValidationError
        If ``returns`` is not 2-D or contains non-finite values.
    InsufficientDataError
        If fewer than two bars are present (no causal ``r_{t -> t+1}`` step exists).
    """
    arr = np.asarray(returns, dtype="float64")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValidationError(f"{name} must be 2-D (n_bars, n_assets), got ndim={arr.ndim}.")
    if arr.shape[1] == 0:
        raise ValidationError(f"{name} must have at least one asset.")
    if not bool(np.isfinite(arr).all()):
        raise ValidationError(f"{name} contains non-finite values.")
    if arr.shape[0] < 2:
        raise InsufficientDataError(
            f"{name} must have at least 2 bars to form one causal reward step, got {arr.shape[0]}."
        )
    return arr


@dataclass(frozen=True, slots=True)
class PortfolioEnvConfig:
    """Immutable configuration of the causal multi-asset portfolio env.

    Attributes
    ----------
    lookback:
        Number of trailing return bars in the observation window (``>= 1``).
    cost_bps:
        Per-side turnover cost in basis points charged on ``||Δw||_1``.
    long_only:
        If ``True`` (default), the action is projected onto the long-only simplex
        (non-negative weights summing to one); else a long/short unit budget.
    """

    lookback: int = 64
    cost_bps: float = 10.0
    long_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this config."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PortfolioStepResult:
    """Immutable result of one env step (the Gym 5-tuple, frozen).

    Attributes
    ----------
    observation:
        The next-bar observation vector (data <= the new ``t`` only).
    reward:
        The strictly-causal reward ``w_t · r_{t->t+1} - turnover_cost``.
    terminated:
        Whether the episode reached the end of the return panel.
    truncated:
        Whether the episode hit ``episode_len`` before the panel end.
    info:
        Auxiliary diagnostics (realized weights, turnover, gross/net return).
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


class PortfolioEnv:
    """A causal multi-asset portfolio environment (gymnasium-compatible API).

    The env wraps a fixed multi-asset return panel and exposes the standard
    ``reset`` / ``step`` API. The weight vector chosen at bar ``t`` (the projection
    of the raw action onto the simplex) earns the ``t -> t+1`` asset returns
    (strictly causal); the observation at ``t`` uses ONLY data ``<= t``. Construction
    is cheap and import-pure — gymnasium is imported lazily only when
    :meth:`as_gym_env` is called.
    """

    def __init__(
        self,
        returns: ReturnPanel,
        config: PortfolioEnvConfig | None = None,
        *,
        episode_len: int | None = None,
    ) -> None:
        """Bind the return panel + config; defer all RNG to :meth:`reset`.

        Parameters
        ----------
        returns:
            The multi-asset per-bar return panel the episode walks over.
        config:
            The env configuration; ``None`` => :class:`PortfolioEnvConfig` defaults.
        episode_len:
            Max bars per episode before truncation; ``None`` => the full panel.

        Raises
        ------
        ValidationError
            If ``returns`` is too short for the look-back window or malformed.
        """
        cfg = config if config is not None else PortfolioEnvConfig()
        if cfg.lookback < 1:
            raise ValidationError(f"PortfolioEnvConfig.lookback must be >= 1, got {cfg.lookback}.")
        if cfg.cost_bps < 0.0 or not np.isfinite(cfg.cost_bps):
            raise ValidationError(
                f"PortfolioEnvConfig.cost_bps must be finite and >= 0, got {cfg.cost_bps}."
            )
        if episode_len is not None and episode_len < 1:
            raise ValidationError(f"episode_len must be >= 1 when given, got {episode_len}.")

        self._returns: FloatArray = _coerce_panel(returns)
        self._config: PortfolioEnvConfig = cfg
        self._n_bars: int = int(self._returns.shape[0])
        self._n_assets: int = int(self._returns.shape[1])
        # Scorable bars: weights can be set at every bar t with a forward return
        # r_{t+1}, i.e. t in [0, N-2]. Look-back independent (the look-back only
        # restricts which *observations* are well-formed).
        self._n_scored: int = self._n_bars - 1
        self._episode_len: int | None = episode_len
        # Mutable per-episode state (set on reset).
        self._t: int = 0
        self._weights: FloatArray = np.zeros(self._n_assets, dtype="float64")
        self._steps_taken: int = 0
        self._done: bool = True

    @property
    def n_assets(self) -> int:
        """Return the number of assets in the bound return panel."""
        return self._n_assets

    @property
    def obs_dim(self) -> int:
        """Return the observation dimension: ``lookback * n_assets + n_assets``.

        The observation is a flattened look-back window of per-asset returns
        (``lookback * n_assets``) concatenated with the current weight vector
        (``n_assets``).
        """
        return self._config.lookback * self._n_assets + self._n_assets

    def reset(self, *, seed: int | None = None) -> tuple[ObservationVector, dict[str, Any]]:
        """Reset to the first decision bar; return ``(observation, info)`` deterministically.

        Parameters
        ----------
        seed:
            Optional seed for the gymnasium-API contract; the env is itself
            deterministic (the panel is fixed), so the seed adds no randomness.

        Returns
        -------
        tuple[ObservationVector, dict[str, Any]]
            The first observation (data <= the start bar) and an info dict.

        Raises
        ------
        InsufficientDataError
            If the panel is too short for one full look-back observation window plus
            a forward return (``N < lookback + 1``).
        """
        _ = seed
        start = self._config.lookback - 1
        if start > self._n_scored - 1:
            raise InsufficientDataError(
                f"return panel of length {self._n_bars} is too short for a look-back of "
                f"{self._config.lookback}: need at least {self._config.lookback + 1} bars."
            )
        self._t = start
        self._weights = np.zeros(self._n_assets, dtype="float64")
        self._steps_taken = 0
        self._done = False
        obs = self._observe(self._t)
        info: dict[str, Any] = {"t": self._t, "weights": self._weights.tolist()}
        return obs, info

    def step(self, action: FloatArray) -> PortfolioStepResult:
        r"""Advance one bar; return the strictly-causal :class:`PortfolioStepResult`.

        The ``action`` (a raw per-asset score vector) is projected onto the weight
        simplex to set the target weights held over the CURRENT bar; the realized
        reward is ``w_t · r_{t -> t+1} - cost_bps/1e4 * ||w_t - w_{t-1}||_1`` (the
        weights set at ``t`` earn the NEXT bar's asset returns). The observation
        returned is for the NEW bar and uses only data ``<= t+1``.

        Parameters
        ----------
        action:
            A raw per-asset score / logit vector ``(n_assets,)`` projected onto the
            simplex.

        Returns
        -------
        PortfolioStepResult
            The next observation, the causal reward, the done flags, and info.

        Raises
        ------
        ValidationError
            If ``action`` is misshaped / non-finite or the episode is already done.
        """
        if self._done:
            raise ValidationError("step called on a finished episode; call reset() first.")

        prev_weights = self._weights
        new_weights = self._resolve_weights(action)
        turnover = float(np.abs(new_weights - prev_weights).sum())
        # The weights set at t earn the NEXT bar's asset returns (strictly causal).
        forward_returns = self._returns[self._t + 1]
        gross = float(new_weights @ forward_returns)
        cost = self._turnover_cost(turnover)
        reward = gross - cost

        # Commit the weights and advance the clock by one bar.
        self._weights = new_weights
        self._steps_taken += 1
        self._t += 1

        # Terminated: the new bar has no forward return (we scored the last bar).
        terminated = self._t >= self._n_scored
        truncated = (
            self._episode_len is not None
            and self._steps_taken >= self._episode_len
            and not terminated
        )
        self._done = bool(terminated or truncated)

        obs_index = self._t if not terminated else self._n_scored - 1
        observation = self._observe(obs_index)
        info: dict[str, Any] = {
            "t": self._t,
            "weights": new_weights.tolist(),
            "turnover": turnover,
            "gross_return": gross,
            "cost": cost,
            "net_return": reward,
        }
        return PortfolioStepResult(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _resolve_weights(self, action: FloatArray) -> WeightVector:
        """Project a raw action score vector onto the (long-only) weight simplex.

        For ``long_only`` (the default) the exact Euclidean simplex projection
        guarantees a valid simplex (non-negative, sums to one) EVERY bar. For the
        long/short regime the action is recentred and L1-normalized to a unit budget.
        """
        arr = np.asarray(action, dtype="float64").ravel()
        if arr.size != self._n_assets:
            raise ValidationError(
                f"action has {arr.size} entries but the env expects {self._n_assets} assets."
            )
        if not bool(np.isfinite(arr).all()):
            raise ValidationError("action must be finite (no NaN/inf).")
        if self._config.long_only:
            return project_to_simplex(arr)
        # Long/short unit budget: recentre then L1-normalize (sum of |w| == 1). A
        # zero vector maps to equal-weight to keep the budget well defined.
        centred = arr - float(arr.mean())
        denom = float(np.abs(centred).sum())
        if denom <= 0.0:
            return np.full(self._n_assets, 1.0 / self._n_assets, dtype="float64")
        return (centred / denom).astype("float64")

    def _turnover_cost(self, turnover: float) -> float:
        """Return the per-bar turnover cost ``cost_bps / 1e4 * turnover`` (return units)."""
        return self._config.cost_bps / 10_000.0 * float(turnover)

    def _observe(self, t: int) -> ObservationVector:
        """Build the observation at decision bar ``t`` (data <= ``t`` only).

        The observation is the FLATTENED trailing look-back window of per-asset
        returns ``[r_{t-lookback+1}, ..., r_t]`` (row-major) concatenated with the
        current weight vector. It NEVER reads ``r_{t+1}`` or beyond, so it is
        strictly causal. The public ``reset`` / ``step`` path only ever asks for
        ``t >= lookback - 1``, so the window is always fully populated.
        """
        lookback = self._config.lookback
        lo = t - lookback + 1
        window = self._returns[lo : t + 1]
        return np.concatenate((window.astype("float64").ravel(), self._weights))

    def rollout(self, actions: FloatArray) -> FloatArray:
        """Replay a full weight-path action sequence and return per-bar net rewards.

        Drives the env from ``reset`` through one action vector per bar and collects
        the per-bar net reward series. This is the step-by-step ORACLE the vectorized
        backtester must reproduce to 1e-10 (the parity look-ahead guard). Each row of
        ``actions`` is projected onto the simplex exactly as :meth:`step` does.

        Parameters
        ----------
        actions:
            The per-bar action / target-weight sequence ``(n_bars, n_assets)``.

        Returns
        -------
        FloatArray
            The per-bar net reward series produced by the step-by-step rollout.

        Raises
        ------
        ValidationError
            If ``actions`` shape does not match the panel's bar/asset count.
        """
        acts = np.asarray(actions, dtype="float64")
        if acts.ndim == 1:
            acts = acts.reshape(-1, 1)
        if acts.shape != (self._n_bars, self._n_assets):
            raise ValidationError(
                f"actions shape {acts.shape} must match the return panel "
                f"({self._n_bars}, {self._n_assets}); the weights at t earn r_{{t+1}}."
            )
        if not bool(np.isfinite(acts).all()):
            raise ValidationError("actions must be finite (no NaN/inf).")
        # Step bar-by-bar over the scorable window t in [0, N-2]. The weights set at
        # bar t earn the NEXT bar's returns; cost is charged on the L1 change vs the
        # previous bar's weights (the book opens flat). This is the single
        # step-by-step oracle the vectorized backtester must reproduce.
        net = np.empty(self._n_scored, dtype="float64")
        prev_weights = np.zeros(self._n_assets, dtype="float64")
        for t in range(self._n_scored):
            weights = self._resolve_weights(acts[t])
            turnover = float(np.abs(weights - prev_weights).sum())
            net[t] = float(weights @ self._returns[t + 1]) - self._turnover_cost(turnover)
            prev_weights = weights
        return net

    def resolved_weight_path(self, actions: FloatArray) -> FloatArray:
        """Return the simplex-projected weight path for a raw action sequence.

        The companion to :meth:`rollout`: maps each raw action row through the same
        simplex projection :meth:`step` applies, yielding the ``(n_bars, n_assets)``
        weight path the vectorized backtester scores. A property test asserts every
        row of the result is a valid simplex.

        Parameters
        ----------
        actions:
            The per-bar raw action sequence ``(n_bars, n_assets)``.

        Returns
        -------
        FloatArray
            The simplex-projected weight path ``(n_bars, n_assets)``.

        Raises
        ------
        ValidationError
            If ``actions`` shape does not match the panel's bar/asset count.
        """
        acts = np.asarray(actions, dtype="float64")
        if acts.ndim == 1:
            acts = acts.reshape(-1, 1)
        if acts.shape != (self._n_bars, self._n_assets):
            raise ValidationError(
                f"actions shape {acts.shape} must match the return panel "
                f"({self._n_bars}, {self._n_assets})."
            )
        rows = [self._resolve_weights(acts[t]) for t in range(self._n_bars)]
        return np.asarray(rows, dtype="float64")

    def as_gym_env(self) -> Any:
        """Return a gymnasium-API wrapper around this env (LAZY ``gymnasium`` import).

        LAZY IMPORT: ``gymnasium`` (the ``[train]`` extra) is imported inside this
        method so importing :mod:`rlallocator.env.portfolio_env` never imports
        gymnasium. The wrapper exposes ``observation_space`` / ``action_space``
        (a Box over the per-asset score vector) and delegates ``reset`` / ``step``
        to this env for SB3 PPO training.

        Returns
        -------
        Any
            A ``gymnasium.Env`` instance suitable for SB3.

        Raises
        ------
        ImportError
            If the ``[train]`` extra (gymnasium) is not installed.
        """
        try:
            import gymnasium as gym
            from gymnasium import spaces
        except ImportError as exc:  # pragma: no cover - exercised only without [train]
            raise ImportError(
                "PortfolioEnv.as_gym_env requires the [train] extra (gymnasium). "
                "Install it with `uv pip install -e '.[train]'`."
            ) from exc

        # The gymnasium adapter below is exercised ONLY with the [train] extra
        # installed (gymnasium present); CI installs the lean extras, so this body is
        # marked no-cover (it is the offline-training-only surface).
        env = self  # pragma: no cover

        class _GymPortfolioEnv(gym.Env):  # type: ignore[misc]  # pragma: no cover
            """Thin gymnasium adapter delegating reset/step to the pure-numpy env."""

            metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

            def __init__(self) -> None:
                super().__init__()
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(env.obs_dim,), dtype=np.float64
                )
                # The action is a raw per-asset score vector projected to the simplex.
                self.action_space = spaces.Box(
                    low=-10.0, high=10.0, shape=(env.n_assets,), dtype=np.float64
                )

            def reset(
                self, *, seed: int | None = None, options: dict[str, Any] | None = None
            ) -> tuple[ObservationVector, dict[str, Any]]:
                super().reset(seed=seed)
                return env.reset(seed=seed)

            def step(
                self, action: Any
            ) -> tuple[ObservationVector, float, bool, bool, dict[str, Any]]:
                result = env.step(np.asarray(action, dtype="float64"))
                return (
                    result.observation,
                    result.reward,
                    result.terminated,
                    result.truncated,
                    result.info,
                )

        return _GymPortfolioEnv()  # pragma: no cover


def weights_are_simplex(weight_path: FloatArray, *, long_only: bool = True) -> bool:
    """Return ``True`` iff EVERY row of ``weight_path`` is a valid simplex.

    The convenience predicate the property suite uses to assert the env / backtester
    weight path is a valid simplex every bar.

    Parameters
    ----------
    weight_path:
        A ``(n_bars, n_assets)`` weight path.
    long_only:
        If ``True`` (default), require non-negative weights per bar.

    Returns
    -------
    bool
        ``True`` iff every bar's weights form a valid simplex.
    """
    w = np.atleast_2d(np.asarray(weight_path, dtype="float64"))
    return all(is_simplex(row, long_only=long_only) for row in w)
