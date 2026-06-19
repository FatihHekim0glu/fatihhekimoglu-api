"""Synthetic single-asset price processes — the honest-null testbed + sanity fixtures.

Generates a single-asset daily price path under three regimes, all seeded through
:func:`rltrader._rng.make_rng` so a given ``(seed, n_obs, ...)`` reproduces the
path byte-for-byte:

- :func:`gbm_regime_path` — geometric Brownian motion with mild regime-switching
  drift/volatility plus microstructure (bid-ask bounce) noise where, BY
  CONSTRUCTION, there is NO exploitable timing edge net of costs. This is the
  deployed DEFAULT: the honest NULL holds (a PPO agent cannot reliably beat
  buy-and-hold after costs).
- :func:`pure_trend_path` — a single persistent positive drift (a LEARNABLE trend).
  The SANITY fixture: an agent that works SHOULD capture this trend and beat flat
  cash, proving the env + training machinery actually learn.
- :func:`pure_noise_path` — a driftless random walk (white-noise returns). The
  strict null: nothing is forecastable, so ranking by signal is provably random.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from rltrader._exceptions import ValidationError
from rltrader._rng import make_rng

#: Default number of bars for the shipped synthetic path (mirrors the API default).
DEFAULT_N_OBS: int = 2000

#: Default number of latent regimes for the regime-switching GBM.
DEFAULT_N_REGIMES: int = 2

#: Anchor price level for every synthetic path (strictly positive by construction).
_START_PRICE: float = 100.0

#: Probability of staying in the current latent regime each bar (sticky chain).
_REGIME_STICKINESS: float = 0.98


@dataclass(frozen=True, slots=True)
class PricePath:
    """Immutable synthetic single-asset price path + its known regime labels.

    Attributes
    ----------
    prices:
        The ``(n_obs,)`` single-asset price level Series (strictly positive),
        indexed by business day.
    regime_labels:
        The ``(n_obs,)`` integer latent-regime label per bar (in time order); a
        single nominal regime for the non-regime processes.
    kind:
        The process kind (``"gbm_regime"`` / ``"pure_trend"`` / ``"pure_noise"``).
    """

    prices: pd.Series
    regime_labels: tuple[int, ...]
    kind: str

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this path's metadata.

        The full price vector is omitted (it is large and not part of the API
        contract); only the shape metadata and regime labels are emitted.
        """
        return {
            "n_obs": int(self.prices.shape[0]),
            "kind": str(self.kind),
            "regime_labels": [int(x) for x in self.regime_labels],
        }


def gbm_regime_path(
    *,
    n_obs: int = DEFAULT_N_OBS,
    n_regimes: int = DEFAULT_N_REGIMES,
    seed: int = 7,
    base_drift: float = 0.0,
    base_vol: float = 0.01,
    microstructure_bps: float = 2.0,
    start: str = "2010-01-01",
) -> PricePath:
    r"""Generate a regime-switching GBM price path with microstructure noise (honest-null DGP).

    The log-return at bar ``t`` is :math:`\mu_{s_t} + \sigma_{s_t}\,\varepsilon_t`
    where the latent regime ``s_t`` follows a sticky Markov chain over
    ``n_regimes`` states (each with its own mild drift/vol), plus an additive
    mean-reverting microstructure (bid-ask bounce) noise term of magnitude
    ``microstructure_bps``. BY CONSTRUCTION the per-bar drift is near-zero and the
    regime is unforecastable from the observable look-back, so there is NO timing
    edge net of costs and the honest NULL holds.

    Parameters
    ----------
    n_obs:
        Number of daily bars (rows).
    n_regimes:
        Number of latent regimes in the sticky Markov chain (``>= 1``).
    seed:
        Master RNG seed (feeds :func:`rltrader._rng.make_rng`).
    base_drift:
        Centre of the per-regime daily drift (kept near zero for the null).
    base_vol:
        Centre of the per-regime daily volatility.
    microstructure_bps:
        Magnitude (bps) of the additive bid-ask-bounce microstructure noise.
    start:
        First business-day date for the index.

    Returns
    -------
    PricePath
        The price path + its known per-bar regime labels.

    Raises
    ------
    ValidationError
        If ``n_obs < 2``, ``n_regimes < 1``, or any volatility is negative.

    """
    _validate_path_args(n_obs, base_vol, name="gbm_regime_path")
    if n_regimes < 1:
        raise ValidationError(f"gbm_regime_path: n_regimes must be >= 1, got {n_regimes}.")

    rng = make_rng(seed)
    index = _node_index(n_obs, start)

    # Regimes drive VOLATILITY only (vol clustering — the empirically dominant
    # regime effect). The drift is held at ``base_drift`` (near zero) in EVERY
    # regime, so no regime carries a forecastable directional edge and the path is
    # driftless on average: the honest NULL. The Itô correction ``-0.5 * vol**2``
    # keeps the per-regime SIMPLE-return mean at ~``base_drift`` (so the level has
    # no upward variance-drag bias that could masquerade as a learnable trend).
    offsets = np.linspace(-1.0, 1.0, num=n_regimes) if n_regimes > 1 else np.zeros(1)
    regime_vols = base_vol * (1.0 + 0.5 * offsets)

    labels = _simulate_regimes(rng, n_obs, n_regimes)
    vol_t = regime_vols[labels]
    drift_t = base_drift - 0.5 * vol_t**2
    diffusion = vol_t * rng.standard_normal(n_obs)

    # Mean-reverting bid-ask-bounce microstructure: the first difference of an
    # i.i.d. innovation is negatively autocorrelated (a transient that reverses)
    # and contributes ZERO net drift to the cumulative path, so it adds no timing
    # edge — only round-trip cost when an agent tries to trade it.
    micro_scale = microstructure_bps / 10_000.0
    micro_innov = micro_scale * rng.standard_normal(n_obs + 1)
    microstructure = np.diff(micro_innov)

    log_returns = drift_t + diffusion + microstructure
    prices = _prices_from_log_returns(log_returns, index)
    return PricePath(prices=prices, regime_labels=tuple(int(x) for x in labels), kind="gbm_regime")


def pure_trend_path(
    *,
    n_obs: int = DEFAULT_N_OBS,
    seed: int = 7,
    drift: float = 0.0008,
    vol: float = 0.01,
    start: str = "2010-01-01",
) -> PricePath:
    r"""Generate a single-asset path with a persistent positive drift (the LEARNABLE SANITY fixture).

    The log-return at bar ``t`` is :math:`\mu + \sigma\,\varepsilon_t` with a
    CONSTANT positive drift ``mu = drift`` that dominates the noise on average, so
    the optimal policy is to be persistently long. An agent whose env + training
    actually work SHOULD capture this trend and beat flat cash net of costs —
    this is the machinery-works sanity check (NOT the honest-null DGP).

    Parameters
    ----------
    n_obs:
        Number of daily bars.
    seed:
        Master RNG seed.
    drift:
        The constant per-bar log-drift (positive; learnable).
    vol:
        The per-bar volatility.
    start:
        First business-day date.

    Returns
    -------
    PricePath
        The trending price path with a single nominal regime label.

    Raises
    ------
    ValidationError
        If ``n_obs < 2`` or ``vol < 0``.

    """
    _validate_path_args(n_obs, vol, name="pure_trend_path")
    rng = make_rng(seed)
    index = _node_index(n_obs, start)
    log_returns = float(drift) + float(vol) * rng.standard_normal(n_obs)
    prices = _prices_from_log_returns(log_returns, index)
    return PricePath(prices=prices, regime_labels=(0,) * n_obs, kind="pure_trend")


def pure_noise_path(
    *,
    n_obs: int = DEFAULT_N_OBS,
    seed: int = 7,
    vol: float = 0.01,
    start: str = "2010-01-01",
) -> PricePath:
    r"""Generate a driftless random-walk price path (the strict null).

    The log-return at bar ``t`` is :math:`\sigma\,\varepsilon_t` with ZERO drift,
    so prices are a driftless geometric random walk and next-bar returns are
    provably unforecastable — the strictest honest-null testbed, driving the
    anti-overfit regression (the agent must NOT beat buy-and-hold).

    Parameters
    ----------
    n_obs:
        Number of daily bars.
    seed:
        Master RNG seed.
    vol:
        The per-bar volatility.
    start:
        First business-day date.

    Returns
    -------
    PricePath
        The driftless price path with a single nominal regime label.

    Raises
    ------
    ValidationError
        If ``n_obs < 2`` or ``vol < 0``.

    """
    _validate_path_args(n_obs, vol, name="pure_noise_path")
    rng = make_rng(seed)
    index = _node_index(n_obs, start)
    log_returns = float(vol) * rng.standard_normal(n_obs)
    prices = _prices_from_log_returns(log_returns, index)
    return PricePath(prices=prices, regime_labels=(0,) * n_obs, kind="pure_noise")


def _node_index(n_obs: int, start: str) -> pd.DatetimeIndex:
    """Return an ``n_obs``-length business-day (Mon-Fri) ``DatetimeIndex``.

    Shared helper for every synthetic process so the three DGPs share an identical
    calendar. Importing/calling it touches no global RNG state.
    """
    return pd.bdate_range(start=start, periods=n_obs)


def _validate_path_args(n_obs: int, vol: float, *, name: str) -> None:
    """Validate the shared ``(n_obs, vol)`` preconditions of every synthetic DGP.

    Raises
    ------
    ValidationError
        If ``n_obs < 2`` (no causal observation/reward step can be formed) or
        ``vol`` is negative / non-finite.
    """
    if n_obs < 2:
        raise ValidationError(f"{name}: n_obs must be >= 2, got {n_obs}.")
    vol_f = float(vol)
    if not np.isfinite(vol_f) or vol_f < 0.0:
        raise ValidationError(f"{name}: vol must be finite and non-negative, got {vol!r}.")


def _prices_from_log_returns(
    log_returns: np.ndarray,
    index: pd.DatetimeIndex,
) -> pd.Series:
    """Build a strictly-positive price path from a log-return vector.

    The first log-return is pinned to ``0.0`` so the path is anchored exactly at
    :data:`_START_PRICE`; the level is ``start * exp(cumsum(log_returns))`` and is
    therefore strictly positive for any finite returns.
    """
    log_returns = np.asarray(log_returns, dtype="float64").copy()
    log_returns[0] = 0.0  # anchor the first observation at the start price.
    prices = _START_PRICE * np.exp(np.cumsum(log_returns))
    return pd.Series(prices, index=index, dtype="float64", name="price")


def _simulate_regimes(
    rng: np.random.Generator,
    n_obs: int,
    n_regimes: int,
) -> np.ndarray:
    """Simulate a sticky-Markov-chain latent-regime label per bar.

    Each bar stays in the current regime with probability :data:`_REGIME_STICKINESS`
    and otherwise jumps to a uniformly-chosen *other* regime, so regimes persist in
    long runs (the empirically realistic shape) yet the chain is unforecastable from
    the observable look-back. With ``n_regimes == 1`` the labels are all zero.
    """
    labels = np.zeros(n_obs, dtype="int64")
    if n_regimes == 1:
        return labels
    stay = rng.random(n_obs) < _REGIME_STICKINESS
    jumps = rng.integers(1, n_regimes, size=n_obs)  # 1..n_regimes-1 (a non-zero offset).
    current = 0
    for t in range(n_obs):
        if t > 0 and not stay[t]:
            current = int((current + jumps[t]) % n_regimes)
        labels[t] = current
    return labels
