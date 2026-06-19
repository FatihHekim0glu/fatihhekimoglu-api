"""Synthetic single-asset OHLC bar processes — the honest-null testbed + sanity fixtures.

Generates a single-asset daily OHLC BAR panel (open/high/low/close with realistic
INTRABAR structure) under three regimes, all seeded through
:func:`algosystem._rng.make_rng` so a given ``(seed, n_obs, ...)`` reproduces the
bars byte-for-byte:

- :func:`gbm_regime_bars` — a regime-switching GBM close process with a realistic
  intrabar high/low envelope and bid-ask-bounce microstructure where, BY
  CONSTRUCTION, the simple MA-crossover / momentum signals have NO exploitable
  edge net of costs. This is the deployed DEFAULT: the honest NULL holds.
- :func:`learnable_trend_bars` — a single persistent positive drift (a LEARNABLE,
  tradeable trend). The SANITY fixture: an MA-crossover that works SHOULD capture
  this trend and beat buy-and-hold net of costs, proving the machinery detects a
  real edge (so the null is honest, not vacuous).
- :func:`pure_noise_bars` — a driftless random walk (white-noise returns). The
  strict null: next-bar returns are provably unforecastable.

INTRABAR INVARIANTS every generated bar MUST satisfy (asserted by the author and
the property suite): ``low <= open <= high``, ``low <= close <= high``, and
``low <= high``, with all prices strictly positive. The signal reads ONLY the
closed ``close`` series; the ``open`` is the NEXT-bar fill price for the execution
engine.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from algosystem._exceptions import ValidationError
from algosystem._rng import make_rng
from algosystem._typing import FloatArray

#: Default number of bars for the shipped synthetic panel (mirrors the API default).
DEFAULT_N_OBS: int = 2000

#: Default number of latent regimes for the regime-switching GBM.
DEFAULT_N_REGIMES: int = 2

#: Anchor close level for every synthetic path (strictly positive by construction).
START_PRICE: float = 100.0

#: Probability of staying in the current latent regime each bar (sticky chain).
REGIME_STICKINESS: float = 0.98

#: Smallest permitted price level (keeps every bar strictly positive).
_MIN_PRICE: float = 1e-8


@dataclass(frozen=True, slots=True)
class BarPath:
    """Immutable synthetic single-asset OHLC bar panel + its known regime labels.

    Attributes
    ----------
    bars:
        The ``(n_obs, 4)`` OHLC DataFrame with columns
        ``["open", "high", "low", "close"]`` (strictly positive, intrabar
        invariants hold), indexed by business day.
    regime_labels:
        The ``(n_obs,)`` integer latent-regime label per bar (in time order); a
        single nominal regime for the non-regime processes.
    kind:
        The process kind (``"gbm_regime"`` / ``"learnable_trend"`` /
        ``"pure_noise"``).
    """

    bars: pd.DataFrame
    regime_labels: tuple[int, ...]
    kind: str

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this path's metadata.

        The full bar panel is omitted (it is large and not part of the API
        contract); only the shape metadata and regime labels are emitted.
        """
        return {
            "n_obs": int(self.bars.shape[0]),
            "kind": str(self.kind),
            "columns": [str(c) for c in self.bars.columns],
            "regime_labels": [int(x) for x in self.regime_labels],
        }


def _business_index(n_obs: int, start: str) -> pd.DatetimeIndex:
    """Return an ``n_obs``-length business-day (Mon-Fri) ``DatetimeIndex``."""
    return pd.bdate_range(start=start, periods=n_obs)


def _bars_from_log_returns(
    log_returns: FloatArray,
    gen: np.random.Generator,
    *,
    intrabar_range_bps: float,
    start: str,
) -> pd.DataFrame:
    """Build a strictly-positive OHLC bar panel from a close log-return vector.

    The close path is anchored at :data:`START_PRICE`; the open of each bar is the
    prior bar's close (a gapless open — the next-bar fill price the execution
    engine reads), and a seeded intrabar half-range envelope wraps each bar so the
    OHLC invariants ``low <= {open, close} <= high`` hold by construction.

    The intrabar envelope is drawn AFTER the close path so the random draws are in
    a fixed order, keeping the whole panel reproducible for a given generator.
    """
    returns = np.asarray(log_returns, dtype="float64").copy()
    returns[0] = 0.0  # anchor the first observation at START_PRICE.
    close = START_PRICE * np.exp(np.cumsum(returns))

    # Open = prior close (gapless); the first open equals the first close.
    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1]

    # Seeded intrabar half-range envelope around the bar's open/close extremes.
    half_range = (intrabar_range_bps / 10_000.0) * close
    span_hi = np.abs(gen.standard_normal(close.size)) * half_range
    span_lo = np.abs(gen.standard_normal(close.size)) * half_range
    bar_max = np.maximum(open_, close)
    bar_min = np.minimum(open_, close)
    high = bar_max + span_hi
    low = np.maximum(bar_min - span_lo, _MIN_PRICE)  # strictly positive.

    index = _business_index(close.size, start)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
        index=index,
    ).astype("float64")


def gbm_regime_bars(
    *,
    n_obs: int = DEFAULT_N_OBS,
    n_regimes: int = DEFAULT_N_REGIMES,
    seed: int = 7,
    base_drift: float = 0.0,
    base_vol: float = 0.01,
    intrabar_range_bps: float = 30.0,
    microstructure_bps: float = 2.0,
    start: str = "2010-01-01",
) -> BarPath:
    r"""Generate a regime-switching GBM OHLC bar panel (the honest-null DGP).

    The close log-return at bar ``t`` is :math:`\mu_{s_t} + \sigma_{s_t}\,
    \varepsilon_t` where the latent regime ``s_t`` follows a sticky Markov chain
    over ``n_regimes`` states (each with its own mild drift/vol), plus an additive
    mean-reverting bid-ask-bounce microstructure term of magnitude
    ``microstructure_bps``. Around each close, a seeded intrabar high/low envelope
    of magnitude ``intrabar_range_bps`` is drawn so the OHLC invariants
    (``low <= {open, close} <= high``) hold. BY CONSTRUCTION the per-bar drift is
    near-zero and the regime is unforecastable from the observable closed bars, so
    the simple signals have NO timing edge net of costs and the honest NULL holds.

    Parameters
    ----------
    n_obs:
        Number of daily bars (rows).
    n_regimes:
        Number of latent regimes in the sticky Markov chain (``>= 1``).
    seed:
        Master RNG seed (feeds :func:`algosystem._rng.make_rng`).
    base_drift:
        Centre of the per-regime daily close drift (kept near zero for the null).
    base_vol:
        Centre of the per-regime daily close volatility.
    intrabar_range_bps:
        Magnitude (bps) of the seeded intrabar high/low envelope around the close.
    microstructure_bps:
        Magnitude (bps) of the additive bid-ask-bounce microstructure noise.
    start:
        First business-day date for the index.

    Returns
    -------
    BarPath
        The OHLC bar panel + its known per-bar regime labels.

    Raises
    ------
    ValidationError
        If ``n_obs < 2``, ``n_regimes < 1``, or any volatility/range is negative.
    """
    if n_obs < 2:
        raise ValidationError(f"gbm_regime_bars: n_obs must be >= 2, got {n_obs}.")
    if n_regimes < 1:
        raise ValidationError(f"gbm_regime_bars: n_regimes must be >= 1, got {n_regimes}.")
    if base_vol < 0.0:
        raise ValidationError(f"gbm_regime_bars: base_vol must be >= 0, got {base_vol}.")
    if intrabar_range_bps < 0.0:
        raise ValidationError(
            f"gbm_regime_bars: intrabar_range_bps must be >= 0, got {intrabar_range_bps}."
        )
    if microstructure_bps < 0.0:
        raise ValidationError(
            f"gbm_regime_bars: microstructure_bps must be >= 0, got {microstructure_bps}."
        )

    gen = make_rng(seed)

    # Per-regime drift/vol: drifts centred near zero (honest null), vols fanned out
    # mildly around ``base_vol`` so the regimes are volatility states, not edges.
    spread = np.linspace(-1.0, 1.0, n_regimes) if n_regimes > 1 else np.zeros(1)
    regime_drift = base_drift + 0.0 * spread  # drift identical & near-zero per regime.
    regime_vol = base_vol * (1.0 + 0.5 * spread)
    regime_vol = np.maximum(regime_vol, 0.0)

    # Sticky Markov regime chain: stay with prob REGIME_STICKINESS, else jump to a
    # uniformly-chosen OTHER regime. Unforecastable from the observable closes.
    labels = np.empty(n_obs, dtype="int64")
    labels[0] = int(gen.integers(0, n_regimes))
    if n_regimes == 1:
        labels[:] = 0
    else:
        stay = gen.random(n_obs) < REGIME_STICKINESS
        jump = gen.integers(0, n_regimes - 1, size=n_obs)
        for t in range(1, n_obs):
            if stay[t]:
                labels[t] = labels[t - 1]
            else:
                # Map the [0, n_regimes-1) draw to an index that skips the current.
                candidate = int(jump[t])
                labels[t] = candidate if candidate < labels[t - 1] else candidate + 1

    # Diffusion term, regime-conditional; mean-reverting (AR(-1)) microstructure
    # bid-ask bounce that averages out so it adds no exploitable drift.
    eps = gen.standard_normal(n_obs)
    diffusion = regime_drift[labels] + regime_vol[labels] * eps
    micro_innov = (microstructure_bps / 10_000.0) * gen.standard_normal(n_obs)
    micro = micro_innov.copy()
    micro[1:] = micro_innov[1:] - micro_innov[:-1]  # first difference = bid-ask bounce.
    log_returns = (diffusion + micro).astype("float64")

    bars = _bars_from_log_returns(
        log_returns, gen, intrabar_range_bps=intrabar_range_bps, start=start
    )
    return BarPath(bars=bars, regime_labels=tuple(int(x) for x in labels), kind="gbm_regime")


def learnable_trend_bars(
    *,
    n_obs: int = DEFAULT_N_OBS,
    seed: int = 7,
    drift: float = 0.0008,
    vol: float = 0.01,
    intrabar_range_bps: float = 30.0,
    start: str = "2010-01-01",
) -> BarPath:
    r"""Generate an OHLC bar panel with a persistent positive drift (the LEARNABLE SANITY fixture).

    The close log-return at bar ``t`` is :math:`\mu + \sigma\,\varepsilon_t` with a
    CONSTANT positive drift ``mu = drift`` that dominates the noise on average, so
    the optimal position is persistently long and a fast/slow MA-crossover SHOULD
    capture the trend and beat buy-and-hold net of costs — the machinery-works
    sanity check (NOT the honest-null DGP). The intrabar high/low envelope is drawn
    exactly as in :func:`gbm_regime_bars` so the OHLC invariants hold.

    Parameters
    ----------
    n_obs:
        Number of daily bars.
    seed:
        Master RNG seed.
    drift:
        The constant per-bar log-drift (positive; learnable).
    vol:
        The per-bar close volatility.
    intrabar_range_bps:
        Magnitude (bps) of the seeded intrabar high/low envelope.
    start:
        First business-day date.

    Returns
    -------
    BarPath
        The trending OHLC bar panel with a single nominal regime label.

    Raises
    ------
    ValidationError
        If ``n_obs < 2`` or ``vol < 0``.
    """
    if n_obs < 2:
        raise ValidationError(f"learnable_trend_bars: n_obs must be >= 2, got {n_obs}.")
    if vol < 0.0:
        raise ValidationError(f"learnable_trend_bars: vol must be >= 0, got {vol}.")
    if intrabar_range_bps < 0.0:
        raise ValidationError(
            f"learnable_trend_bars: intrabar_range_bps must be >= 0, got {intrabar_range_bps}."
        )

    gen = make_rng(seed)
    log_returns = (drift + vol * gen.standard_normal(n_obs)).astype("float64")
    bars = _bars_from_log_returns(
        log_returns, gen, intrabar_range_bps=intrabar_range_bps, start=start
    )
    return BarPath(bars=bars, regime_labels=tuple([0] * n_obs), kind="learnable_trend")


def regime_trend_bars(
    *,
    n_obs: int = DEFAULT_N_OBS,
    seed: int = 7,
    block: int = 300,
    drift: float = 0.002,
    vol: float = 0.01,
    intrabar_range_bps: float = 30.0,
    start: str = "2010-01-01",
) -> BarPath:
    r"""Generate a DIRECTIONAL regime-trend OHLC bar panel (the tradeable SANITY fixture).

    The close log-return at bar ``t`` is :math:`s_t\,\mu + \sigma\,\varepsilon_t`
    where the sign ``s_t in {+1, -1}`` flips every ``block`` bars, so the path
    alternates between persistent UP-trend and persistent DOWN-trend regimes. Unlike
    the pure monotonic :func:`learnable_trend_bars` (on which buy-and-hold IS the
    optimal exposure and a long/short trend-follower can never win), this directional
    regime structure is one a long/short MA-crossover SHOULD beat buy-and-hold on,
    DM-significant net of costs: the crossover flips short through the down-trends
    that drag a static long position down. This is the SANITY DGP that proves the
    FULL long/short pipeline (signal -> backtest -> DM-vs-buy-hold) captures a real,
    tradeable edge — so the honest null on the no-edge DGPs is honest, not vacuous.

    The intrabar high/low envelope is drawn exactly as in :func:`gbm_regime_bars` so
    the OHLC invariants ``low <= {open, close} <= high`` hold by construction.

    Parameters
    ----------
    n_obs:
        Number of daily bars.
    seed:
        Master RNG seed.
    block:
        Bars per directional regime block (the trend persists this long before the
        sign flips); ``>= 1``.
    drift:
        The per-bar trend magnitude (positive; signed by the regime each block).
    vol:
        The per-bar close volatility.
    intrabar_range_bps:
        Magnitude (bps) of the seeded intrabar high/low envelope.
    start:
        First business-day date.

    Returns
    -------
    BarPath
        The directional regime-trend OHLC bar panel with per-bar regime labels
        (``0`` in an up block, ``1`` in a down block).

    Raises
    ------
    ValidationError
        If ``n_obs < 2``, ``block < 1``, or ``vol < 0``.
    """
    if n_obs < 2:
        raise ValidationError(f"regime_trend_bars: n_obs must be >= 2, got {n_obs}.")
    if block < 1:
        raise ValidationError(f"regime_trend_bars: block must be >= 1, got {block}.")
    if vol < 0.0:
        raise ValidationError(f"regime_trend_bars: vol must be >= 0, got {vol}.")
    if intrabar_range_bps < 0.0:
        raise ValidationError(
            f"regime_trend_bars: intrabar_range_bps must be >= 0, got {intrabar_range_bps}."
        )

    gen = make_rng(seed)
    # Alternating directional regime sign: + for the first block of bars, - for the
    # next, and so on. A long/short trend-follower SHOULD profit from flipping with
    # the sign, which a static buy-and-hold cannot do.
    block_idx = np.arange(n_obs) // block
    signs = np.where(block_idx % 2 == 0, 1.0, -1.0)
    log_returns = (signs * drift + vol * gen.standard_normal(n_obs)).astype("float64")
    bars = _bars_from_log_returns(
        log_returns, gen, intrabar_range_bps=intrabar_range_bps, start=start
    )
    labels = tuple(int(x % 2) for x in block_idx)
    return BarPath(bars=bars, regime_labels=labels, kind="regime_trend")


def pure_noise_bars(
    *,
    n_obs: int = DEFAULT_N_OBS,
    seed: int = 7,
    vol: float = 0.01,
    intrabar_range_bps: float = 30.0,
    start: str = "2010-01-01",
) -> BarPath:
    r"""Generate a driftless random-walk OHLC bar panel (the strict null).

    The close log-return at bar ``t`` is :math:`\sigma\,\varepsilon_t` with ZERO
    drift, so closes are a driftless geometric random walk and next-bar returns are
    provably unforecastable — the strictest honest-null testbed. The intrabar
    envelope is drawn as in :func:`gbm_regime_bars` so the OHLC invariants hold.

    Parameters
    ----------
    n_obs:
        Number of daily bars.
    seed:
        Master RNG seed.
    vol:
        The per-bar close volatility.
    intrabar_range_bps:
        Magnitude (bps) of the seeded intrabar high/low envelope.
    start:
        First business-day date.

    Returns
    -------
    BarPath
        The driftless OHLC bar panel with a single nominal regime label.

    Raises
    ------
    ValidationError
        If ``n_obs < 2`` or ``vol < 0``.
    """
    if n_obs < 2:
        raise ValidationError(f"pure_noise_bars: n_obs must be >= 2, got {n_obs}.")
    if vol < 0.0:
        raise ValidationError(f"pure_noise_bars: vol must be >= 0, got {vol}.")
    if intrabar_range_bps < 0.0:
        raise ValidationError(
            f"pure_noise_bars: intrabar_range_bps must be >= 0, got {intrabar_range_bps}."
        )

    gen = make_rng(seed)
    log_returns = (vol * gen.standard_normal(n_obs)).astype("float64")
    bars = _bars_from_log_returns(
        log_returns, gen, intrabar_range_bps=intrabar_range_bps, start=start
    )
    return BarPath(bars=bars, regime_labels=tuple([0] * n_obs), kind="pure_noise")


def assert_ohlc_invariants(bars: pd.DataFrame) -> None:
    """Assert the per-bar OHLC invariants on a bar panel.

    Every bar MUST satisfy ``low <= open <= high``, ``low <= close <= high``, and
    all prices strictly positive — the intrabar consistency the synthetic
    generators guarantee and the property suite enforces.

    Parameters
    ----------
    bars:
        An OHLC DataFrame with columns ``["open", "high", "low", "close"]``.

    Raises
    ------
    ValidationError
        If a required column is missing or any intrabar invariant is violated.
    """
    if not {"open", "high", "low", "close"}.issubset(set(bars.columns)):
        raise ValidationError(
            "assert_ohlc_invariants: bars must have open/high/low/close columns."
        )

    open_ = bars["open"].to_numpy(dtype="float64")
    high = bars["high"].to_numpy(dtype="float64")
    low = bars["low"].to_numpy(dtype="float64")
    close = bars["close"].to_numpy(dtype="float64")

    if not bool(np.all(np.isfinite(np.concatenate([open_, high, low, close])))):
        raise ValidationError("assert_ohlc_invariants: bars contain non-finite prices.")
    if not bool(np.all((open_ > 0.0) & (high > 0.0) & (low > 0.0) & (close > 0.0))):
        raise ValidationError("assert_ohlc_invariants: all prices must be strictly positive.")
    if not bool(np.all(high >= low)):
        raise ValidationError("assert_ohlc_invariants: high must be >= low for every bar.")
    if not bool(np.all((high >= open_) & (high >= close))):
        raise ValidationError(
            "assert_ohlc_invariants: high must be >= open and >= close for every bar."
        )
    if not bool(np.all((low <= open_) & (low <= close))):
        raise ValidationError(
            "assert_ohlc_invariants: low must be <= open and <= close for every bar."
        )
