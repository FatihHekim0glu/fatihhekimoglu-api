"""Synthetic multi-asset return panels — the honest-null testbed + sanity fixtures.

Generates a multi-asset daily return PANEL under three regimes, all seeded through
:func:`rlallocator._rng.make_rng` so a given ``(seed, n_obs, n_assets, ...)``
reproduces the panel byte-for-byte:

- :func:`factor_regime_panel` — a linear factor model (a market factor + a handful of
  style factors) with REGIME-SWITCHING factor correlations / volatilities plus
  per-asset idiosyncratic noise, where BY CONSTRUCTION every asset has the SAME
  risk-adjusted expected return so no static or dynamic allocation reliably beats 1/N
  net of turnover costs. This is the deployed DEFAULT: the honest NULL holds.
- :func:`learnable_edge_panel` — the same factor structure but ONE asset carries a
  persistent positive risk-adjusted PREMIUM (a higher Sharpe than its peers). The
  SANITY fixture: an allocator that works SHOULD tilt toward the premium asset and
  beat 1/N — proving the env + training machinery actually learn, so the null is
  honest, not vacuous.
- :func:`pure_noise_panel` — driftless i.i.d. cross-sectional white-noise returns
  (no factor, no premium). The strict null: nothing is forecastable.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from rlallocator._exceptions import ValidationError
from rlallocator._rng import make_rng

# quantcore-candidate: mirrors rl-trader:data/synthetic.py, GENERALIZED from a
# single-asset GBM to a multi-asset factor + regime-switching-correlation panel.

#: Default number of bars for the shipped synthetic panel.
DEFAULT_N_OBS: int = 2000

#: Default number of assets in the basket (mirrors the API default).
DEFAULT_N_ASSETS: int = 6

#: Default number of latent factors (a market factor + style factors).
DEFAULT_N_FACTORS: int = 3

#: Default number of latent volatility/correlation regimes.
DEFAULT_N_REGIMES: int = 2

#: Probability of staying in the current latent regime each bar (sticky chain).
_REGIME_STICKINESS: float = 0.98


@dataclass(frozen=True, slots=True)
class ReturnPanelData:
    """Immutable synthetic multi-asset return panel + its known regime labels.

    Attributes
    ----------
    returns:
        The ``(n_obs, n_assets)`` per-bar return panel, indexed by business day,
        columns ``asset_0 .. asset_{N-1}``.
    regime_labels:
        The ``(n_obs,)`` integer latent-regime label per bar (in time order).
    kind:
        The panel kind (``"factor_regime"`` / ``"learnable_edge"`` / ``"pure_noise"``).
    premium_asset:
        The index of the premium-bearing asset for ``"learnable_edge"`` (``-1`` for
        the null panels, which carry no edge).
    """

    returns: pd.DataFrame
    regime_labels: tuple[int, ...]
    kind: str
    premium_asset: int = -1

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this panel's metadata.

        The full return matrix is omitted (it is large and not part of the API
        contract); only the shape metadata, regime labels, and premium index emit.
        """
        return {
            "n_obs": int(self.returns.shape[0]),
            "n_assets": int(self.returns.shape[1]),
            "kind": str(self.kind),
            "premium_asset": int(self.premium_asset),
            "regime_labels": [int(x) for x in self.regime_labels],
        }


def factor_regime_panel(
    *,
    n_obs: int = DEFAULT_N_OBS,
    n_assets: int = DEFAULT_N_ASSETS,
    n_factors: int = DEFAULT_N_FACTORS,
    n_regimes: int = DEFAULT_N_REGIMES,
    seed: int = 7,
    base_vol: float = 0.01,
    idio_vol: float = 0.006,
    start: str = "2010-01-01",
) -> ReturnPanelData:
    r"""Generate a factor + regime-switching-correlation multi-asset panel (honest-null DGP).

    Each asset's return is :math:`r_{i,t} = \beta_i^\top f_t + \varepsilon_{i,t}`
    where the common factors ``f_t`` have a REGIME-SWITCHING covariance (the latent
    regime ``s_t`` follows a sticky Markov chain that scales factor volatility) and
    ``epsilon`` is per-asset idiosyncratic noise. BY CONSTRUCTION every asset has
    near-ZERO drift and the SAME unconditional risk-adjusted return, so no static or
    dynamic re-weighting reliably beats 1/N net of turnover costs — the honest NULL
    holds. The regime drives only VOLATILITY/correlation (the empirically dominant,
    unforecastable-from-the-look-back effect), carrying no directional edge.

    Parameters
    ----------
    n_obs:
        Number of daily bars (rows).
    n_assets:
        Number of assets (columns; ``>= 2``).
    n_factors:
        Number of common factors (``>= 1``).
    n_regimes:
        Number of latent regimes in the sticky Markov chain (``>= 1``).
    seed:
        Master RNG seed (feeds :func:`rlallocator._rng.make_rng`).
    base_vol:
        Centre of the per-bar factor volatility.
    idio_vol:
        Per-asset idiosyncratic volatility.
    start:
        First business-day date for the index.

    Returns
    -------
    ReturnPanelData
        The return panel + its known per-bar regime labels (premium_asset ``-1``).

    Raises
    ------
    ValidationError
        If any dimension is below its minimum or a volatility is negative.
    """
    _validate_panel_args(
        n_obs=n_obs,
        n_assets=n_assets,
        n_factors=n_factors,
        n_regimes=n_regimes,
        vols=(base_vol, idio_vol),
        name="factor_regime_panel",
    )
    rng = make_rng(seed)
    labels = _simulate_regimes(rng, n_obs, n_regimes)
    factors, betas = _factor_block(
        rng,
        n_obs=n_obs,
        n_assets=n_assets,
        n_factors=n_factors,
        labels=labels,
        n_regimes=n_regimes,
        base_vol=base_vol,
    )
    idio = idio_vol * rng.standard_normal((n_obs, n_assets))
    # No directional drift: every asset shares the same (zero) unconditional mean,
    # so the honest NULL holds — no re-weighting beats 1/N net of costs.
    returns = factors @ betas.T + idio
    frame = _panel_frame(returns, n_obs=n_obs, start=start)
    return ReturnPanelData(
        returns=frame,
        regime_labels=tuple(int(x) for x in labels),
        kind="factor_regime",
        premium_asset=-1,
    )


def learnable_edge_panel(
    *,
    n_obs: int = DEFAULT_N_OBS,
    n_assets: int = DEFAULT_N_ASSETS,
    n_factors: int = DEFAULT_N_FACTORS,
    n_regimes: int = DEFAULT_N_REGIMES,
    seed: int = 7,
    base_vol: float = 0.01,
    idio_vol: float = 0.006,
    premium: float = 0.0010,
    premium_asset: int = 0,
    start: str = "2010-01-01",
) -> ReturnPanelData:
    r"""Generate a factor panel where ONE asset carries a persistent premium (the SANITY fixture).

    Identical factor + regime structure to :func:`factor_regime_panel`, but the asset
    at index ``premium_asset`` is given a CONSTANT positive per-bar drift ``premium``
    on top of its factor + idiosyncratic return — a higher risk-adjusted return
    (Sharpe) than its peers. The optimal allocator OVER-weights that asset; an
    allocator whose env + training actually work SHOULD tilt toward it and beat 1/N
    net of costs. This is the machinery-works, anti-vacuous-null sanity check (NOT the
    honest-null DGP).

    Parameters
    ----------
    n_obs:
        Number of daily bars.
    n_assets:
        Number of assets (``>= 2``).
    n_factors:
        Number of common factors (``>= 1``).
    n_regimes:
        Number of latent regimes (``>= 1``).
    seed:
        Master RNG seed.
    base_vol:
        Centre of the per-bar factor volatility.
    idio_vol:
        Per-asset idiosyncratic volatility.
    premium:
        The constant per-bar excess drift added to the premium asset (positive).
    premium_asset:
        The index of the premium-bearing asset (``0 <= premium_asset < n_assets``).
    start:
        First business-day date.

    Returns
    -------
    ReturnPanelData
        The return panel with a single premium asset (``premium_asset`` recorded).

    Raises
    ------
    ValidationError
        If any dimension is below its minimum, a volatility is negative, or
        ``premium_asset`` is out of range.
    """
    _validate_panel_args(
        n_obs=n_obs,
        n_assets=n_assets,
        n_factors=n_factors,
        n_regimes=n_regimes,
        vols=(base_vol, idio_vol),
        name="learnable_edge_panel",
    )
    if not 0 <= premium_asset < n_assets:
        raise ValidationError(
            f"learnable_edge_panel: premium_asset must be in [0, {n_assets}), got {premium_asset}."
        )
    rng = make_rng(seed)
    labels = _simulate_regimes(rng, n_obs, n_regimes)
    factors, betas = _factor_block(
        rng,
        n_obs=n_obs,
        n_assets=n_assets,
        n_factors=n_factors,
        labels=labels,
        n_regimes=n_regimes,
        base_vol=base_vol,
    )
    idio = idio_vol * rng.standard_normal((n_obs, n_assets))
    returns = factors @ betas.T + idio
    # Persistent positive drift on the premium asset only (a learnable edge).
    returns[:, premium_asset] += float(premium)
    frame = _panel_frame(returns, n_obs=n_obs, start=start)
    return ReturnPanelData(
        returns=frame,
        regime_labels=tuple(int(x) for x in labels),
        kind="learnable_edge",
        premium_asset=int(premium_asset),
    )


def pure_noise_panel(
    *,
    n_obs: int = DEFAULT_N_OBS,
    n_assets: int = DEFAULT_N_ASSETS,
    seed: int = 7,
    vol: float = 0.01,
    start: str = "2010-01-01",
) -> ReturnPanelData:
    r"""Generate a driftless i.i.d. cross-sectional white-noise panel (the strict null).

    Each asset's return is :math:`\sigma\,\varepsilon_{i,t}` with ZERO drift and no
    common factor, so cross-sectional and time-series structure are both absent —
    nothing is forecastable. The strictest honest-null testbed, driving the
    anti-overfit regression (the allocator must NOT beat 1/N).

    Parameters
    ----------
    n_obs:
        Number of daily bars.
    n_assets:
        Number of assets (``>= 2``).
    seed:
        Master RNG seed.
    vol:
        The per-bar per-asset volatility.
    start:
        First business-day date.

    Returns
    -------
    ReturnPanelData
        The driftless white-noise panel with a single nominal regime label.

    Raises
    ------
    ValidationError
        If ``n_obs < 2``, ``n_assets < 2``, or ``vol < 0``.
    """
    _validate_panel_args(
        n_obs=n_obs,
        n_assets=n_assets,
        n_factors=1,
        n_regimes=1,
        vols=(vol,),
        name="pure_noise_panel",
    )
    rng = make_rng(seed)
    returns = float(vol) * rng.standard_normal((n_obs, n_assets))
    frame = _panel_frame(returns, n_obs=n_obs, start=start)
    return ReturnPanelData(
        returns=frame,
        regime_labels=(0,) * n_obs,
        kind="pure_noise",
        premium_asset=-1,
    )


def _factor_block(
    rng: np.random.Generator,
    *,
    n_obs: int,
    n_assets: int,
    n_factors: int,
    labels: np.ndarray,
    n_regimes: int,
    base_vol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(factors, betas)``: regime-scaled factor returns + per-asset loadings.

    The factor returns are i.i.d. standard normals scaled by a per-regime volatility
    multiplier (the regime-switching correlation/vol effect — regimes scale the
    common factor volatility, which co-moves the assets). The loadings ``betas`` are
    fixed positive exposures so the assets share a common (market) factor; the first
    factor is a broad market factor (all assets load on it) and the rest are style
    factors with mixed-sign loadings.
    """
    # Per-regime factor-volatility multiplier (regimes scale common volatility/co-movement).
    offsets = np.linspace(-1.0, 1.0, num=n_regimes) if n_regimes > 1 else np.zeros(1)
    regime_mult = 1.0 + 0.5 * offsets  # (n_regimes,)
    vol_t = base_vol * regime_mult[labels]  # (n_obs,)
    raw = rng.standard_normal((n_obs, n_factors))
    factors = raw * vol_t[:, None]
    # Loadings: a broad market factor (column 0, all positive) + style factors.
    betas = rng.uniform(0.3, 1.0, size=(n_assets, n_factors))
    if n_factors > 1:
        betas[:, 1:] = rng.uniform(-0.6, 0.6, size=(n_assets, n_factors - 1))
    return factors, betas


def _panel_frame(returns: np.ndarray, *, n_obs: int, start: str) -> pd.DataFrame:
    """Wrap a return matrix in a business-day-indexed, asset-columned DataFrame."""
    index = pd.bdate_range(start=start, periods=n_obs)
    columns = [f"asset_{i}" for i in range(returns.shape[1])]
    return pd.DataFrame(np.asarray(returns, dtype="float64"), index=index, columns=columns)


def _validate_panel_args(
    *,
    n_obs: int,
    n_assets: int,
    n_factors: int,
    n_regimes: int,
    vols: tuple[float, ...],
    name: str,
) -> None:
    """Validate the shared dimension + volatility preconditions of every synthetic DGP.

    Raises
    ------
    ValidationError
        If ``n_obs < 2``, ``n_assets < 2``, ``n_factors < 1``, ``n_regimes < 1``, or
        any volatility is negative / non-finite.
    """
    if n_obs < 2:
        raise ValidationError(f"{name}: n_obs must be >= 2, got {n_obs}.")
    if n_assets < 2:
        raise ValidationError(f"{name}: n_assets must be >= 2, got {n_assets}.")
    if n_factors < 1:
        raise ValidationError(f"{name}: n_factors must be >= 1, got {n_factors}.")
    if n_regimes < 1:
        raise ValidationError(f"{name}: n_regimes must be >= 1, got {n_regimes}.")
    for vol in vols:
        vol_f = float(vol)
        if not np.isfinite(vol_f) or vol_f < 0.0:
            raise ValidationError(f"{name}: vol must be finite and non-negative, got {vol!r}.")


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
