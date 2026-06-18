"""Synthetic multivariate RETURNS generator — the honest-null testbed.

Generates a small basket of correlated daily-return series from a transparent,
seeded data-generating process where, BY CONSTRUCTION, a deep model cannot
reliably beat the naive random-walk baseline:

- a single weak COMMON FACTOR (cross-sectional correlation) driving all series;
- per-series IDIOSYNCRATIC Gaussian noise (the dominant, unpredictable term);
- optional MILD AR(1) autocorrelation (a tiny, economically-negligible amount of
  structure) so the panel is not literally i.i.d. but the predictable component is
  far below the noise floor.

Because the conditional mean is dominated by unforecastable noise, the next-step
return is effectively a random walk: the naive ``r_hat = 0`` forecast is the OOS
floor and any deep edge is statistically indistinguishable (DM insignificant,
DSR ~ 0). This is the deliberate testbed for the honest NULL.

All randomness flows through :func:`mvtsforecast._rng.make_rng`, so a given
``(seed, basket, n_obs, ...)`` reproduces the panel byte-for-byte. Importing this
module has no side effects.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from mvtsforecast._exceptions import ValidationError
from mvtsforecast._rng import make_rng

#: Default basket for the shipped synthetic panel (mirrors the API default).
DEFAULT_BASKET: tuple[str, ...] = ("SPY", "TLT", "GLD")


def _business_index(n_obs: int, start: str = "2015-01-01") -> pd.DatetimeIndex:
    """Return an ``n_obs``-length business-day (Mon-Fri) ``DatetimeIndex``."""
    return pd.bdate_range(start=start, periods=n_obs)


def synthetic_panel(
    basket: Sequence[str] = DEFAULT_BASKET,
    *,
    n_obs: int = 1500,
    seed: int = 7,
    factor_vol: float = 0.004,
    idio_vol: float = 0.010,
    ar1: float = 0.02,
    start: str = "2015-01-01",
) -> pd.DataFrame:
    r"""Generate a seeded multivariate daily-RETURNS panel (the honest-null DGP).

    Each series is

    .. math::

        r_{t,i} = \phi\, r_{t-1,i} + \beta_i f_t + \varepsilon_{t,i},
        \quad f_t \sim \mathcal{N}(0, \sigma_f^2),\;
        \varepsilon_{t,i} \sim \mathcal{N}(0, \sigma_\varepsilon^2),

    with a weak common factor :math:`f_t`, per-series loadings :math:`\beta_i`, a
    small AR(1) coefficient :math:`\phi` = ``ar1``, and a DOMINANT idiosyncratic
    noise term. Because :math:`\sigma_\varepsilon \gg` the predictable component,
    the next-step conditional mean is negligible and the naive forecast is the OOS
    floor — the honest NULL holds by construction.

    Parameters
    ----------
    basket:
        Column labels (asset tickers) for the generated panel.
    n_obs:
        Number of daily observations (rows).
    seed:
        Master RNG seed (feeds :func:`mvtsforecast._rng.make_rng`).
    factor_vol:
        Standard deviation of the common factor :math:`f_t`.
    idio_vol:
        Standard deviation of the idiosyncratic noise :math:`\varepsilon_{t,i}`.
    ar1:
        AR(1) autocorrelation coefficient :math:`\phi` (kept small).
    start:
        First business-day date for the index.

    Returns
    -------
    pandas.DataFrame
        A ``(n_obs, len(basket))`` float64 returns panel indexed by business day.

    Raises
    ------
    ValidationError
        If ``basket`` is empty or has duplicates, ``n_obs < 2``, ``|ar1| >= 1``,
        or any volatility is negative.
    """
    names = list(basket)
    if len(names) == 0:
        raise ValidationError("synthetic_panel: basket must be non-empty.")
    if len(set(names)) != len(names):
        raise ValidationError("synthetic_panel: basket must not contain duplicates.")
    if n_obs < 2:
        raise ValidationError(f"synthetic_panel: n_obs must be >= 2, got {n_obs}.")
    if abs(ar1) >= 1.0:
        raise ValidationError(f"synthetic_panel: |ar1| must be < 1, got {ar1}.")
    if factor_vol < 0.0 or idio_vol < 0.0:
        raise ValidationError("synthetic_panel: volatilities must be non-negative.")

    n_assets = len(names)
    gen = make_rng(seed)

    # Per-series factor loadings (centred near one, mildly dispersed).
    betas = gen.uniform(0.6, 1.4, size=n_assets)
    factor = gen.standard_normal(n_obs) * factor_vol
    idio = gen.standard_normal((n_obs, n_assets)) * idio_vol

    # Innovations = common factor (broadcast through loadings) + idiosyncratic.
    innovations = np.outer(factor, betas) + idio

    # Mild AR(1) recursion so the panel is not literally i.i.d.; the predictable
    # component stays far below the noise floor.
    returns = np.empty_like(innovations)
    returns[0, :] = innovations[0, :]
    for t in range(1, n_obs):
        returns[t, :] = ar1 * returns[t - 1, :] + innovations[t, :]

    index = _business_index(n_obs, start=start)
    return pd.DataFrame(returns, index=index, columns=names, dtype="float64")


def weak_factor_panel(
    basket: Sequence[str] = DEFAULT_BASKET,
    *,
    n_obs: int = 1500,
    seed: int = 7,
    start: str = "2015-01-01",
) -> pd.DataFrame:
    """Convenience: a synthetic panel with a deliberately WEAK common factor.

    A thin wrapper over :func:`synthetic_panel` with a low ``factor_vol`` relative
    to ``idio_vol`` so cross-sectional correlation is present but small — the
    canonical "weak factor" fixture used to show that even a real (but tiny)
    common signal does not let a deep model beat naive after costs.

    Parameters
    ----------
    basket:
        Column labels for the panel.
    n_obs:
        Number of observations.
    seed:
        Master RNG seed.
    start:
        First business-day date.

    Returns
    -------
    pandas.DataFrame
        The weak-factor returns panel.

    Raises
    ------
    ValidationError
        Propagated from :func:`synthetic_panel`.
    """
    # A small factor relative to the (default) idiosyncratic noise: cross-sectional
    # correlation is present but weak, so any common signal is far below the noise
    # floor and a deep model still cannot beat naive after costs.
    return synthetic_panel(
        basket,
        n_obs=n_obs,
        seed=seed,
        factor_vol=0.0015,
        idio_vol=0.010,
        ar1=0.0,
        start=start,
    )


def random_walk_panel(
    basket: Sequence[str] = DEFAULT_BASKET,
    *,
    n_obs: int = 1500,
    seed: int = 7,
    idio_vol: float = 0.010,
    start: str = "2015-01-01",
) -> pd.DataFrame:
    """A pure random-walk returns panel (no factor, no autocorrelation).

    Each series is i.i.d. Gaussian noise — the strictest honest-null testbed,
    where the next-step return is genuinely unpredictable and the naive forecast
    is provably the OOS floor. Used by the random-walk anti-leakage regression
    test (deep must NOT beat naive).

    Parameters
    ----------
    basket:
        Column labels for the panel.
    n_obs:
        Number of observations.
    seed:
        Master RNG seed.
    idio_vol:
        Standard deviation of each independent series.
    start:
        First business-day date.

    Returns
    -------
    pandas.DataFrame
        The i.i.d. random-walk returns panel.

    Raises
    ------
    ValidationError
        If ``basket`` is empty/duplicated, ``n_obs < 2``, or ``idio_vol < 0``.
    """
    names = list(basket)
    if len(names) == 0:
        raise ValidationError("random_walk_panel: basket must be non-empty.")
    if len(set(names)) != len(names):
        raise ValidationError("random_walk_panel: basket must not contain duplicates.")
    if n_obs < 2:
        raise ValidationError(f"random_walk_panel: n_obs must be >= 2, got {n_obs}.")
    if idio_vol < 0.0:
        raise ValidationError(f"random_walk_panel: idio_vol must be >= 0, got {idio_vol}.")

    gen = make_rng(seed)
    # Pure i.i.d. Gaussian noise: no common factor, no autocorrelation. The
    # next-step return is genuinely unpredictable, so the naive r_hat = 0 forecast
    # is provably the OOS floor (the strictest honest-null testbed).
    returns = gen.standard_normal((n_obs, len(names))) * idio_vol
    index = _business_index(n_obs, start=start)
    return pd.DataFrame(returns, index=index, columns=names, dtype="float64")
