"""Synthetic event panel with a KNOWN injected effect (ground truth).

The deployed default and the entire test suite run on a deterministic synthetic
event panel so the placebo/HAC/DiD machinery is validated against ground truth:

- ``synthetic_event_panel`` injects a KNOWN cumulative abnormal return (CAR) into
  the event windows of rate-sensitive names, on top of a one-factor market model
  plus idiosyncratic noise. A correct event study must recover the injected CAR
  within tolerance.
- ``pure_noise_panel`` injects NO effect, so a correct, honest pipeline must
  return ``fed_effect_is_tradable=False`` and a placebo percentile that is
  ~uniform.

Everything is seeded via :func:`fedcausal._rng.make_rng`, so the same request
yields byte-identical panels. Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from fedcausal._constants import (
    DEFAULT_ESTIMATION_GAP,
    DEFAULT_ESTIMATION_WINDOW,
    MAX_EVENT_HALF_WIDTH,
)
from fedcausal._exceptions import ValidationError
from fedcausal._rng import make_rng

if TYPE_CHECKING:
    from fedcausal._typing import SurpriseLabel

#: Fixed reference start date for the synthetic panel (deterministic by default).
_DEFAULT_START: date = date(2015, 1, 1)

#: Sign-dependent move (per rate-sensitive name, spread over the event window)
#: added on hawkish/dovish events to create cross-sectional DiD heterogeneity.
#: Scaled by ``injected_car`` so a richer effect amplifies the treated/control gap.
_SURPRISE_TILT_FRAC: float = 0.5


@dataclass(frozen=True, slots=True)
class SyntheticPanel:
    """A synthetic event panel bundled with its ground-truth metadata.

    Attributes
    ----------
    returns:
        Wide panel of single-name simple returns (rows = trading date, columns =
        ticker).
    market:
        The 1-D market-factor return series used as the market-model regressor.
    announcement_dates:
        The FOMC announcement dates injected into the panel.
    surprises:
        The surprise label per announcement date (parallel to
        ``announcement_dates``).
    rate_sensitive:
        The subset of columns designated rate-sensitive (the DiD "treated" group
        that mechanically moves more around announcements).
    injected_car:
        The ground-truth cumulative abnormal return injected into rate-sensitive
        names over the event window (``0.0`` for a pure-noise panel).
    seed:
        The master seed the panel was generated from.
    """

    returns: pd.DataFrame
    market: pd.Series
    announcement_dates: list[date]
    surprises: list[SurpriseLabel]
    rate_sensitive: list[str]
    injected_car: float
    seed: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable summary ``dict`` (excludes bulk panels)."""
        return {
            "n_dates": int(self.returns.shape[0]),
            "n_names": int(self.returns.shape[1]),
            "n_events": len(self.announcement_dates),
            "rate_sensitive": list(self.rate_sensitive),
            "injected_car": float(self.injected_car),
            "seed": int(self.seed),
            "meta": dict(self.meta),
        }


def _classify_sign(rate_change: float) -> SurpriseLabel:
    """Map a signed synthetic rate change to a surprise label (no leakage)."""
    if rate_change > 0:
        return "hawkish"
    if rate_change < 0:
        return "dovish"
    return "neutral"


def _validate_params(
    *,
    n_names: int,
    n_events: int,
    spacing: int,
    estimation_window: int,
    event_half_width: int,
    rate_sensitive_frac: float,
    noise_scale: float,
) -> None:
    """Validate generator sizing parameters, raising :class:`ValidationError`."""
    if n_names < 2:
        raise ValidationError(f"n_names must be >= 2, got {n_names}.")
    if n_events < 1:
        raise ValidationError(f"n_events must be >= 1, got {n_events}.")
    if event_half_width < 1 or event_half_width > MAX_EVENT_HALF_WIDTH:
        raise ValidationError(
            f"event_half_width must be in [1, {MAX_EVENT_HALF_WIDTH}], got {event_half_width}."
        )
    if estimation_window < 2:
        raise ValidationError(f"estimation_window must be >= 2, got {estimation_window}.")
    # Spacing must exceed the full event window so adjacent windows never straddle.
    if spacing <= 2 * event_half_width:
        raise ValidationError(
            f"spacing ({spacing}) must exceed 2*event_half_width ({2 * event_half_width})."
        )
    if not 0.0 < rate_sensitive_frac < 1.0:
        raise ValidationError(
            f"rate_sensitive_frac must lie strictly in (0, 1), got {rate_sensitive_frac}."
        )
    if noise_scale <= 0.0 or not np.isfinite(noise_scale):
        raise ValidationError(f"noise_scale must be a positive finite float, got {noise_scale}.")


def synthetic_event_panel(
    *,
    n_names: int = 30,
    n_events: int = 16,
    spacing: int = 40,
    estimation_window: int = DEFAULT_ESTIMATION_WINDOW,
    event_half_width: int = 1,
    injected_car: float = 0.01,
    rate_sensitive_frac: float = 0.4,
    noise_scale: float = 0.01,
    seed: int = 7,
    start: date | None = None,
) -> SyntheticPanel:
    """Generate a synthetic event panel with a KNOWN injected CAR.

    Each name follows a one-factor market model (``alpha + beta * market + eps``).
    Around each FOMC announcement, a KNOWN ``injected_car`` is added to the event
    window of the rate-sensitive subset, split across the ``2k + 1`` event-window
    days, and a (smaller) sign-dependent move is added for hawkish/dovish events
    to create cross-sectional DiD heterogeneity. The estimation window carries NO
    injected effect, so an estimation-window-only market model is unbiased.

    Parameters
    ----------
    n_names:
        Number of single-name columns.
    n_events:
        Number of FOMC announcements to inject.
    spacing:
        Trading-day spacing between consecutive announcements (kept large enough
        that event windows never straddle).
    estimation_window:
        Pre-event estimation-window length (used to size the leading burn-in so
        the first event has enough history).
    event_half_width:
        The half-width ``k`` of the event window.
    injected_car:
        The ground-truth CAR injected into rate-sensitive names per event.
    rate_sensitive_frac:
        Fraction of names designated rate-sensitive (the DiD treated group).
    noise_scale:
        Idiosyncratic daily return standard deviation.
    seed:
        Master RNG seed.
    start:
        Optional first calendar date (defaults to a fixed reference date).

    Returns
    -------
    SyntheticPanel
        The panel plus its ground-truth metadata.

    Raises
    ------
    ValidationError
        If any size parameter is out of range.
    """
    _validate_params(
        n_names=n_names,
        n_events=n_events,
        spacing=spacing,
        estimation_window=estimation_window,
        event_half_width=event_half_width,
        rate_sensitive_frac=rate_sensitive_frac,
        noise_scale=noise_scale,
    )

    rng = make_rng(seed)
    k = event_half_width

    # ---- trading-day grid -------------------------------------------------- #
    # Leading burn-in must cover the estimation window + the estimation gap + the
    # event half-width so the FIRST event has a complete, non-overlapping
    # pre-event estimation window. A trailing pad covers the last event window.
    burn_in = estimation_window + DEFAULT_ESTIMATION_GAP + k
    first_event_pos = burn_in
    last_event_pos = first_event_pos + (n_events - 1) * spacing
    n_obs = last_event_pos + k + 1
    start_date = start if start is not None else _DEFAULT_START
    grid = pd.date_range(start=start_date, periods=n_obs, freq="B")

    event_positions = [first_event_pos + i * spacing for i in range(n_events)]
    announcement_dates = [grid[pos].date() for pos in event_positions]

    # ---- names & rate-sensitivity ----------------------------------------- #
    tickers = [f"N{i:03d}" for i in range(n_names)]
    n_rate_sensitive = max(1, min(n_names - 1, round(n_names * rate_sensitive_frac)))
    rate_sensitive = tickers[:n_rate_sensitive]
    rate_sensitive_mask = np.zeros(n_names, dtype=bool)
    rate_sensitive_mask[:n_rate_sensitive] = True

    # ---- one-factor market model ------------------------------------------ #
    # Market factor: zero-mean daily returns. Per-name alpha/beta are stable.
    market = rng.normal(0.0, 0.008, size=n_obs)
    alpha = rng.uniform(-0.0002, 0.0002, size=n_names)
    beta = rng.uniform(0.6, 1.4, size=n_names)
    eps = rng.normal(0.0, noise_scale, size=(n_obs, n_names))

    # Base returns: r_it = alpha_i + beta_i * market_t + eps_it (NO event effect).
    base = alpha[None, :] + np.outer(market, beta) + eps
    returns = base.copy()

    # ---- inject the KNOWN CAR + sign-dependent DiD heterogeneity ----------- #
    # Spread the injected CAR evenly across the (2k + 1) event-window days of the
    # rate-sensitive names so the summed abnormal return over the window recovers
    # ``injected_car``. Hawkish/dovish events add a sign-dependent tilt (also
    # spread over the window) to give the treated-vs-control DiD a signal.
    window_len = 2 * k + 1
    per_day_car = injected_car / window_len
    tilt = _SURPRISE_TILT_FRAC * abs(injected_car)
    per_day_tilt = tilt / window_len

    surprises: list[SurpriseLabel] = []
    # Deterministic surprise pattern: cycle hawkish / dovish / neutral so all
    # three signs are present regardless of n_events (>=3 gives all three).
    sign_cycle = (1.0, -1.0, 0.0)
    for i, pos in enumerate(event_positions):
        rate_change = sign_cycle[i % len(sign_cycle)]
        surprises.append(_classify_sign(rate_change))
        lo = pos - k
        hi = pos + k  # inclusive
        # Known CAR into rate-sensitive names (the recovery target).
        returns[lo : hi + 1, rate_sensitive_mask] += per_day_car
        # Sign-dependent tilt into rate-sensitive names only (DiD heterogeneity:
        # treated names move with the surprise sign; controls do not).
        if rate_change != 0.0:
            returns[lo : hi + 1, rate_sensitive_mask] += rate_change * per_day_tilt

    returns_df = pd.DataFrame(returns, index=grid, columns=tickers).astype("float64")
    market_series = pd.Series(market, index=grid, name="market", dtype="float64")

    meta: dict[str, Any] = {
        "n_obs": int(n_obs),
        "spacing": int(spacing),
        "event_half_width": int(k),
        "estimation_window": int(estimation_window),
        "noise_scale": float(noise_scale),
        "rate_sensitive_frac": float(rate_sensitive_frac),
    }
    return SyntheticPanel(
        returns=returns_df,
        market=market_series,
        announcement_dates=announcement_dates,
        surprises=surprises,
        rate_sensitive=rate_sensitive,
        injected_car=float(injected_car),
        seed=int(seed),
        meta=meta,
    )


def rate_sensitive_panel(
    *,
    seed: int = 7,
    injected_car: float = 0.012,
    **kwargs: Any,
) -> SyntheticPanel:
    """Generate a panel with pronounced rate-sensitivity DiD heterogeneity.

    A thin preset over :func:`synthetic_event_panel` that amplifies the gap
    between rate-sensitive ("treated") and insensitive ("control") names around
    hawkish vs. dovish surprises, so the difference-in-differences design has a
    clear (but still honestly fragile, net-of-cost) treated-minus-control signal.

    Parameters
    ----------
    seed:
        Master RNG seed.
    injected_car:
        The CAR injected into rate-sensitive names.
    **kwargs:
        Forwarded to :func:`synthetic_event_panel`.

    Returns
    -------
    SyntheticPanel
        The rate-sensitive panel with ground-truth metadata.
    """
    return synthetic_event_panel(seed=seed, injected_car=injected_car, **kwargs)


def pure_noise_panel(
    *,
    seed: int = 7,
    **kwargs: Any,
) -> SyntheticPanel:
    """Generate a NO-EFFECT panel (the honest-null control).

    Identical structure to :func:`synthetic_event_panel` but with
    ``injected_car=0`` and no sign-dependent move: there is genuinely no abnormal
    return around announcements. A correct pipeline must yield
    ``fed_effect_is_tradable=False`` and a placebo percentile of the observed CAR
    that is ~uniform on this panel.

    Parameters
    ----------
    seed:
        Master RNG seed.
    **kwargs:
        Forwarded to :func:`synthetic_event_panel` (``injected_car`` is forced
        to ``0.0``).

    Returns
    -------
    SyntheticPanel
        The pure-noise panel (``injected_car == 0.0``).
    """
    kwargs.pop("injected_car", None)
    return synthetic_event_panel(seed=seed, injected_car=0.0, **kwargs)
