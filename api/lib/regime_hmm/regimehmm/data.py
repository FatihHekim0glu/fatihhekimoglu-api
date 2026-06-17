"""Data layer: synthetic regime-switch generator + real EOD price loader.

Two responsibilities, both side-effect-free at import:

1. A SYNTHETIC regime-switch generator (:func:`generate_regime_switch`) — a 2/3-state
   Gaussian HMM-style return series with PERSISTENT states of sharply different
   volatility, seeded via :func:`regimehmm._rng.make_rng`. Every test in the suite
   runs on this generator; no test touches the network.

2. A real EOD price loader (:func:`get_prices`, :func:`compute_returns`) for the
   deployed backend: it tries the Polygon provider (when ``source_pref="polygon"``)
   and degrades to the synthetic series on any upstream failure, reporting the
   provenance via the returned :data:`DataSource`.

Heavy data dependencies (polygon/httpx, yfinance, diskcache, pyarrow) live behind
the ``data`` extra and are imported LAZILY inside the loader functions. Importing
this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

import numpy as np
import pandas as pd

from regimehmm._exceptions import ValidationError
from regimehmm._rng import make_rng
from regimehmm._typing import PricesLike
from regimehmm._validation import ensure_dataframe, ensure_series

#: Where a price/return series ultimately came from. Returned alongside data so
#: callers (and the API ``data_source`` field) can report provenance.
DataSource = Literal["polygon", "yfinance", "synthetic", "cache"]

#: Supported synthetic feature sets, mirroring the API ``feature_set`` field.
FeatureSet = Literal["returns", "returns_vol", "returns_vol_macro"]

#: Canonical default per-state mean returns in ascending-mean order (state ``0``
#: is the calm, modestly-positive low-vol regime; the last state is the
#: turbulent, negative-drift crisis regime). Used when ``means`` is omitted.
_DEFAULT_MEANS: tuple[float, ...] = (0.0008, -0.0004, -0.0010, -0.0016)

#: Canonical default per-state volatilities, sharply different and ascending
#: (a calm low-vol regime ... a high-vol crisis regime). Used when ``vols`` is
#: omitted; the first ``n_states`` entries are taken.
_DEFAULT_VOLS: tuple[float, ...] = (0.006, 0.012, 0.020, 0.035)


@dataclass(frozen=True, slots=True)
class RegimeSwitchSeries:
    """A synthetic regime-switch sample with its ground-truth hidden states.

    Attributes
    ----------
    returns:
        The generated per-period return series (a ``pd.Series`` indexed by a
        business-day :class:`pandas.DatetimeIndex`).
    states:
        The ground-truth hidden-state path that generated ``returns`` (canonical
        order: ``0`` = lowest-mean / lowest-vol regime). Used to score the decoder
        in tests; NEVER available to the model at fit time.
    means:
        The ``(n_states,)`` per-state mean returns used by the generator.
    vols:
        The ``(n_states,)`` per-state volatilities (persistent, sharply different).
    transmat:
        The ``(n_states, n_states)`` row-stochastic generating transition matrix.
    """

    returns: pd.Series
    states: np.ndarray
    means: np.ndarray
    vols: np.ndarray
    transmat: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this sample."""
        return {
            "returns": {str(k): float(v) for k, v in self.returns.items()},
            "states": [int(s) for s in self.states],
            "means": [float(m) for m in self.means],
            "vols": [float(v) for v in self.vols],
            "transmat": [[float(v) for v in row] for row in self.transmat],
            "meta": dict(self.meta),
        }


def generate_regime_switch(
    n_obs: int,
    *,
    n_states: int = 2,
    persistence: float = 0.97,
    means: tuple[float, ...] | None = None,
    vols: tuple[float, ...] | None = None,
    seed: int = 7,
) -> RegimeSwitchSeries:
    r"""Generate a seeded synthetic regime-switch return series.

    Builds a ``persistence``-sticky ``n_states``-state Markov chain (a transition
    matrix with ``persistence`` on the diagonal and the remainder spread evenly
    off-diagonal), walks it for ``n_obs`` steps, and emits a Gaussian return from
    the active state's ``(mean, vol)`` at each step. The default per-state ``vols``
    are sharply different (e.g. a calm low-vol regime and a turbulent high-vol
    regime) and the states are PERSISTENT, exactly the structure a Gaussian HMM is
    meant to recover.

    DETERMINISM: all randomness flows from :func:`regimehmm._rng.make_rng(seed)`, so
    a fixed ``(n_obs, n_states, persistence, seed)`` reproduces the series and its
    ground-truth states byte-for-byte. This is the generator the ENTIRE test suite
    runs on; it touches no network.

    GROUND TRUTH: the returned ``states`` are the true generating path (canonical
    order). Tests use them to score the decoder, but the model never sees them at
    fit time.

    Parameters
    ----------
    n_obs:
        The number of observations to generate (``>= 1``).
    n_states:
        The number of regimes (``2`` or ``3`` for the canonical fixtures).
    persistence:
        The self-transition probability on the diagonal (``in (0, 1)``); higher =
        stickier regimes.
    means:
        Optional per-state mean returns; defaults to a spread of small
        positive/negative means in canonical (ascending-mean) order.
    vols:
        Optional per-state volatilities; defaults to sharply different, ascending
        vols (low-vol calm regime ... high-vol crisis regime).
    seed:
        Master seed for the generator.

    Returns
    -------
    RegimeSwitchSeries
        The frozen sample (returns + ground-truth states + generating parameters).

    Raises
    ------
    ValidationError
        If ``n_obs < 1``, ``n_states < 1``, ``persistence`` is outside ``(0, 1)``,
        or a supplied ``means``/``vols`` length does not match ``n_states``.
    """
    if n_obs < 1:
        raise ValidationError(f"generate_regime_switch: n_obs must be >= 1, got {n_obs}.")
    if n_states < 1:
        raise ValidationError(f"generate_regime_switch: n_states must be >= 1, got {n_states}.")
    if not 0.0 < persistence < 1.0:
        raise ValidationError(
            f"generate_regime_switch: persistence must be in (0, 1), got {persistence}."
        )

    means_arr = _resolve_state_params(means, _DEFAULT_MEANS, n_states, "means")
    vols_arr = _resolve_state_params(vols, _DEFAULT_VOLS, n_states, "vols")
    if bool((vols_arr <= 0.0).any()):
        raise ValidationError("generate_regime_switch: every vol must be strictly positive.")

    transmat = _sticky_transition(n_states, persistence)

    gen = make_rng(seed)
    states = np.empty(n_obs, dtype=np.intp)
    returns = np.empty(n_obs, dtype="float64")

    # Walk the sticky Markov chain, emitting one Gaussian draw per step from the
    # active state's (mean, vol). Start in state 0 (the canonical calm regime).
    state = 0
    for t in range(n_obs):
        if t > 0:
            state = int(gen.choice(n_states, p=transmat[state]))
        states[t] = state
        returns[t] = gen.normal(loc=float(means_arr[state]), scale=float(vols_arr[state]))

    index = pd.date_range("2010-01-01", periods=n_obs, freq="B")
    series = pd.Series(returns, index=index, name="regime_switch")
    return RegimeSwitchSeries(
        returns=series,
        states=states,
        means=means_arr,
        vols=vols_arr,
        transmat=transmat,
        meta={
            "n_obs": int(n_obs),
            "n_states": int(n_states),
            "persistence": float(persistence),
            "seed": int(seed),
        },
    )


def _resolve_state_params(
    supplied: tuple[float, ...] | None,
    defaults: tuple[float, ...],
    n_states: int,
    name: str,
) -> np.ndarray:
    """Validate/coerce a per-state parameter tuple to an ``(n_states,)`` array.

    When ``supplied`` is ``None`` the first ``n_states`` canonical ``defaults``
    are taken (which requires ``n_states <= len(defaults)``); otherwise the
    supplied length must equal ``n_states``.
    """
    if supplied is None:
        if n_states > len(defaults):
            raise ValidationError(
                f"generate_regime_switch: no default {name} for n_states={n_states} "
                f"(defaults cover up to {len(defaults)} states); pass {name} explicitly."
            )
        return np.asarray(defaults[:n_states], dtype="float64")
    if len(supplied) != n_states:
        raise ValidationError(
            f"generate_regime_switch: {name} has length {len(supplied)} but n_states={n_states}."
        )
    return np.asarray(supplied, dtype="float64")


def _sticky_transition(n_states: int, persistence: float) -> np.ndarray:
    """Row-stochastic ``(n_states, n_states)`` matrix: ``persistence`` on the diagonal.

    The remaining ``1 - persistence`` mass is spread evenly across the
    off-diagonal entries of each row (for ``n_states == 1`` the single state is
    absorbing, i.e. a degenerate ``[[1.0]]``).
    """
    if n_states == 1:
        return np.ones((1, 1), dtype="float64")
    off = (1.0 - persistence) / (n_states - 1)
    transmat = np.full((n_states, n_states), off, dtype="float64")
    np.fill_diagonal(transmat, persistence)
    return transmat


def build_features(
    returns: pd.Series,
    *,
    feature_set: FeatureSet = "returns_vol",
    vol_window: int = 21,
) -> pd.DataFrame:
    r"""Build the HMM emission feature matrix from a return series.

    Assembles the per-observation feature columns selected by ``feature_set``:

    * ``"returns"`` — the return itself (column 0, the canonicalization key);
    * ``"returns_vol"`` — return plus a TRAILING realized-volatility feature
      (rolling std over ``vol_window``, using only past returns — no lookahead);
    * ``"returns_vol_macro"`` — the above plus a slow macro/trend feature.

    NO-LOOKAHEAD: every derived feature at ``t`` uses returns ``<= t`` only
    (trailing windows), so the feature matrix is causal. The leading rows with an
    incomplete window are dropped. Standardization (StandardScaler) is applied
    train-only by the caller, never here.

    Parameters
    ----------
    returns:
        The per-period return series.
    feature_set:
        Which feature columns to build.
    vol_window:
        Trailing window length for the realized-volatility feature.

    Returns
    -------
    pandas.DataFrame
        The causal feature matrix (column 0 is the return, the canonicalization
        key), leading incomplete-window rows dropped.

    Raises
    ------
    ValidationError
        If ``feature_set`` is unsupported or ``vol_window < 1``.
    """
    if feature_set not in ("returns", "returns_vol", "returns_vol_macro"):
        raise ValidationError(f"build_features: unsupported feature_set {feature_set!r}.")
    if vol_window < 1:
        raise ValidationError(f"build_features: vol_window must be >= 1, got {vol_window}.")

    ret = ensure_series(returns, name="returns")

    # Column 0 is always the raw return — the canonicalization key downstream.
    columns: dict[str, pd.Series] = {"return": ret}

    if feature_set in ("returns_vol", "returns_vol_macro"):
        # TRAILING realized volatility: rolling std over PAST returns only. The
        # window closes at t (inclusive of the current return), so the feature at
        # t depends on returns <= t — no lookahead. Leading rows with an
        # incomplete window become NaN and are dropped below.
        columns["realized_vol"] = ret.rolling(window=vol_window, min_periods=vol_window).std(ddof=0)

    if feature_set == "returns_vol_macro":
        # A slow macro/trend feature: a long trailing mean of returns (a causal
        # drift proxy). Window length is a multiple of the vol window so it is
        # strictly slower-moving; uses past returns <= t only.
        macro_window = vol_window * 6
        columns["macro_trend"] = ret.rolling(window=macro_window, min_periods=macro_window).mean()

    features = pd.DataFrame(columns)
    # Drop leading rows whose trailing windows were incomplete (NaN), so every
    # surviving row is fully causal.
    features = features.dropna(axis=0, how="any")
    return features.astype("float64")


def _synthetic_prices(ticker: str, start: date, end: date, *, seed: int = 7) -> pd.DataFrame:
    """Deterministic synthetic single-ticker price panel from a regime-switch series.

    Offline/CI fallback used by :func:`get_prices`: generates a regime-switch
    return series via :func:`generate_regime_switch`, compounds it into a strictly
    positive price level, and returns a one-column ``date x ticker`` frame. Seeded
    off the request so the same ``(ticker, start, end, seed)`` is byte-identical.
    """
    if end <= start:
        raise ValidationError(f"_synthetic_prices: end ({end}) must be after start ({start}).")
    index = pd.date_range(start=start, end=end, freq="B")
    n_obs = len(index)

    # Empty span (e.g. a single weekend day): return a typed empty panel.
    if n_obs == 0:
        return pd.DataFrame(index=index, columns=[ticker], dtype="float64")

    # Derive a deterministic seed from the request (ticker + span + seed), masked
    # to 31 bits so the same request is byte-identical across runs/platforms.
    digest = hash((ticker, start.isoformat(), end.isoformat(), int(seed)))
    sample = generate_regime_switch(n_obs, n_states=2, seed=digest & 0x7FFFFFFF)

    # Compound the regime-switch returns into a strictly positive price level,
    # anchoring the first observation at 100.0 so the panel starts clean.
    rets = sample.returns.to_numpy(dtype="float64").copy()
    rets[0] = 0.0
    prices = 100.0 * np.cumprod(1.0 + rets)
    return pd.DataFrame({ticker: prices}, index=index, dtype="float64")


def get_prices(
    ticker: str,
    start: date,
    end: date,
    *,
    source_pref: Literal["polygon", "yfinance", "auto", "synthetic"] = "auto",
    seed: int = 7,
) -> tuple[pd.DataFrame, DataSource]:
    """Load a single-ticker EOD price panel with graceful synthetic fallback.

    With ``source_pref="polygon"`` the real Polygon EOD provider
    (:class:`regimehmm.data_providers.polygon.PolygonProvider`) is tried first and,
    on ANY failure (missing key, network error, empty payload), falls through to a
    deterministic synthetic regime-switch price series so the loader never
    hard-fails. ``source_pref="synthetic"`` forces the synthetic path (used by the
    backend's ``data_source_pref`` and by offline runs).

    LAZY IMPORT: the polygon provider (and ``httpx``) are imported inside this
    function, never at module import time.

    Parameters
    ----------
    ticker:
        The asset symbol (e.g. ``"SPY"``).
    start, end:
        Inclusive date range.
    source_pref:
        Source preference; ``"auto"`` and ``"polygon"`` try Polygon then fall back
        to synthetic, ``"synthetic"`` forces the offline path.
    seed:
        Seed for the synthetic fallback.

    Returns
    -------
    tuple[pandas.DataFrame, DataSource]
        The single-column price panel and the source it came from.

    Raises
    ------
    ValidationError
        If ``ticker`` is empty or ``end <= start``.
    """
    if not ticker or not ticker.strip():
        raise ValidationError("get_prices: ticker must be a non-empty string.")
    if end <= start:
        raise ValidationError(f"get_prices: end ({end}) must be after start ({start}).")

    # ``synthetic`` forces the offline path; ``polygon`` and ``auto`` try the real
    # Polygon provider first and fall through to synthetic on ANY failure (missing
    # key, network error, empty payload) so the loader never hard-fails.
    if source_pref in ("polygon", "auto"):
        try:
            frame = _fetch_polygon(ticker, start, end)
        except Exception:
            frame = None
        if frame is not None and not frame.empty:
            return frame.astype("float64"), "polygon"

    return _synthetic_prices(ticker, start, end, seed=seed), "synthetic"


def _fetch_polygon(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Fetch a single-ticker adjusted-close panel from Polygon (lazy import). May raise.

    LAZY IMPORT: :class:`regimehmm.data_providers.polygon.PolygonProvider` (and,
    inside it, ``httpx``) are imported here, never at module import time, so the
    ``data`` extra stays optional and importing this module touches no network.
    """
    from regimehmm.data_providers.polygon import PolygonProvider

    frame = PolygonProvider().fetch([ticker], start, end)
    if frame.empty or bool(frame.isna().to_numpy().all()):
        raise ValueError(f"Polygon returned no usable price data for {ticker!r}.")
    return frame


def compute_returns(prices: PricesLike) -> pd.Series:
    r"""Convert a single-ticker price panel to a simple return series.

    NO-LOOKAHEAD REQUIREMENT: returns are computed with
    ``prices.pct_change(fill_method=None)`` — prices are NEVER forward-filled before
    differencing (ffill-then-diff manufactures spurious zero returns across gaps and
    leaks information). The leading NaN row is dropped and the result is squeezed to
    a 1-D Series.

    Parameters
    ----------
    prices:
        A single-column price panel (rows = date).

    Returns
    -------
    pandas.Series
        The simple-return series with the leading NaN removed.

    Raises
    ------
    ValidationError
        If ``prices`` is malformed or has more than one column.
    """
    frame = ensure_dataframe(prices, name="prices", allow_nan=True)
    if frame.shape[1] != 1:
        raise ValidationError(
            f"compute_returns: expected a single-column price panel, got {frame.shape[1]} columns."
        )

    # NO-LOOKAHEAD REQUIREMENT: never forward-fill prices before differencing.
    # ffill-then-diff manufactures spurious zero returns across gaps and leaks
    # information; ``fill_method=None`` differences on each column's observed
    # values only.
    returns = frame.pct_change(fill_method=None)

    # Drop the leading all-NaN row produced by pct_change and squeeze to 1-D.
    series = returns.iloc[1:, 0].astype("float64")
    series.name = str(frame.columns[0])
    return series
