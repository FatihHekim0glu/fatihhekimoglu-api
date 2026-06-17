"""Synthetic sentiment + price generator and data loaders.

The shipped DEFAULT (tests and the deployed tool) runs on a SYNTHETIC generator —
no Reddit/Pushshift/Polygon keys required — constructed so the in-sample
sentiment-return correlation DECAYS out-of-sample and FAILS the Deflated Sharpe
after costs. That is the honest null BY CONSTRUCTION: there is a mild early
relationship dominated by contemporaneous attention/return feedback that does not
persist, so a leakage-free backtest cannot extract a durable edge.

The generator emits, per (ticker, day):

- a daily sentiment panel (mean compound score, mention count, positive-share),
- a correlated-ish price/return panel,

all seeded via :func:`wsb_sentiment._rng.make_rng` so a given ``(tickers, start,
end, seed)`` reproduces byte-identical data.

Real-data loaders (cached parquet sentiment, Polygon prices) are provided for the
ingest path; they import their heavy dependencies LAZILY. A point-in-time
universe hook restricts the tradable set to a PIT-consistent S&P-500 membership so
no future constituent is selected.

Importing this module has no side effects.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

import numpy as np
import pandas as pd

from wsb_sentiment._exceptions import ValidationError
from wsb_sentiment._rng import make_rng
from wsb_sentiment._typing import PricesLike
from wsb_sentiment._validation import ensure_dataframe


def _stable_hash(*parts: str) -> int:
    """Return a process-independent 31-bit hash of ``parts``.

    Python's builtin :func:`hash` is salted per-process (``PYTHONHASHSEED``), so it
    cannot back a reproducible synthetic generator. A BLAKE2b digest of the
    null-joined parts gives the same value in every process, so a given request
    reproduces byte-identical data and a byte-identical PIT universe.
    """
    joined = "\x00".join(parts).encode("utf-8")
    digest = hashlib.blake2b(joined, digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFF


#: Where a sentiment/price panel ultimately came from. Returned alongside the
#: data so callers (and the API ``data_source`` field) can report provenance.
DataSource = Literal["polygon", "cache", "synthetic"]

# quantcore-candidate: synthetic structure mirrors hrp-portfolio:src/hrp/data.py
# (seeded GBM prices) extended with a decaying sentiment-return coupling.


@dataclass(frozen=True, slots=True)
class SyntheticPanel:
    """A synthetic sentiment + price bundle with honest-null structure.

    Attributes
    ----------
    mean_compound:
        Wide ``day x ticker`` panel of the daily mean compound sentiment.
    mention_count:
        Wide ``day x ticker`` panel of the daily mention count.
    positive_share:
        Wide ``day x ticker`` panel of the daily positive-mention share.
    prices:
        Wide ``day x ticker`` panel of synthetic close prices.
    universe_mask:
        Wide ``day x ticker`` boolean point-in-time tradability mask.
    data_source:
        Always ``"synthetic"`` for this generator.
    """

    mean_compound: pd.DataFrame
    mention_count: pd.DataFrame
    positive_share: pd.DataFrame
    prices: pd.DataFrame
    universe_mask: pd.DataFrame
    data_source: DataSource = "synthetic"
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` (ISO date keys, NaN -> None)."""

        def _panel(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
            is_bool: dict[str, bool] = {
                str(c): bool(pd.api.types.is_bool_dtype(df[c])) for c in df.columns
            }
            return {
                str(idx): {
                    str(c): (None if pd.isna(v) else (bool(v) if is_bool[str(c)] else float(v)))
                    for c, v in row.items()
                }
                for idx, row in df.iterrows()
            }

        return {
            "mean_compound": _panel(self.mean_compound),
            "mention_count": _panel(self.mention_count),
            "positive_share": _panel(self.positive_share),
            "prices": _panel(self.prices),
            "universe_mask": _panel(self.universe_mask),
            "data_source": str(self.data_source),
            "meta": dict(self.meta),
        }


def _business_days(start: date, end: date) -> pd.DatetimeIndex:
    """Inclusive business-day (Mon-Fri) index spanning ``[start, end]``."""
    return pd.date_range(start=start, end=end, freq="B")


def _seed_from_request(tickers: list[str], start: date, end: date, seed: int) -> int:
    """Derive a deterministic 31-bit master seed from the full request.

    Folding the ticker set and date span into the user ``seed`` means a given
    ``(tickers, start, end, seed)`` reproduces a byte-identical panel, while two
    different requests get independent (but still reproducible) randomness. The
    fold uses a PROCESS-INDEPENDENT hash so reproducibility survives ``-R``/
    ``PYTHONHASHSEED`` randomization.
    """
    return _stable_hash(",".join(tickers), start.isoformat(), end.isoformat(), str(int(seed)))


def generate_synthetic_panel(
    tickers: list[str],
    start: date,
    end: date,
    *,
    seed: int = 7,
    in_sample_corr: float = 0.8,
    oos_corr: float = 0.0,
    mention_lambda: float = 25.0,
    decay_at: float = 0.5,
) -> SyntheticPanel:
    """Generate a seeded synthetic sentiment + price panel with a DECAYING edge.

    The generator couples next-day returns to today's sentiment with a coefficient
    that is ``in_sample_corr`` early in the sample and decays toward ``oos_corr``,
    reaching ``oos_corr`` at the ``decay_at`` fraction of the sample and staying
    there. A separate CONTEMPORANEOUS attention/return feedback term confounds the
    relationship: it inflates the same-bar (untradable) correlation without adding
    any predictive power. The result: a clear in-sample LAGGED correlation that
    largely vanishes out-of-sample — failing the Deflated Sharpe after costs, the
    honest null by construction. The decay is robust across seeds (property-tested).

    Parameters
    ----------
    tickers:
        The ticker symbols to simulate.
    start, end:
        Inclusive date range (business days).
    seed:
        Master RNG seed; ``(tickers, start, end, seed)`` reproduces the panel.
    in_sample_corr:
        The early-sample sentiment -> next-day-return coupling coefficient
        (a unitless weight on the prior day's sentiment, not the literal
        correlation; larger means a stronger early predictive relationship).
    oos_corr:
        The late-sample coupling the relationship decays toward (``0.0`` = no
        out-of-sample predictability).
    mention_lambda:
        The base Poisson rate for daily per-ticker mention counts.
    decay_at:
        The sample fraction (``(0, 1]``) by which the coupling reaches
        ``oos_corr``; defaults to ``0.5`` so predictability is exhausted by the
        train/test boundary used downstream.

    Returns
    -------
    SyntheticPanel
        The seeded sentiment + price bundle.

    Raises
    ------
    ValidationError
        If ``tickers`` is empty, the date range is empty, or any coupling /
        ``mention_lambda`` argument is non-finite or negative.
    """
    if not tickers:
        raise ValidationError("generate_synthetic_panel: tickers must be non-empty.")
    if len(set(tickers)) != len(tickers):
        raise ValidationError("generate_synthetic_panel: tickers must be unique.")
    for label, value in (
        ("in_sample_corr", in_sample_corr),
        ("oos_corr", oos_corr),
        ("mention_lambda", mention_lambda),
    ):
        if not np.isfinite(value):
            raise ValidationError(f"generate_synthetic_panel: {label} must be finite, got {value}.")
    if mention_lambda <= 0.0:
        raise ValidationError(
            f"generate_synthetic_panel: mention_lambda must be positive, got {mention_lambda}."
        )
    if not (0.0 < decay_at <= 1.0):
        raise ValidationError(
            f"generate_synthetic_panel: decay_at must be in (0, 1], got {decay_at}."
        )

    index = _business_days(start, end)
    n_obs = len(index)
    n_assets = len(tickers)
    if n_obs == 0:
        raise ValidationError(
            f"generate_synthetic_panel: empty date range [{start}, {end}] (no business days)."
        )

    gen = make_rng(_seed_from_request(tickers, start, end, seed))

    # --- Daily sentiment: bounded, mildly autocorrelated in [-1, 1] -------- #
    raw = gen.standard_normal((n_obs, n_assets))
    smoothed = pd.DataFrame(raw, index=index, columns=tickers).ewm(span=5).mean().to_numpy()
    mean_compound = np.tanh(0.8 * smoothed)

    # --- Mention counts: Poisson with a per-ticker base rate --------------- #
    # The rate co-moves weakly with the absolute sentiment of the prior day, so
    # "attention" spikes alongside strong opinions (the confound we want present).
    base_rate = mention_lambda * gen.uniform(0.5, 1.5, size=n_assets)
    prior_abs = np.vstack([np.zeros((1, n_assets)), np.abs(mean_compound[:-1])])
    lam = base_rate[None, :] * (1.0 + 0.6 * prior_abs)
    mention_count = gen.poisson(lam=lam).astype("float64")

    # --- Positive-share: logistic transform of the compound score ---------- #
    positive_share = 1.0 / (1.0 + np.exp(-3.0 * mean_compound))

    # --- Returns: decaying lagged-sentiment edge + contemporaneous feedback  #
    # The lagged-sentiment coupling decays from ``in_sample_corr`` to ``oos_corr``,
    # reaching ``oos_corr`` at the ``decay_at`` fraction of the sample and staying
    # there, so PREDICTIVE power is exhausted by the train/test boundary and any
    # tradable relationship vanishes out-of-sample. A separate contemporaneous
    # attention/return feedback term (sentiment co-moves with the SAME-bar return)
    # inflates the in-sample same-bar correlation without being tradable — the
    # honest-null confound.
    t_frac = np.arange(n_obs, dtype="float64") / float(max(n_obs - 1, 1))
    ramp = np.clip(t_frac / decay_at, 0.0, 1.0)
    coupling = (in_sample_corr + (oos_corr - in_sample_corr) * ramp).reshape(-1, 1)
    idio = gen.standard_normal((n_obs, n_assets))
    factor = gen.standard_normal(n_obs).reshape(-1, 1)

    dt = 1.0 / 252.0
    sigma = gen.uniform(0.20, 0.50, size=n_assets)
    betas = gen.uniform(0.5, 1.3, size=n_assets)
    base_vol = (sigma * np.sqrt(dt))[None, :]

    # Tradable component: TODAY's return reacts to YESTERDAY's sentiment.
    lagged_sentiment = np.vstack([np.zeros((1, n_assets)), mean_compound[:-1]])
    predictive = coupling * lagged_sentiment * base_vol

    # Confound: same-bar feedback (NOT tradable) plus a common market factor and
    # idiosyncratic noise dominate the predictive term.
    contemporaneous = 0.25 * mean_compound * base_vol
    diffusion = base_vol * idio + (0.08 * np.sqrt(dt)) * factor * betas[None, :]

    log_ret = predictive + contemporaneous + diffusion
    log_ret[0, :] = 0.0  # anchor the first observation at the start price

    start_prices = gen.uniform(15.0, 250.0, size=n_assets)
    prices = start_prices[None, :] * np.exp(np.cumsum(log_ret, axis=0))

    cols = pd.Index(tickers)
    mean_compound_df = pd.DataFrame(mean_compound, index=index, columns=cols)
    mention_count_df = pd.DataFrame(mention_count, index=index, columns=cols)
    positive_share_df = pd.DataFrame(positive_share, index=index, columns=cols)
    prices_df = pd.DataFrame(prices, index=index, columns=cols)
    universe_mask = pit_universe(tickers, index, source_pref="synthetic")

    meta: dict[str, Any] = {
        "seed": int(seed),
        "n_obs": int(n_obs),
        "n_assets": int(n_assets),
        "in_sample_corr": float(in_sample_corr),
        "oos_corr": float(oos_corr),
        "mention_lambda": float(mention_lambda),
        "decay_at": float(decay_at),
        "start": start.isoformat(),
        "end": end.isoformat(),
    }
    return SyntheticPanel(
        mean_compound=mean_compound_df,
        mention_count=mention_count_df,
        positive_share=positive_share_df,
        prices=prices_df,
        universe_mask=universe_mask,
        data_source="synthetic",
        meta=meta,
    )


def _load_cached_panel(
    tickers: list[str],
    start: date,
    end: date,
    cache_path: str,
) -> SyntheticPanel | None:
    """Try to read a precomputed sentiment + price parquet bundle (lazy import).

    LAZY IMPORT: ``pyarrow``/``pandas`` parquet reading (the ``data`` extra) is
    imported here, never at module import time. The cache directory is expected to
    hold ``mean_compound.parquet``, ``mention_count.parquet``,
    ``positive_share.parquet``, and ``prices.parquet`` (wide ``day x ticker``).
    Returns ``None`` (rather than raising) on any miss so the caller falls back to
    the synthetic generator.
    """
    import os

    names = ("mean_compound", "mention_count", "positive_share", "prices")
    paths = {name: os.path.join(cache_path, f"{name}.parquet") for name in names}
    if not all(os.path.isfile(p) for p in paths.values()):
        return None

    frames: dict[str, pd.DataFrame] = {}
    for name, path in paths.items():
        frame = pd.read_parquet(path)
        frame.index = pd.to_datetime(frame.index)
        # Restrict to the requested tickers (present subset) and date window.
        present = [t for t in tickers if t in frame.columns]
        if not present:
            return None
        window = (frame.index >= pd.Timestamp(start)) & (frame.index <= pd.Timestamp(end))
        frames[name] = frame.loc[window, present].astype("float64")

    index = frames["mean_compound"].index
    if len(index) == 0:
        return None
    present_tickers = [str(c) for c in frames["mean_compound"].columns]
    universe_mask = pit_universe(present_tickers, pd.DatetimeIndex(index), source_pref="synthetic")
    return SyntheticPanel(
        mean_compound=frames["mean_compound"],
        mention_count=frames["mention_count"],
        positive_share=frames["positive_share"],
        prices=frames["prices"],
        universe_mask=universe_mask,
        data_source="cache",
        meta={"cache_path": str(cache_path)},
    )


def load_sentiment_panel(
    tickers: list[str],
    start: date,
    end: date,
    *,
    source_pref: Literal["auto", "cache", "synthetic"] = "synthetic",
    seed: int = 7,
    cache_path: str | None = None,
) -> tuple[SyntheticPanel, DataSource]:
    """Load the daily sentiment + price panel for the request (synthetic default).

    Resolution order: a precomputed cached parquet (``"cache"``) when available,
    else the deterministic :func:`generate_synthetic_panel` (``"synthetic"``). The
    deployed default is ``"synthetic"`` so the result is reproducible and the null
    is honest; no live Pushshift/PRAW ingestion or VADER scoring happens here.

    LAZY IMPORT: the parquet/diskcache reader (the ``data`` extra) is imported
    inside :func:`_load_cached_panel`, never at module import time.

    Parameters
    ----------
    tickers:
        The ticker symbols to load.
    start, end:
        Inclusive date range.
    source_pref:
        Preferred source; ``"synthetic"`` (the deployed default) always succeeds.
        ``"cache"``/``"auto"`` try the parquet cache first then fall back.
    seed:
        Master RNG seed forwarded to the synthetic generator.
    cache_path:
        Optional path to a precomputed sentiment parquet directory.

    Returns
    -------
    tuple[SyntheticPanel, DataSource]
        The loaded panel and the source it came from.

    Raises
    ------
    ValidationError
        If ``source_pref`` is not one of ``{"auto", "cache", "synthetic"}``.
    """
    if source_pref not in ("auto", "cache", "synthetic"):
        raise ValidationError(
            "load_sentiment_panel: source_pref must be one of "
            f"{{'auto', 'cache', 'synthetic'}}, got {source_pref!r}."
        )

    if source_pref in ("auto", "cache") and cache_path is not None:
        try:
            cached = _load_cached_panel(tickers, start, end, cache_path)
        except Exception:  # cache is best-effort; a read failure never aborts the run
            cached = None
        if cached is not None:
            return cached, "cache"
        if source_pref == "cache":
            # An explicit cache request that misses falls back to synthetic rather
            # than failing, so the deployed default is always reproducible.
            pass

    panel = generate_synthetic_panel(tickers, start, end, seed=seed)
    return panel, "synthetic"


def compute_returns(prices: PricesLike) -> pd.DataFrame:
    r"""Convert a price panel to forward-safe simple returns.

    NO-LOOKAHEAD REQUIREMENT: returns are computed with
    ``prices.pct_change(fill_method=None)`` — prices are NEVER forward-filled
    before differencing, because ffill-then-diff manufactures spurious zero returns
    across gaps and leaks information. The leading all-NaN row is dropped.

    Parameters
    ----------
    prices:
        A wide panel of prices (rows = date, columns = ticker).

    Returns
    -------
    pandas.DataFrame
        Simple returns with the leading NaN row removed.

    Raises
    ------
    ValidationError
        If ``prices`` cannot be coerced to a 2-D numeric panel.
    """
    # Coerce (allowing NaN: real price panels have gaps) without mutating input.
    frame = ensure_dataframe(prices, name="prices", allow_nan=True)
    # fill_method=None is critical: do NOT forward-fill before differencing.
    returns = frame.pct_change(fill_method=None)
    # Drop only the leading all-NaN row produced by differencing; interior NaNs
    # (genuine gaps) are preserved for the caller to handle explicitly.
    return returns.iloc[1:]


def pit_universe(
    tickers: list[str],
    sessions: pd.DatetimeIndex,
    *,
    source_pref: Literal["auto", "synthetic"] = "synthetic",
) -> pd.DataFrame:
    """Return a point-in-time tradability mask (no future-constituent selection).

    Builds a wide ``day x ticker`` boolean mask of S&P-500 membership AS-OF each
    date so the tradable signal never selects a symbol on the basis of FUTURE index
    membership. The real path consults the Polygon S&P-500 universe; the synthetic
    default simulates a PIT-consistent membership set. Meme tickers outside the
    universe are descriptive-only (mask ``False``) and never traded.

    The synthetic membership is MONOTONE in time per ticker: each ticker is granted
    membership on a deterministic "admission date" derived from its symbol and stays
    a member thereafter. Crucially the admission date depends only on the symbol and
    the session calendar, never on any future price/sentiment realization, so the
    mask carries no look-ahead.

    Parameters
    ----------
    tickers:
        The candidate ticker symbols.
    sessions:
        The trading-session calendar to build the mask over.
    source_pref:
        Preferred universe source; ``"synthetic"`` simulates a PIT membership.

    Returns
    -------
    pandas.DataFrame
        A wide ``day x ticker`` boolean tradability mask.

    Raises
    ------
    ValidationError
        If ``tickers`` is empty, ``sessions`` is empty, or ``source_pref`` is
        unsupported.
    """
    if source_pref not in ("auto", "synthetic"):
        raise ValidationError(
            f"pit_universe: source_pref must be 'auto' or 'synthetic', got {source_pref!r}."
        )
    if not tickers:
        raise ValidationError("pit_universe: tickers must be non-empty.")
    sessions = pd.DatetimeIndex(sessions)
    n_obs = len(sessions)
    if n_obs == 0:
        raise ValidationError("pit_universe: sessions must be non-empty.")

    columns: dict[str, np.ndarray] = {}
    for ticker in tickers:
        # Deterministic admission fraction in [0, 1) from the symbol alone. A
        # fraction of 0 means "in the universe for the whole sample"; larger
        # fractions admit the ticker partway through (and it then stays in). The
        # hash is PROCESS-INDEPENDENT so the PIT mask is reproducible.
        admit_frac = _stable_hash("pit_admit", str(ticker)) / float(0x7FFFFFFF)
        # Roughly two-thirds of names are full-sample members; the rest are
        # admitted partway through, so the mask is non-trivially time-varying.
        admit_idx = 0 if admit_frac < (2.0 / 3.0) else int(admit_frac * n_obs)
        admit_idx = min(admit_idx, n_obs - 1)
        col = np.zeros(n_obs, dtype=bool)
        col[admit_idx:] = True
        columns[ticker] = col

    return pd.DataFrame(columns, index=sessions, columns=pd.Index(tickers))
