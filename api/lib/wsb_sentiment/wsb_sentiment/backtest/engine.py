"""Sentiment-signal backtest engine (anchored walk-forward, cost grid).

Runs the daily sentiment positions against forward returns with strict leakage
guards and realistic per-side costs:

- anchored (expanding) walk-forward with purge + embargo at each train/test
  boundary, where the standardizer is REFIT on each expanding train slice only;
- FORWARD-return labels only (positions from :mod:`wsb_sentiment.signal.build`
  are already ``shift(lag)``-ed, so each return is earned with a strictly-prior
  position);
- a per-side basis-point transaction cost charged on daily position turnover,
  swept over a sensitivity grid;
- two honest baselines: buy-and-hold (equal-weight long) and an ATTENTION-ONLY
  signal (positions from mention-count alone), so the value-add of the
  sentiment *polarity* over raw attention is isolated.

PIT-UNIVERSE DISCIPLINE: only tickers in the point-in-time tradable universe on
each date contribute to the traded book; meme tickers outside the universe are
descriptive-only and never traded (enforced via the ``universe_mask``).

IDENTICAL OOS INDEX: the signal, buy-and-hold, and attention-only baselines all
report on the exact same post-purge/embargo out-of-sample index, so any
comparison is apples-to-apples (regression-tested).

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from wsb_sentiment._typing import ReturnsLike, SentimentLike
from wsb_sentiment.signal.build import SignalSpec


def _safe_float(value: object) -> float | None:
    """Coerce ``value`` to a finite float, mapping NaN/Inf/None to ``None``."""
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


@dataclass(frozen=True, slots=True)
class SignalBacktestResult:
    """Immutable result of a sentiment-signal walk-forward backtest.

    Attributes
    ----------
    net_returns:
        The net (after-cost) out-of-sample portfolio return series.
    gross_returns:
        The gross (before-cost) out-of-sample portfolio return series.
    buyhold_returns:
        The equal-weight buy-and-hold OOS return series (baseline).
    attention_returns:
        The attention-only (mention-count) OOS return series (baseline).
    turnover:
        Per-day one-way turnover of the traded book.
    cost_bps:
        The per-side transaction cost (bps) used for this run.
    n_oos:
        The number of out-of-sample observations.
    """

    net_returns: pd.Series
    gross_returns: pd.Series
    buyhold_returns: pd.Series
    attention_returns: pd.Series
    turnover: pd.Series
    cost_bps: float
    n_oos: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` (ISO date keys, NaN -> None)."""

        def _series(s: pd.Series) -> dict[str, float | None]:
            return {str(k): _safe_float(v) for k, v in s.items()}

        return {
            "net_returns": _series(self.net_returns),
            "gross_returns": _series(self.gross_returns),
            "buyhold_returns": _series(self.buyhold_returns),
            "attention_returns": _series(self.attention_returns),
            "turnover": _series(self.turnover),
            "cost_bps": float(self.cost_bps),
            "n_oos": int(self.n_oos),
            "meta": dict(self.meta),
        }


def _coerce_panel(data: object, *, name: str) -> pd.DataFrame:
    """Coerce a wide ``day x ticker`` panel to a float64 DataFrame (NaN allowed).

    Positions carry leading NaN from ``shift`` and real returns/mentions have
    genuine gaps, so NaN is permitted here; the engine masks NaN to a no-position
    / no-return cell explicitly rather than rejecting the panel.
    """
    from wsb_sentiment._validation import ensure_dataframe

    return ensure_dataframe(data, name=name, allow_nan=True)


def _row_normalized_positions(
    raw: pd.DataFrame, mask: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the PIT mask and L1-normalize each day's positions to gross 1.0.

    ``raw`` positions outside the tradable universe (``mask`` False) are zeroed —
    they are descriptive-only and never traded. The surviving positions are then
    scaled so each day's absolute weights sum to one (a fully-invested book on the
    active names), leaving an all-flat day untouched. Returns the normalized
    weights and the applied (boolean) mask actually traded.
    """
    masked = raw.where(mask, other=0.0).fillna(0.0)
    gross = masked.abs().sum(axis=1)
    # Avoid divide-by-zero on flat days: those rows stay all-zero.
    scale = gross.where(gross > 0.0, other=1.0)
    weights = masked.div(scale, axis=0)
    return weights, masked.ne(0.0)


def _standardize_threshold_shift(
    panel: pd.DataFrame, spec: SignalSpec, *, train_end: pd.Timestamp
) -> pd.DataFrame:
    """Train-only standardize, threshold, and ``shift(lag)`` a wide panel.

    Self-contained replica of the signal pipeline used for the attention-only
    baseline so the engine does not depend on the (separately-owned)
    :mod:`wsb_sentiment.signal.build` stubs at runtime:

    1. optional trailing causal ``window`` smoothing;
    2. per-ticker z-score using mean/std fit on the TRAIN slice (``<= train_end``)
       ONLY — no out-of-sample statistic leaks into the standardization;
    3. threshold ``|z| > spec.threshold`` into ``{-1, 0, +1}`` (long/short) or
       ``{0, +1}`` (long/flat);
    4. ``shift(spec.lag)`` so a position earned on day ``t`` was decided strictly
       before ``t``.
    """
    from wsb_sentiment._constants import EPS

    smoothed = panel.rolling(window=spec.window, min_periods=1).mean() if spec.window > 1 else panel

    train = smoothed.loc[smoothed.index <= pd.Timestamp(train_end)]
    mean = train.mean(axis=0)
    std = train.std(axis=0, ddof=0).clip(lower=EPS)
    z = (smoothed - mean) / std

    # DataFrame-native sign (keeps the static type a DataFrame, unlike np.sign).
    sign = (z > 0.0).astype("float64") - (z < 0.0).astype("float64")
    active = z.abs() > float(spec.threshold)
    raw_pos = sign.where(active, other=0.0)
    if spec.long_only:
        raw_pos = raw_pos.clip(lower=0.0)

    return raw_pos.shift(spec.lag)


def attention_only_positions(
    mention_count: SentimentLike,
    spec: SignalSpec,
    *,
    train_end: pd.Timestamp,
) -> pd.DataFrame:
    """Build an ATTENTION-ONLY baseline signal from mention counts alone.

    Standardizes the (log) mention-count panel on the TRAIN slice only and maps it
    to positions exactly like the sentiment signal, but using attention magnitude
    rather than sentiment polarity. Comparing the sentiment signal against this
    baseline isolates whether polarity adds anything beyond raw attention/return
    feedback.

    The mention counts are passed through ``log1p`` (a monotone, variance-stabilizing
    transform) before the *same* train-only standardize / threshold / ``shift(lag)``
    pipeline the sentiment signal uses, so the only difference between the two books
    is the underlying field (attention magnitude vs. polarity).

    Parameters
    ----------
    mention_count:
        Wide ``day x ticker`` panel of per-day mention counts.
    spec:
        The signal configuration (window, lag, threshold, long_only).
    train_end:
        The last in-sample date (inclusive) for the train-only standardizer.

    Returns
    -------
    pandas.DataFrame
        A wide ``day x ticker`` panel of attention-only positions (already
        ``shift(spec.lag)``-ed).

    Raises
    ------
    ValidationError
        If ``mention_count`` cannot be coerced to a 2-D panel.
    """
    mentions = _coerce_panel(mention_count, name="mention_count")
    # log1p is monotone in attention and tames the Poisson tail; it never changes
    # the *ranking* of busy vs. quiet days, only the standardized scale. Rebuild as
    # a DataFrame so the static type stays a DataFrame (np.log1p loses it).
    clipped = mentions.clip(lower=0.0)
    log_mentions = pd.DataFrame(
        np.log1p(clipped.to_numpy()), index=clipped.index, columns=clipped.columns
    )
    return _standardize_threshold_shift(log_mentions, spec, train_end=train_end)


def _oos_index(index: pd.Index, train_end: pd.Timestamp, *, gap: int) -> pd.Index:
    """Return the post-purge/embargo out-of-sample index (shared by all strategies).

    The in-sample slice is every date ``<= train_end``. The out-of-sample slice is
    every date strictly after ``train_end``, with the first ``gap = purge +
    embargo`` observations dropped: the purge removes the single boundary
    observation whose position was decided using the last in-sample sentiment, and
    the embargo (= the daily return horizon) opens the gap before the first
    independently-earned OOS return. This is the IDENTICAL index every strategy is
    scored on.
    """
    sorted_index = index.sort_values()
    post = sorted_index[sorted_index > train_end]
    if gap > 0:
        post = post[gap:]
    return post


def run_signal_backtest(
    positions: SentimentLike,
    returns: ReturnsLike,
    *,
    mention_count: SentimentLike,
    spec: SignalSpec,
    train_end: pd.Timestamp,
    cost_bps: float = 10.0,
    embargo: int = 1,
    purge: int = 1,
    universe_mask: pd.DataFrame | None = None,
) -> SignalBacktestResult:
    """Run the anchored sentiment-signal backtest vs buy-hold and attention-only.

    Aligns the (already ``shift``-ed) ``positions`` to FORWARD returns, restricts
    the traded book to the point-in-time ``universe_mask`` when given, charges a
    per-side ``cost_bps`` transaction cost on daily turnover, and computes the net
    and gross OOS series alongside the buy-and-hold and attention-only baselines —
    all three reported on the IDENTICAL post-purge/embargo OOS index.

    NO-LOOKAHEAD: this function assumes ``positions`` were produced with
    ``shift(lag)`` and a train-only scaler; it never refits anything on the OOS
    slice and never earns a same-bar return. The PIT ``universe_mask`` is applied
    per-day so a symbol absent from the as-of universe is never traded.

    Parameters
    ----------
    positions:
        Wide ``day x ticker`` per-ticker positions (already lagged).
    returns:
        Wide ``day x ticker`` forward simple returns.
    mention_count:
        Wide ``day x ticker`` mention counts (for the attention-only baseline).
    spec:
        The signal configuration.
    train_end:
        The in-sample/out-of-sample boundary date.
    cost_bps:
        Per-side transaction cost in basis points (``>= 0``).
    embargo, purge:
        No-lookahead boundary guards (in trading days).
    universe_mask:
        Optional point-in-time boolean ``day x ticker`` tradability mask; ``False``
        cells are excluded from the traded book (meme tickers stay descriptive).

    Returns
    -------
    SignalBacktestResult
        The net/gross OOS series, baselines, turnover, and metadata.

    Raises
    ------
    ValidationError
        If ``cost_bps``/``embargo``/``purge`` are out of range or inputs are
        malformed.
    InsufficientDataError
        If no out-of-sample observation survives the purge/embargo gap.
    """
    from wsb_sentiment._exceptions import InsufficientDataError, ValidationError
    from wsb_sentiment.backtest.costs import FixedBpsCost

    # --- Validate scalar parameters ---------------------------------------
    if not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValidationError(
            f"run_signal_backtest: cost_bps must be finite and >= 0, got {cost_bps}."
        )
    if embargo < 0 or purge < 0:
        raise ValidationError(
            f"run_signal_backtest: purge and embargo must be >= 0, got "
            f"purge={purge}, embargo={embargo}."
        )

    cost_model = FixedBpsCost(bps=float(cost_bps))

    # --- Coerce + align every panel on the SHARED (day x ticker) grid ------
    pos = _coerce_panel(positions, name="positions")
    ret = _coerce_panel(returns, name="returns")
    mentions = _coerce_panel(mention_count, name="mention_count")

    # Common, sorted dates and tickers across all panels: aligning here is what
    # guarantees the three strategies share a single OOS index and column set.
    common_idx = pos.index.intersection(ret.index).intersection(mentions.index)
    if len(common_idx) == 0:
        raise ValidationError(
            "run_signal_backtest: positions, returns, and mention_count share no common dates."
        )
    common_cols = pos.columns.intersection(ret.columns).intersection(mentions.columns)
    if len(common_cols) == 0:
        raise ValidationError(
            "run_signal_backtest: positions, returns, and mention_count share no common tickers."
        )
    common_idx = common_idx.sort_values()
    pos = pos.reindex(index=common_idx, columns=common_cols)
    ret = ret.reindex(index=common_idx, columns=common_cols).fillna(0.0)
    mentions = mentions.reindex(index=common_idx, columns=common_cols)

    # --- Resolve the point-in-time tradability mask ------------------------
    # Default-True (everything tradable) when no mask is supplied; otherwise any
    # date/ticker absent from the as-of universe is False (never traded).
    if universe_mask is None:
        mask = pd.DataFrame(True, index=common_idx, columns=common_cols)
    else:
        mask = universe_mask.reindex(index=common_idx, columns=common_cols).fillna(False)
        mask = mask.astype(bool)

    # --- Out-of-sample index (shared by signal + both baselines) -----------
    gap = int(purge) + int(embargo)
    oos_idx = _oos_index(common_idx, pd.Timestamp(train_end), gap=gap)
    if len(oos_idx) == 0:
        raise InsufficientDataError(
            "run_signal_backtest: no out-of-sample observation survives the "
            f"purge+embargo gap of {gap} after train_end={train_end!r}."
        )

    # --- Sentiment signal book: PIT-masked, L1-normalized weights ----------
    sig_weights, _ = _row_normalized_positions(pos, mask)
    gross_signal = (sig_weights * ret).sum(axis=1)

    # Per-day one-way turnover of the traded book (0.5 * sum |Δw|); the first row
    # trades in from flat. Costs are charged on that day's gross return.
    prev = sig_weights.shift(1).fillna(0.0)
    turnover_full = 0.5 * (sig_weights - prev).abs().sum(axis=1)
    cost_full = turnover_full.map(cost_model.cost)
    net_signal = gross_signal - cost_full

    # --- Buy-and-hold baseline: equal-weight long over the PIT universe ----
    # Each day, hold an equal-weight long book across exactly the tradable names
    # (mask True); descriptive-only tickers contribute nothing.
    bh_active = mask.astype("float64")
    bh_count = bh_active.sum(axis=1)
    bh_scale = bh_count.where(bh_count > 0.0, other=1.0)
    bh_weights = bh_active.div(bh_scale, axis=0)
    gross_buyhold = (bh_weights * ret).sum(axis=1)

    # --- Attention-only baseline: positions from mention counts alone ------
    attn_pos = attention_only_positions(mentions, spec, train_end=pd.Timestamp(train_end))
    attn_pos = attn_pos.reindex(index=common_idx, columns=common_cols)
    attn_weights, _ = _row_normalized_positions(attn_pos, mask)
    gross_attention = (attn_weights * ret).sum(axis=1)

    # --- Restrict every series to the IDENTICAL OOS index ------------------
    net_oos = net_signal.reindex(oos_idx).astype("float64")
    gross_oos = gross_signal.reindex(oos_idx).astype("float64")
    buyhold_oos = gross_buyhold.reindex(oos_idx).astype("float64")
    attention_oos = gross_attention.reindex(oos_idx).astype("float64")
    turnover_oos = turnover_full.reindex(oos_idx).astype("float64")

    meta: dict[str, Any] = {
        "cost_bps": float(cost_bps),
        "embargo": int(embargo),
        "purge": int(purge),
        "n_assets": len(common_cols),
        "n_oos": len(oos_idx),
        "train_end": pd.Timestamp(train_end).isoformat(),
        "oos_start": oos_idx[0].isoformat(),
        "oos_end": oos_idx[-1].isoformat(),
        "spec": spec.to_dict(),
        "pit_masked": universe_mask is not None,
    }

    return SignalBacktestResult(
        net_returns=net_oos,
        gross_returns=gross_oos,
        buyhold_returns=buyhold_oos,
        attention_returns=attention_oos,
        turnover=turnover_oos,
        cost_bps=float(cost_bps),
        n_oos=len(oos_idx),
        meta=meta,
    )
