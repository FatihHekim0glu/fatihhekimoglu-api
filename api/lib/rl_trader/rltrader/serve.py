"""Serve entrypoint the backend calls (onnxruntime + numpy, NO torch / sb3 / gymnasium).

The FastAPI router calls :func:`run_backtest` to evaluate the committed PPO policy
+ the live baselines on the synthetic price path (or a real single-asset basket
loaded via the CLI path) and return a JSON-safe summary plus two Plotly figures.
The buy-hold / flat / random baselines are computed LIVE (pure numpy); the RL agent
is served from its committed ONNX policy via onnxruntime — torch / sb3 / gymnasium
are NEVER imported. The honest ``rl_beats_baseline`` verdict is the PURE function of
the inference outputs (median-seed DM vs. buy-hold AND DSR>0 AND across-seed Sharpe
lower bound > 0).

The committed offline-trained metrics are read from ``artifacts/metrics.json`` so
the deployed default returns the precomputed per-seed OOS equity + the seed lottery
instantly without re-training; the request path NEVER trains.

Importing this module has no side effects (onnxruntime / plotly are imported lazily
inside the serve / figure paths).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rltrader._exceptions import ValidationError

if TYPE_CHECKING:
    from rltrader._typing import FloatArray

#: The package's committed-artifacts directory (ONNX policy + metrics.json).
_ARTIFACTS_DIR: Path = Path(__file__).resolve().parent / "artifacts"
#: Committed precomputed-metrics filename.
_METRICS_FILENAME: str = "metrics.json"
#: Honest multiplicity fallback for the DSR (#seeds x #HP configs) when the committed
#: metrics are absent; the exact count is committed by ``train.py``.
_DEFAULT_N_EFFECTIVE_TRIALS: int = 5
#: Per-trade slippage (bps) — kept in lock-step with the training cost model.
_SLIPPAGE_BPS: float = 1.0
#: The hard cap on ``n_seeds`` the router enforces (mirrored here defensively).
_MAX_SEEDS: int = 16
#: A synthetic path length comfortably longer than one OOS episode (warm-up + room).
_MIN_N_OBS: int = 64


@dataclass(frozen=True, slots=True)
class RlTraderSummary:
    """Immutable, JSON-safe summary of the RL-vs-baseline single-asset comparison.

    Attributes
    ----------
    oos_sharpe_rl_median:
        The MEDIAN-seed OOS net Sharpe of the PPO agent (net of costs + slippage).
    oos_sharpe_buyhold:
        The buy-and-hold OOS net Sharpe.
    oos_sharpe_flat:
        The flat-cash OOS net Sharpe (the zero-risk floor; ~0).
    seed_sharpe_lo:
        The across-seed OOS-Sharpe LOWER bound (the seed-lottery dispersion floor).
    seed_sharpe_hi:
        The across-seed OOS-Sharpe UPPER bound.
    dm_pvalue_vs_buyhold:
        The Diebold-Mariano p-value of the median-seed RL net return vs. buy-hold.
    deflated_sharpe:
        The Deflated Sharpe (honest seed x HP ``n_trials``) of the median-seed RL.
    turnover:
        The median-seed RL agent's total one-way turnover.
    max_drawdown:
        The median-seed RL agent's worst peak-to-trough drawdown (``<= 0``).
    rl_beats_baseline:
        The PURE verdict: ``True`` iff the median-seed beats buy-hold DM-significant
        AND DSR>0 AND the across-seed Sharpe lower bound > 0, net of costs.
    n_effective_trials:
        The honest multiplicity count used for the DSR (#seeds x #HP configs).
    data_source:
        Provenance of the input path (``"synthetic"`` / ``"polygon"``).
    """

    oos_sharpe_rl_median: float
    oos_sharpe_buyhold: float
    oos_sharpe_flat: float
    seed_sharpe_lo: float
    seed_sharpe_hi: float
    dm_pvalue_vs_buyhold: float
    deflated_sharpe: float
    turnover: float
    max_drawdown: float
    rl_beats_baseline: bool
    n_effective_trials: int
    data_source: str

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this summary."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RlTraderRun:
    """Immutable bundle returned to the backend: summary + two Plotly figures.

    Attributes
    ----------
    summary:
        The :class:`RlTraderSummary`.
    equity_figure:
        A Plotly ``{data, layout}`` dict: the RL median equity curve + buy-hold +
        the across-seed band.
    seed_lottery_figure:
        A Plotly ``{data, layout}`` dict: the OOS-Sharpe dispersion across seeds
        with the buy-hold Sharpe marked.
    """

    summary: RlTraderSummary
    equity_figure: dict[str, Any] = field(default_factory=dict)
    seed_lottery_figure: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this run."""
        return {
            "summary": self.summary.to_dict(),
            "equity_figure": self.equity_figure,
            "seed_lottery_figure": self.seed_lottery_figure,
        }


def run_backtest(
    *,
    n_seeds: int = 5,
    cost_bps: float = 5.0,
    lookback: int = 32,
    episode_len: int = 252,
    data_source_pref: str = "synthetic",
    seed: int = 7,
) -> RlTraderRun:
    """Run the end-to-end RL-vs-baseline backtest; return a JSON-safe summary + figures.

    Builds (or loads) the single-asset price path, computes the buy-hold / flat /
    random baselines LIVE (pure numpy) through the vectorized backtester, serves the
    committed PPO policy from its ONNX artifact (onnxruntime, NO torch) to produce
    the RL position sequence per seed, reads the committed per-seed OOS metrics +
    seed lottery from ``artifacts/metrics.json``, runs the Diebold-Mariano test of
    the median-seed RL net return vs. buy-hold, derives the PURE
    ``rl_beats_baseline`` verdict, and assembles the equity-curve + seed-lottery
    Plotly figures. NEVER trains on the request path.

    Parameters
    ----------
    n_seeds:
        Number of training seeds reflected in the seed lottery (capped by the
        router, e.g. <= 16).
    cost_bps:
        Per-side transaction cost in basis points.
    lookback:
        Observation look-back window length.
    episode_len:
        OOS episode length.
    data_source_pref:
        ``"synthetic"`` (default) or ``"auto"`` for the real PIT path.
    seed:
        Master RNG seed for the synthetic path.

    Returns
    -------
    RlTraderRun
        The summary and figures for the backend response.

    Raises
    ------
    ValidationError
        If the request is invalid (bad lookback / n_seeds / cost).
    ArtifactError
        If the committed ONNX policy cannot be loaded.
    """
    _validate_request(n_seeds=n_seeds, cost_bps=cost_bps, lookback=lookback, episode_len=episode_len)

    returns, data_source = _resolve_returns(
        episode_len=episode_len, data_source_pref=data_source_pref, seed=seed
    )

    # Baselines run LIVE (pure numpy) through the SHARED vectorized backtester so their
    # accounting is identical to the RL agent's. The RL agent is served from the
    # committed ONNX policy via onnxruntime (NEVER torch).
    baseline_nets = _live_baselines(returns, cost_bps=cost_bps, seed=seed)
    rl_net, rl_positions, rl_equity = _serve_rl_policy(
        returns, lookback=lookback, cost_bps=cost_bps
    )

    committed = _read_committed_metrics()
    summary = _build_summary(
        rl_net=rl_net,
        rl_positions=rl_positions,
        baseline_nets=baseline_nets,
        committed=committed,
        n_seeds=n_seeds,
        data_source=data_source,
    )
    equity_fig, lottery_fig = _build_figures(
        rl_equity=rl_equity,
        buyhold_net=baseline_nets["buy_hold"],
        committed=committed,
        summary=summary,
    )
    return RlTraderRun(
        summary=summary,
        equity_figure=equity_fig,
        seed_lottery_figure=lottery_fig,
    )


def _validate_request(
    *, n_seeds: int, cost_bps: float, lookback: int, episode_len: int
) -> None:
    """Validate the request scalars (the router's field validators, enforced here too)."""
    if n_seeds < 1 or n_seeds > _MAX_SEEDS:
        raise ValidationError(f"n_seeds must be in [1, {_MAX_SEEDS}], got {n_seeds}.")
    if not math.isfinite(cost_bps) or cost_bps < 0.0:
        raise ValidationError(f"cost_bps must be finite and >= 0, got {cost_bps}.")
    if lookback < 1:
        raise ValidationError(f"lookback must be >= 1, got {lookback}.")
    if episode_len < 1:
        raise ValidationError(f"episode_len must be >= 1, got {episode_len}.")


def _resolve_returns(
    *, episode_len: int, data_source_pref: str, seed: int
) -> tuple[FloatArray, str]:
    """Build the per-bar return path + provenance (synthetic default; Polygon PIT path).

    The deployed default is the seeded GBM-regime synthetic path (no key / network).
    ``data_source_pref='auto'`` attempts the real Polygon PIT bars and falls straight
    through to the synthetic path on any failure (the loader does this internally).
    """
    n_obs = max(_MIN_N_OBS, int(episode_len) * 4)
    if data_source_pref == "auto":
        from datetime import date

        from rltrader.data.loaders import load_single_asset_bars

        _, returns_series, source = load_single_asset_bars(
            "SPY",
            start=date(2010, 1, 1),
            end=date(2020, 1, 1),
            data_source_pref="polygon",
            seed=seed,
        )
    else:
        from rltrader.data.loaders import synthetic_default_path

        _, returns_series, source = synthetic_default_path(
            n_obs=n_obs, seed=seed, kind="gbm_regime"
        )
    return np.asarray(returns_series.to_numpy(), dtype="float64"), str(source)


def _live_baselines(
    returns: FloatArray, *, cost_bps: float, seed: int
) -> dict[str, FloatArray]:
    """Run the buy-hold / flat / random baselines LIVE and return their net-return series."""
    from rltrader.agents.baselines import run_baseline

    nets: dict[str, FloatArray] = {}
    for name in ("buy_hold", "flat_cash", "random_action"):
        result = run_baseline(
            name, returns, cost_bps=cost_bps, slippage_bps=_SLIPPAGE_BPS, seed=seed
        )
        nets[name] = np.asarray(result.net_returns, dtype="float64")
    return nets


def _serve_rl_policy(
    returns: FloatArray, *, lookback: int, cost_bps: float
) -> tuple[FloatArray | None, FloatArray | None, FloatArray | None]:
    """Serve the committed ONNX policy on the OOS path (onnxruntime, NEVER torch).

    Builds the look-ahead-safe observation matrix (data <= t only, the env's
    construction), maps it to per-bar positions via the committed ``policy.onnx``,
    and scores them through the SHARED vectorized backtester aligned to the same
    window as the baselines. Returns ``(None, None, None)`` when the committed
    artifact is absent or unloadable (the policy is a separate offline deliverable).
    """
    from rltrader._exceptions import ArtifactError
    from rltrader.agents.onnx_policy import OnnxPolicy, default_artifact_path
    from rltrader.env.backtester import vectorized_backtest

    if not default_artifact_path().is_file():
        return None, None, None

    observations = _observation_matrix(returns, lookback=lookback)
    if observations.shape[0] == 0:
        return None, None, None

    try:
        decisions = OnnxPolicy().predict_positions(observations)
    except ArtifactError:
        return None, None, None

    positions = np.zeros(returns.size, dtype="float64")
    start = lookback - 1
    positions[start : start + decisions.size] = decisions
    result = vectorized_backtest(
        returns,
        positions,
        cost_bps=cost_bps,
        slippage_bps=_SLIPPAGE_BPS,
        initial_position=0.0,
    )
    return (
        np.asarray(result.net_returns, dtype="float64"),
        np.asarray(result.positions, dtype="float64"),
        np.asarray(result.equity_curve, dtype="float64"),
    )


def _observation_matrix(returns: FloatArray, *, lookback: int) -> FloatArray:
    """Build the per-bar observation matrix the env emits (data <= t only, position flat).

    Mirrors :meth:`rltrader.env.trading_env.TradingEnv._observe`: each scored bar
    ``t in [lookback-1, N-2]`` gets the trailing return window concatenated with the
    flat current position. NEVER reads ``r_{t+1}`` or beyond (strict causality).
    """
    r = np.asarray(returns, dtype="float64").ravel()
    n = r.size
    start = lookback - 1
    rows: list[FloatArray] = []
    for t in range(start, n - 1):
        window = r[t - lookback + 1 : t + 1]
        rows.append(np.concatenate((window, np.array([0.0], dtype="float64"))))
    if not rows:
        return np.empty((0, lookback + 1), dtype="float64")
    return np.asarray(rows, dtype="float64")


def _build_summary(
    *,
    rl_net: FloatArray | None,
    rl_positions: FloatArray | None,
    baseline_nets: dict[str, FloatArray],
    committed: dict[str, Any],
    n_seeds: int,
    data_source: str,
) -> RlTraderSummary:
    """Assemble the JSON-safe summary + the PURE ``rl_beats_baseline`` verdict.

    HONEST DEFAULT: when the committed offline ``metrics.json`` is present, the
    RL-side scalars (median-seed OOS Sharpe, the across-seed seed-lottery band, the
    Diebold-Mariano-vs-buy-hold, the Deflated Sharpe, turnover, drawdown, and the
    PURE verdict) come from that precomputed OOS evaluation — the deployed default
    returns the committed honest result, it does NOT re-derive an edge from an
    arbitrarily-sized live path. The baselines are always computed LIVE. Only when no
    committed metrics exist does the summary fall back to the live-served ONNX policy
    (or, absent even that, the honest-null verdict ``False``).
    """
    from rltrader.evaluation.metrics import oos_sharpe

    bh_sharpe = _safe_float(oos_sharpe(baseline_nets["buy_hold"]))
    flat_sharpe = _safe_float(oos_sharpe(baseline_nets["flat_cash"]))

    # Committed offline metrics present: return the precomputed honest OOS result.
    if committed.get("oos_sharpe_rl_median") is not None:
        return RlTraderSummary(
            oos_sharpe_rl_median=_safe_float(committed["oos_sharpe_rl_median"]),
            oos_sharpe_buyhold=bh_sharpe,
            oos_sharpe_flat=flat_sharpe,
            seed_sharpe_lo=_safe_float(committed.get("seed_sharpe_lo", 0.0)),
            seed_sharpe_hi=_safe_float(committed.get("seed_sharpe_hi", 0.0)),
            dm_pvalue_vs_buyhold=_safe_float(committed.get("dm_pvalue_vs_buyhold", 1.0)),
            deflated_sharpe=_safe_float(committed.get("deflated_sharpe", 0.0)),
            turnover=_safe_float(committed.get("turnover", 0.0)),
            max_drawdown=_safe_float(committed.get("max_drawdown", 0.0)),
            rl_beats_baseline=bool(committed.get("rl_beats_baseline", False)),
            n_effective_trials=int(committed.get("n_effective_trials", n_seeds)),
            data_source=data_source,
        )

    # No committed metrics: fall back to the live-served ONNX policy (a single-path
    # comparison) or, absent even that, the honest-null verdict (no narrated edge).
    return _live_summary(
        rl_net=rl_net,
        rl_positions=rl_positions,
        bh_net=baseline_nets["buy_hold"],
        bh_sharpe=bh_sharpe,
        flat_sharpe=flat_sharpe,
        n_seeds=n_seeds,
        data_source=data_source,
    )


def _live_summary(
    *,
    rl_net: FloatArray | None,
    rl_positions: FloatArray | None,
    bh_net: FloatArray,
    bh_sharpe: float,
    flat_sharpe: float,
    n_seeds: int,
    data_source: str,
) -> RlTraderSummary:
    """Build the summary from the live-served ONNX policy (no committed metrics path).

    The committed offline metrics are the deployed default; this live fallback covers
    the bare-package state (no ``metrics.json`` committed yet). With a single served
    policy the seed lottery collapses onto the one OOS Sharpe — which CANNOT clear the
    strictly-positive-lower-bound gate unless that Sharpe is itself ``> 0`` — so the
    verdict stays honest. Absent even a served policy the verdict is ``False``.
    """
    from rltrader._constants import PERIODS_PER_YEAR
    from rltrader.evaluation.metrics import max_drawdown, oos_sharpe, turnover
    from rltrader.evaluation.verdict import derive_verdict

    n_trials = max(1, n_seeds)
    if rl_net is None or rl_positions is None:
        return RlTraderSummary(
            oos_sharpe_rl_median=0.0,
            oos_sharpe_buyhold=bh_sharpe,
            oos_sharpe_flat=flat_sharpe,
            seed_sharpe_lo=0.0,
            seed_sharpe_hi=0.0,
            dm_pvalue_vs_buyhold=1.0,
            deflated_sharpe=0.0,
            turnover=0.0,
            max_drawdown=0.0,
            rl_beats_baseline=False,
            n_effective_trials=n_trials,
            data_source=data_source,
        )

    rl_sharpe = oos_sharpe(rl_net)
    # A single served Sharpe is the (degenerate) seed band: lo == hi == that Sharpe.
    seed_point = _safe_float(rl_sharpe)
    dm_stat, dm_pvalue = _dm_or_null(rl_net, bh_net)
    deflated = _deflated_or_zero(rl_sharpe, rl_net.size, n_trials, 0.0, PERIODS_PER_YEAR)
    verdict = derive_verdict(
        dm_statistic=dm_stat,
        dm_pvalue=dm_pvalue,
        deflated_sharpe=deflated,
        seed_sharpe_lo=seed_point,
        n_effective_trials=n_trials,
    )
    return RlTraderSummary(
        oos_sharpe_rl_median=seed_point,
        oos_sharpe_buyhold=bh_sharpe,
        oos_sharpe_flat=flat_sharpe,
        seed_sharpe_lo=seed_point,
        seed_sharpe_hi=seed_point,
        dm_pvalue_vs_buyhold=_safe_float(dm_pvalue),
        deflated_sharpe=_safe_float(deflated),
        turnover=_safe_float(turnover(rl_positions)),
        max_drawdown=_safe_float(max_drawdown(rl_net)),
        rl_beats_baseline=bool(verdict.rl_beats_baseline),
        n_effective_trials=n_trials,
        data_source=data_source,
    )


def _build_figures(
    *,
    rl_equity: FloatArray | None,
    buyhold_net: FloatArray,
    committed: dict[str, Any],
    summary: RlTraderSummary,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the equity-curve + seed-lottery Plotly figures (lazy plotly).

    The equity figure overlays the RL median equity (committed precomputed curve when
    present, else the live-served curve) + buy-hold + the across-seed band; the
    seed-lottery figure shows the committed per-seed OOS-Sharpe dispersion with the
    buy-hold Sharpe marked. Both are empty ``{}`` when their inputs are unavailable.
    """
    from rltrader.env.backtester import equity_curve
    from rltrader.plots import equity_curve_figure, seed_lottery_figure

    bh_equity = equity_curve(np.asarray(buyhold_net, dtype="float64"))

    # Prefer the committed precomputed RL median equity (the honest offline curve);
    # fall back to the live-served curve, length-aligned to buy-hold.
    committed_equity = committed.get("rl_median_equity")
    rl_curve: FloatArray | None
    if committed_equity:
        rl_curve = np.asarray(committed_equity, dtype="float64")
    elif rl_equity is not None:
        rl_curve = np.asarray(rl_equity, dtype="float64")
    else:
        rl_curve = None

    equity_fig: dict[str, Any] = {}
    if rl_curve is not None and rl_curve.size:
        rl_curve, bh_aligned, band = _align_equity(rl_curve, bh_equity, committed)
        equity_fig = equity_curve_figure(
            rl_curve,
            bh_aligned,
            seed_band_lo=band[0] if band is not None else None,
            seed_band_hi=band[1] if band is not None else None,
        )

    lottery_fig: dict[str, Any] = {}
    seed_sharpes = committed.get("seed_sharpes")
    if seed_sharpes:
        lottery_fig = seed_lottery_figure(
            np.asarray(seed_sharpes, dtype="float64"),
            buyhold_sharpe=summary.oos_sharpe_buyhold,
        )
    return equity_fig, lottery_fig


def _align_equity(
    rl_curve: FloatArray, bh_equity: FloatArray, committed: dict[str, Any]
) -> tuple[FloatArray, FloatArray, tuple[FloatArray, FloatArray] | None]:
    """Trim the RL median curve, buy-hold curve, and seed band to a common length."""
    band_lo = committed.get("seed_band_lo")
    band_hi = committed.get("seed_band_hi")
    lengths = [rl_curve.size, bh_equity.size]
    if band_lo and band_hi:
        lengths.extend([len(band_lo), len(band_hi)])
    n = min(lengths)
    rl = rl_curve[:n]
    bh = bh_equity[:n]
    band: tuple[FloatArray, FloatArray] | None = None
    if band_lo and band_hi:
        band = (
            np.asarray(band_lo, dtype="float64")[:n],
            np.asarray(band_hi, dtype="float64")[:n],
        )
    return rl, bh, band


def _dm_or_null(rl_net: FloatArray, buyhold_net: FloatArray) -> tuple[float, float]:
    """Run Diebold-Mariano, returning the honest null ``(0.0, 1.0)`` on a degeneracy."""
    from rltrader.evaluation.diebold_mariano import diebold_mariano

    try:
        stat, pvalue = diebold_mariano(rl_net, buyhold_net)
    except ValidationError:
        return 0.0, 1.0
    return float(stat), float(pvalue)


def _deflated_or_zero(
    observed_sharpe: float,
    n_obs: int,
    n_trials: int,
    variance: float,
    periods_per_year: int,
) -> float:
    """Compute the Deflated Sharpe; floor a NaN observed Sharpe at ``0.0`` (the DSR floor)."""
    from rltrader.evaluation.dsr import deflated_sharpe_ratio

    if not math.isfinite(observed_sharpe) or n_obs < 2:
        return 0.0
    per_obs = observed_sharpe / math.sqrt(periods_per_year)
    return deflated_sharpe_ratio(
        per_obs,
        n_obs=n_obs,
        n_trials=max(1, n_trials),
        variance_of_trial_sharpes=max(0.0, variance),
    )


def _read_committed_metrics() -> dict[str, Any]:
    """Load the committed ``artifacts/metrics.json`` (the offline seed-lottery / DSR / trials).

    Returns an empty dict when the artifact has not been committed yet (the live
    comparison then supplies its own fallback multiplicity count) or is unreadable.
    """
    path = _ARTIFACTS_DIR / _METRICS_FILENAME
    if not path.is_file():
        return {}
    try:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # pragma: no cover - defensive against a corrupt file
        return {}
    return payload


def _safe_float(value: Any) -> float:
    """Coerce a scalar to a finite ``float`` (NaN/Inf -> 0.0) for JSON safety."""
    import math

    out = float(value)
    return out if math.isfinite(out) else 0.0
