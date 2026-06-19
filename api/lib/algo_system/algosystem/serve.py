"""Serve entrypoint the backend calls (pure numpy/scipy/statsmodels, NO torch/onnx).

The FastAPI router calls :func:`run_system` to run the FULL pipeline on the
synthetic default (or a real single-asset basket loaded via the CLI path): a causal
signal -> a purged walk-forward backtest -> a simulated bar-by-bar paper-broker
replay -> the backtest<->live PARITY ORACLE -> OOS metrics + DM + DSR + PBO -> the
PURE ``system_has_edge`` verdict -> the equity + drawdown figures. Everything is
pure numpy/scipy/statsmodels — torch / onnx / onnxruntime / sklearn are NEVER
imported. The honest verdict is the PURE function of the inference outputs
(DM-vs-buy-hold AND DSR > 1-alpha AND PBO < 0.5, net of costs).

The deployed default runs the pipeline live on the cheap synthetic bars (vectorized
backtest + a fast paper-broker replay); the request path NEVER trains a heavy
model. Importing this module has no side effects (plotly is imported lazily inside
the figure path).

Honest headline: the backtest and the simulated live execution match to the cent
(the parity oracle passes) and the bar-finality guard holds; the strategy shows NO
robust edge after costs, a Deflated-Sharpe correction, and a PBO overfitting check
(``system_has_edge=False``).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from algosystem._exceptions import ValidationError
from algosystem._typing import FloatArray
from algosystem.backtest.bar_finality import BarStatus, check_finality
from algosystem.backtest.engine import vectorized_backtest, walk_forward_signal_backtest
from algosystem.data.loaders import load_single_asset_bars, synthetic_default_bars
from algosystem.evaluation.diebold_mariano import diebold_mariano
from algosystem.evaluation.dsr import deflated_sharpe_ratio
from algosystem.evaluation.metrics import strategy_metrics
from algosystem.evaluation.pbo import probability_of_backtest_overfitting
from algosystem.evaluation.verdict import system_has_edge
from algosystem.execution.paper_broker import PaperBrokerConfig, replay
from algosystem.execution.parity import assert_parity, check_parity
from algosystem.signals.library import SignalSpec, build_signal

if TYPE_CHECKING:
    import pandas as pd

#: The honest multiplicity grid (#signals x #param configs) evaluated for the PBO
#: matrix and the Deflated-Sharpe ``n_trials``. Counting the FULL grid (not just the
#: requested config) is the honest multiplicity correction; this mirrors the CLI
#: grid in :mod:`algosystem.cli` so the serve and CLI verdicts agree.
_CONFIG_GRID: tuple[SignalSpec, ...] = (
    SignalSpec("ma_crossover", {"fast": 5, "slow": 20}),
    SignalSpec("ma_crossover", {"fast": 10, "slow": 50}),
    SignalSpec("ma_crossover", {"fast": 20, "slow": 100}),
    SignalSpec("ma_crossover", {"fast": 10, "slow": 30}),
    SignalSpec("momentum", {"lookback": 10}),
    SignalSpec("momentum", {"lookback": 20}),
    SignalSpec("momentum", {"lookback": 40}),
)

#: Number of synthetic bars the serve path generates by default (mirrors the API +
#: CLI defaults).
_DEFAULT_N_OBS: int = 2000

#: CSCV split count for the PBO estimate (even, >= 2).
_PBO_SPLITS: int = 16


@dataclass(frozen=True, slots=True)
class AlgoSystemSummary:
    """Immutable, JSON-safe summary of the signal-vs-buy-hold single-asset comparison.

    Attributes
    ----------
    oos_sharpe:
        The strategy's OOS net Sharpe (net of costs + slippage).
    buyhold_sharpe:
        The buy-and-hold OOS net Sharpe.
    dm_pvalue_vs_buyhold:
        The Diebold-Mariano p-value of the strategy net return vs. buy-and-hold.
    deflated_sharpe:
        The Deflated Sharpe (honest #signals x #param-config ``n_trials``) — a
        probability in ``[0, 1]``.
    pbo:
        The Probability of Backtest Overfitting (CSCV), in ``[0, 1]``.
    backtest_live_parity_max_diff:
        The max abs per-bar diff between the backtest and paper-broker equity curves
        (the parity oracle; ``~0`` when they coincide).
    bar_finality_ok:
        ``True`` iff no order was attributed to a forming/partial bar.
    turnover:
        The strategy's total one-way turnover.
    max_drawdown:
        The strategy's worst peak-to-trough drawdown (``<= 0``).
    system_has_edge:
        The PURE verdict: ``True`` iff the strategy beats buy-hold DM-significant AND
        DSR > 1-alpha AND PBO < 0.5, net of costs.
    n_effective_trials:
        The honest multiplicity count used for the DSR (#signals x #param configs).
    data_source:
        Provenance of the input bars (``"synthetic"`` / ``"polygon"``).
    """

    oos_sharpe: float
    buyhold_sharpe: float
    dm_pvalue_vs_buyhold: float
    deflated_sharpe: float
    pbo: float
    backtest_live_parity_max_diff: float
    bar_finality_ok: bool
    turnover: float
    max_drawdown: float
    system_has_edge: bool
    n_effective_trials: int
    data_source: str

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this summary."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AlgoSystemRun:
    """Immutable bundle returned to the backend: summary + two Plotly figures.

    Attributes
    ----------
    summary:
        The :class:`AlgoSystemSummary`.
    equity_figure:
        A Plotly ``{data, layout}`` dict: the backtest equity overlaid on the
        paper-broker "live" equity + buy-hold (they should visually COINCIDE).
    drawdown_figure:
        A Plotly ``{data, layout}`` dict: the strategy drawdown curve.
    """

    summary: AlgoSystemSummary
    equity_figure: dict[str, Any] = field(default_factory=dict)
    drawdown_figure: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this run."""
        return {
            "summary": self.summary.to_dict(),
            "equity_figure": self.equity_figure,
            "drawdown_figure": self.drawdown_figure,
        }


def _selected_spec(signal: str, fast: int, slow: int, lookback: int) -> SignalSpec:
    """Build the SELECTED :class:`SignalSpec` from the request parameters.

    The request exposes only ``ma_crossover`` / ``momentum``; ``flat`` is the
    internal baseline and is never selectable from the API.
    """
    if signal == "ma_crossover":
        return SignalSpec("ma_crossover", {"fast": fast, "slow": slow})
    if signal == "momentum":
        return SignalSpec("momentum", {"lookback": lookback})
    raise ValidationError(f"signal must be 'ma_crossover' or 'momentum', got {signal!r}.")


def _align_positions(positions: FloatArray, n_returns: int) -> FloatArray:
    """Align a per-bar position vector (one per close) to the return path length.

    The close series has ``N`` bars; :func:`algosystem.data.compute_returns` drops
    the first (NaN) return, so the per-bar return path has ``N - 1`` entries where
    ``returns[i]`` is the ``close[i] -> close[i+1]`` return. Dropping the first
    position keeps the two vectors the same length and the next-bar-fill causality
    intact (mirrors :func:`algosystem.cli._align_positions`).
    """
    pos = np.asarray(positions, dtype="float64").ravel()
    aligned = pos[pos.size - n_returns :]
    if aligned.size != n_returns:  # pragma: no cover - defensive: lengths always match
        raise ValidationError(
            f"_align_positions: cannot align {pos.size} positions to {n_returns} returns."
        )
    return aligned


def _per_obs_sharpe(net_returns: FloatArray) -> float:
    """Per-observation (non-annualized) Sharpe for the DSR / PBO ranking.

    Mean over the sample standard deviation (``ddof=1``); a numerically-flat series
    has an undefined Sharpe and returns ``0.0`` (mirrors
    :func:`algosystem.cli._per_obs_sharpe`).
    """
    arr = np.asarray(net_returns, dtype="float64").ravel()
    if arr.size < 2:
        return 0.0
    sigma = float(np.std(arr, ddof=1))
    if sigma <= 0.0:
        return 0.0
    return float(np.mean(arr)) / sigma


def _sample_moments(net_returns: FloatArray) -> tuple[float, float]:
    """Return the sample skewness + FULL (non-excess) kurtosis of a net-return series.

    The DSR's PSR bracket uses the FULL kurtosis (Gaussian = 3), so a flat / tiny
    series falls back to the Gaussian ``(0.0, 3.0)`` moments. Uses scipy lazily (the
    ``data`` extra) so importing this module stays light.
    """
    arr = np.asarray(net_returns, dtype="float64").ravel()
    if arr.size < 3 or float(np.std(arr, ddof=1)) <= 0.0:
        return 0.0, 3.0
    from scipy import stats

    skew = float(stats.skew(arr))
    kurtosis = float(stats.kurtosis(arr, fisher=False))  # FULL kurtosis (Gaussian = 3).
    return skew, kurtosis


def _safe_float(value: object) -> float:
    """Coerce ``value`` to a finite float, mapping NaN/Inf/None to ``0.0``.

    The serve summary is JSON-bound, so a NaN OOS Sharpe (a numerically-flat series)
    or an Inf must never leak across the API boundary; they are clamped to ``0.0``
    so the response is always valid JSON. The verdict itself is computed from the
    raw statistics BEFORE this clamp, so the honest gate is never softened by it.
    """
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(out):
        return 0.0
    return out


def run_system(
    *,
    signal: str = "ma_crossover",
    fast: int = 10,
    slow: int = 50,
    lookback: int = 20,
    cost_bps: float = 5.0,
    slippage_bps: float = 2.0,
    data_source_pref: str = "synthetic",
    seed: int = 7,
) -> AlgoSystemRun:
    """Run the end-to-end pipeline; return a JSON-safe summary + figures.

    Builds (or loads) the single-asset OHLC bars, computes the causal signal's
    target positions, runs them through the purged walk-forward vectorized backtest
    AND the simulated paper-broker replay, ASSERTS backtest<->live parity (the
    oracle), computes the buy-and-hold baseline, runs the Diebold-Mariano test, the
    Deflated Sharpe (honest #signals x #param-config ``n_trials``), and the PBO/CSCV,
    derives the PURE ``system_has_edge`` verdict, and assembles the
    backtest-vs-live equity + drawdown Plotly figures. NEVER trains on the request
    path; everything is pure numpy/scipy/statsmodels.

    Parameters
    ----------
    signal:
        ``"ma_crossover"`` (default) or ``"momentum"``.
    fast:
        The fast MA window (for ``ma_crossover``; must be ``< slow``).
    slow:
        The slow MA window (for ``ma_crossover``).
    lookback:
        The momentum lookback (for ``momentum``).
    cost_bps:
        Per-side transaction cost in basis points.
    slippage_bps:
        Per-trade slippage in basis points.
    data_source_pref:
        ``"synthetic"`` (default) or ``"auto"`` for the real PIT path.
    seed:
        Master RNG seed for the synthetic path.

    Returns
    -------
    AlgoSystemRun
        The summary and figures for the backend response.

    Raises
    ------
    ValidationError
        If the request is invalid (e.g. ``fast >= slow``, negative cost).
    ParityError
        If the backtest<->live parity oracle fails (a look-ahead / fill-timing bug).
    """
    if not math.isfinite(cost_bps) or cost_bps < 0.0:
        raise ValidationError(f"cost_bps must be finite and >= 0, got {cost_bps!r}.")
    if not math.isfinite(slippage_bps) or slippage_bps < 0.0:
        raise ValidationError(f"slippage_bps must be finite and >= 0, got {slippage_bps!r}.")
    selected = _selected_spec(signal, fast, slow, lookback)

    # 1) DATA. The deployed default is the seeded synthetic GBM-regime bars (the
    # honest-null DGP) — no key, no network. ``data_source_pref="auto"`` routes to
    # the offline PIT loader, which itself falls back to synthetic when no Polygon
    # key / network is available, so the request path is always offline-safe.
    bars, returns, data_source = _load_bars(data_source_pref=data_source_pref, seed=seed)
    close: pd.Series = bars["close"]
    ret = np.asarray(returns.to_numpy(dtype="float64"), dtype="float64")
    n_ret = ret.size

    # 2) SIGNAL GRID (honest multiplicity). Evaluate the FULL #signals x #param grid
    # through the strictly-causal vectorized backtester, collecting each config's
    # per-bar net returns (for the PBO matrix + trial-Sharpe variance) and the
    # SELECTED config's net returns + positions. The selected config is appended
    # (de-duplicated) so the DSR n_trials honestly counts it.
    grid: list[SignalSpec] = list(_CONFIG_GRID)
    if selected not in grid:
        grid.append(selected)

    net_columns: list[FloatArray] = []
    trial_sharpes: list[float] = []
    selected_net: FloatArray | None = None
    selected_positions: FloatArray | None = None
    for spec in grid:
        positions = _align_positions(build_signal(spec, close), n_ret)
        result = vectorized_backtest(ret, positions, cost_bps=cost_bps, slippage_bps=slippage_bps)
        net_columns.append(result.net_returns)
        trial_sharpes.append(_per_obs_sharpe(result.net_returns))
        if spec == selected:
            selected_net = result.net_returns
            selected_positions = result.positions
    assert selected_net is not None  # the selected spec is always in the grid.
    assert selected_positions is not None

    selected_pos_full = _align_positions(build_signal(selected, close), n_ret)

    # 3) PARITY ORACLE (the load-bearing backtest<->live look-ahead catch). Replay
    # the SELECTED config bar by bar through the simulated paper broker and ASSERT
    # the vectorized backtest equity curve equals the paper-broker equity curve to
    # 1e-10. ``assert_parity`` RAISES ``ParityError`` on any divergence (the serve
    # path must never silently ship a leaky curve); on success it returns the agreed
    # backtest equity curve. ``check_parity`` is also run to surface the max-diff.
    parity = check_parity(ret, selected_pos_full, cost_bps=cost_bps, slippage_bps=slippage_bps)
    backtest_equity = assert_parity(
        ret, selected_pos_full, cost_bps=cost_bps, slippage_bps=slippage_bps
    )
    # The paper-broker "live" curve (it coincides with the backtest to 1e-10, which
    # is exactly what the parity oracle just proved) — used for the overlay figure.
    live = replay(
        ret,
        selected_pos_full,
        PaperBrokerConfig(cost_bps=cost_bps, slippage_bps=slippage_bps),
    )
    live_equity = np.asarray(live.equity_curve, dtype="float64").ravel()

    # 4) BAR-FINALITY GUARD (a REAL-DATA safeguard). The deployed synthetic /
    # committed feed contains ONLY finalized bars, so every decision bar is CLOSED
    # and ``bar_finality_ok`` is True by construction HERE. The guard's actual
    # teeth — rejecting an order triggered by a FORMING (partial) bar — are proven
    # by the ``guard_order`` / ``check_finality`` unit + property tests; on a live
    # feed with a still-forming last bar that bar would be marked FORMING and
    # surface ``bar_finality_ok=False`` rather than silently trading a partial bar.
    finality = check_finality([BarStatus.CLOSED] * live.n_bars)

    # 5) BUY-AND-HOLD BASELINE: a constant long position over the same path + the
    # same frictions (the bar the strategy must clear).
    buyhold_pos = np.ones(n_ret, dtype="float64")
    buyhold = vectorized_backtest(
        ret, buyhold_pos, cost_bps=cost_bps, slippage_bps=slippage_bps
    )

    # 5b) PURGED WALK-FORWARD OOS. The HEADLINE metrics (Sharpe/drawdown/turnover),
    # the Diebold-Mariano test and the DSR observed-Sharpe are computed on the
    # CONCATENATED purged-walk-forward OUT-OF-SAMPLE folds (purge>=1 boundary
    # observation + embargo=1 return horizon), NOT on the full in-sample path — so
    # ``oos_sharpe`` is genuinely out-of-sample. The selected config and the
    # buy-hold baseline are folded with IDENTICAL geometry, so their OOS net-return
    # paths align bar-for-bar for the DM differential. (The parity oracle above runs
    # on the full path because backtest<->live agreement is a fill-accounting
    # property, independent of the train/test folding.)
    wf_selected = walk_forward_signal_backtest(
        ret, selected_pos_full, cost_bps=cost_bps, slippage_bps=slippage_bps
    )
    wf_buyhold = walk_forward_signal_backtest(
        ret, buyhold_pos, cost_bps=cost_bps, slippage_bps=slippage_bps
    )

    # 6) METRICS (net of costs + slippage) for the strategy + the buy-hold baseline,
    # both on the purged-walk-forward OOS folds.
    metrics = strategy_metrics(wf_selected.net_returns, wf_selected.positions)
    buyhold_metrics = strategy_metrics(wf_buyhold.net_returns, wf_buyhold.positions)

    # 7) DIEBOLD-MARIANO of the strategy vs. buy-and-hold per-bar OOS net return.
    dm_statistic, dm_pvalue = diebold_mariano(wf_selected.net_returns, wf_buyhold.net_returns)

    # 8) DEFLATED SHARPE with the HONEST grid-wide n_trials + the selected config's
    # OOS per-obs Sharpe and sample moments; PBO/CSCV over the full grid's net-return
    # matrix (CSCV does its OWN in-sample/out-of-sample splitting, so it correctly
    # consumes the full-sample grid). The DSR is non-increasing in n_trials.
    n_trials = len(grid)
    sel_arr = np.asarray(wf_selected.net_returns, dtype="float64").ravel()
    skew, kurtosis = _sample_moments(sel_arr)
    var_trials = float(np.var(np.asarray(trial_sharpes, dtype="float64"), ddof=1))
    dsr = deflated_sharpe_ratio(
        _per_obs_sharpe(sel_arr),
        n_obs=int(sel_arr.size),
        n_trials=n_trials,
        variance_of_trial_sharpes=var_trials,
        skew=skew,
        kurtosis=kurtosis,
    )
    performance = np.column_stack(net_columns)
    pbo_result = probability_of_backtest_overfitting(performance, n_splits=_PBO_SPLITS)

    # 9) THE PURE VERDICT: system_has_edge True iff DM-significant AND DSR > 1-alpha
    # AND PBO < 0.5, all net of costs. Computed from the RAW statistics (before the
    # JSON-safe ``_safe_float`` clamp), so the honest gate is never softened. On the
    # synthetic null this is False (the documented honest-NULL outcome).
    verdict = system_has_edge(dm_statistic, dm_pvalue, dsr, pbo_result.pbo, n_trials)

    summary = AlgoSystemSummary(
        oos_sharpe=_safe_float(metrics.oos_sharpe),
        buyhold_sharpe=_safe_float(buyhold_metrics.oos_sharpe),
        dm_pvalue_vs_buyhold=_safe_float(dm_pvalue),
        deflated_sharpe=_safe_float(dsr),
        pbo=_safe_float(pbo_result.pbo),
        backtest_live_parity_max_diff=_safe_float(parity.max_abs_diff),
        bar_finality_ok=bool(finality.ok),
        turnover=_safe_float(metrics.turnover),
        max_drawdown=_safe_float(metrics.max_drawdown),
        system_has_edge=bool(verdict.system_has_edge),
        n_effective_trials=int(n_trials),
        data_source=str(data_source),
    )

    # 10) FIGURES (lazy plotly). The backtest + live curves coincide to 1e-10 (the
    # parity oracle), so the overlay visually proves the agreement; the drawdown
    # comes from the strategy net returns. Built last so a missing ``viz`` extra
    # only affects the figures, never the (already-computed) honest summary.
    equity_figure, drawdown_figure = _build_figures(
        backtest_equity=backtest_equity,
        live_equity=live_equity,
        buyhold_equity=np.asarray(buyhold.equity_curve, dtype="float64").ravel(),
        strategy_net_returns=sel_arr,
    )
    return AlgoSystemRun(
        summary=summary,
        equity_figure=equity_figure,
        drawdown_figure=drawdown_figure,
    )


def _load_bars(
    *, data_source_pref: str, seed: int
) -> tuple[pd.DataFrame, pd.Series, str]:
    """Load the request's OHLC bars + per-bar close returns and tag the provenance.

    The deployed default (``"synthetic"``) builds the seeded GBM-regime honest-null
    bars with no key / network. ``"auto"`` / ``"polygon"`` route through the offline
    PIT loader, which itself falls back to deterministic synthetic bars whenever a
    Polygon key / network / the ``data`` extra is absent, so the serve path is
    always offline-safe.
    """
    if data_source_pref in ("auto", "polygon"):
        from datetime import date

        # A fixed, deterministic single-asset span; the loader falls back to seeded
        # synthetic bars when Polygon is unavailable (the offline-safe request path).
        bars, returns, source = load_single_asset_bars(
            "SPY",
            start=date(2015, 1, 1),
            end=date(2023, 1, 1),
            data_source_pref="polygon" if data_source_pref == "polygon" else "synthetic",
            seed=seed,
        )
        return bars, returns, str(source)
    # ``"synthetic"`` (the deployed default): the honest-null GBM-regime DGP.
    bars, returns, source = synthetic_default_bars(
        n_obs=_DEFAULT_N_OBS, seed=seed, kind="gbm_regime"
    )
    return bars, returns, str(source)


def _build_figures(
    *,
    backtest_equity: FloatArray,
    live_equity: FloatArray,
    buyhold_equity: FloatArray,
    strategy_net_returns: FloatArray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Assemble the equity-overlay + drawdown Plotly figures (lazy plotly import).

    Imported here (the ``viz`` extra) so importing :mod:`algosystem.serve` pulls in
    nothing heavy. The backtest + live equity curves coincide to 1e-10 (the parity
    oracle), so the overlay visually proves the backtest<->live agreement; the
    drawdown comes from the strategy net returns.
    """
    from algosystem.plots import drawdown_figure, equity_overlay_figure

    equity = equity_overlay_figure(backtest_equity, live_equity, buyhold_equity)
    drawdown = drawdown_figure(strategy_net_returns)
    return equity, drawdown
