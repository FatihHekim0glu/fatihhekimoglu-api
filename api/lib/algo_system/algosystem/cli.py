"""Command-line interface (Typer): backtest / paper / compare on the algo-system pipeline.

The CLI is the OFFLINE entry point (the deployed default uses the FastAPI
``run_system`` path). Three commands:

- ``backtest`` — run the vectorized purged walk-forward backtest of a signal on the
  synthetic default (or real PIT bars via ``--data-source polygon``) and print the
  OOS metrics + the PURE verdict;
- ``paper`` — replay the same signal bar-by-bar through the simulated paper broker
  and print the "live" equity summary;
- ``compare`` — run BOTH paths and print the backtest<->live parity max-diff (the
  oracle), the metrics, the DM / DSR / PBO, and the PURE verdict.

Typer (the ``dev`` extra) is imported LAZILY inside :func:`_build_app` / :func:`main`
so importing this module has no side effects and pulls in nothing heavy. The full
pipeline is assembled HERE from the leakage-free primitives (data -> signal ->
vectorized backtest + paper-broker replay -> parity oracle -> metrics -> DM / DSR /
PBO -> PURE verdict); it does NOT depend on the serve-time ``run_system`` entry
point. Demos are behind the ``__main__`` guard. Importing this module has no side
effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from algosystem._exceptions import ValidationError
from algosystem._typing import FloatArray
from algosystem.backtest.engine import vectorized_backtest
from algosystem.data.loaders import synthetic_default_bars
from algosystem.evaluation.diebold_mariano import diebold_mariano
from algosystem.evaluation.dsr import deflated_sharpe_ratio
from algosystem.evaluation.metrics import strategy_metrics
from algosystem.evaluation.pbo import probability_of_backtest_overfitting
from algosystem.evaluation.verdict import VerdictResult, system_has_edge
from algosystem.execution.parity import check_parity
from algosystem.signals.library import SignalSpec, build_signal

if TYPE_CHECKING:
    import pandas as pd
    import typer

#: The honest multiplicity grid (#signals x #param configs) evaluated for the PBO
#: matrix and the Deflated-Sharpe ``n_trials``. Counting the FULL grid (not just the
#: requested config) is the honest multiplicity correction.
_CONFIG_GRID: tuple[SignalSpec, ...] = (
    SignalSpec("ma_crossover", {"fast": 5, "slow": 20}),
    SignalSpec("ma_crossover", {"fast": 10, "slow": 50}),
    SignalSpec("ma_crossover", {"fast": 20, "slow": 100}),
    SignalSpec("ma_crossover", {"fast": 10, "slow": 30}),
    SignalSpec("momentum", {"lookback": 10}),
    SignalSpec("momentum", {"lookback": 20}),
    SignalSpec("momentum", {"lookback": 40}),
)

#: Number of synthetic bars the CLI generates by default (mirrors the API default).
_DEFAULT_N_OBS: int = 2000

#: CSCV split count for the PBO estimate (even, >= 2).
_PBO_SPLITS: int = 16


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Immutable result of the offline CLI pipeline run (one selected config).

    Attributes
    ----------
    signal:
        The selected signal name.
    oos_sharpe:
        The selected strategy's OOS net Sharpe (net of costs + slippage).
    buyhold_sharpe:
        The buy-and-hold OOS net Sharpe.
    dm_statistic:
        The Diebold-Mariano statistic of the strategy net return vs. buy-and-hold
        (positive favours the strategy).
    dm_pvalue:
        The two-sided DM p-value.
    deflated_sharpe:
        The Deflated Sharpe (honest #signals x #param-config ``n_trials``).
    pbo:
        The Probability of Backtest Overfitting (CSCV).
    parity_max_diff:
        The max abs per-bar diff between the backtest and paper-broker equity curves
        (the parity oracle; ``~0`` when they coincide).
    parity_ok:
        ``True`` iff the parity oracle passed (backtest == live to ``1e-10``).
    turnover:
        The selected strategy's total one-way turnover.
    max_drawdown:
        The selected strategy's worst peak-to-trough drawdown (``<= 0``).
    n_trials:
        The honest multiplicity count (#signals x #param configs).
    data_source:
        Provenance of the input bars (``"synthetic"`` / ``"polygon"``).
    verdict:
        The PURE :class:`algosystem.evaluation.verdict.VerdictResult`.
    """

    signal: str
    oos_sharpe: float
    buyhold_sharpe: float
    dm_statistic: float
    dm_pvalue: float
    deflated_sharpe: float
    pbo: float
    parity_max_diff: float
    parity_ok: bool
    turnover: float
    max_drawdown: float
    n_trials: int
    data_source: str
    verdict: VerdictResult


def _per_obs_sharpe(net_returns: FloatArray) -> float:
    """Per-observation (non-annualized) Sharpe for the DSR / PBO ranking.

    Mean over the sample standard deviation (``ddof=1``); a numerically-flat series
    has an undefined Sharpe and returns ``0.0`` (it can never be the in-sample best
    and ranks at the bottom out-of-sample — the conservative, overfit-leaning
    choice that mirrors :func:`algosystem.evaluation.pbo._block_sharpe`).
    """
    arr = np.asarray(net_returns, dtype="float64").ravel()
    if arr.size < 2:
        return 0.0
    sigma = float(np.std(arr, ddof=1))
    if sigma <= 0.0:
        return 0.0
    return float(np.mean(arr)) / sigma


def _align_positions(positions: FloatArray, n_returns: int) -> FloatArray:
    """Align a per-bar position vector (one per close) to the return path length.

    The close series has ``N`` bars; :func:`algosystem.data.compute_returns` drops
    the first (NaN) return, so the per-bar return path has ``N - 1`` entries where
    ``returns[i]`` is the ``close[i] -> close[i+1]`` return. The position decided at
    the close of bar ``i+1`` earns ``returns[i+1]`` under the engine's internal
    ``shift``; dropping the first position keeps the two vectors the same length and
    the next-bar-fill causality intact.
    """
    pos = np.asarray(positions, dtype="float64").ravel()
    aligned = pos[pos.size - n_returns :]
    if aligned.size != n_returns:  # pragma: no cover - defensive: lengths always match
        raise ValidationError(
            f"_align_positions: cannot align {pos.size} positions to {n_returns} returns."
        )
    return aligned


def _selected_spec(signal: str, fast: int, slow: int, lookback: int) -> SignalSpec:
    """Build the selected :class:`SignalSpec` from the CLI parameters."""
    if signal == "ma_crossover":
        return SignalSpec("ma_crossover", {"fast": fast, "slow": slow})
    if signal == "momentum":
        return SignalSpec("momentum", {"lookback": lookback})
    raise ValidationError(
        f"signal must be 'ma_crossover' or 'momentum', got {signal!r}."
    )


def run_pipeline(
    *,
    signal: str = "ma_crossover",
    fast: int = 10,
    slow: int = 50,
    lookback: int = 20,
    cost_bps: float = 5.0,
    slippage_bps: float = 2.0,
    data_source_pref: str = "synthetic",
    seed: int = 7,
    n_obs: int = _DEFAULT_N_OBS,
) -> PipelineResult:
    """Run the full offline pipeline for one selected config; return the metrics + verdict.

    Loads the synthetic default OHLC bars, evaluates the FULL #signals x #param
    grid through the vectorized backtester, runs the backtest<->live parity oracle
    on the selected config, computes the OOS metrics, the Diebold-Mariano test vs.
    buy-and-hold, the Deflated Sharpe (honest grid-wide ``n_trials``), the PBO/CSCV,
    and derives the PURE ``system_has_edge`` verdict. Pure numpy/scipy/statsmodels;
    no network on the synthetic path; never trains.

    Parameters
    ----------
    signal:
        ``"ma_crossover"`` (default) or ``"momentum"`` (the SELECTED config).
    fast, slow:
        The fast / slow MA windows (for ``ma_crossover``; ``fast < slow``).
    lookback:
        The momentum lookback (for ``momentum``).
    cost_bps, slippage_bps:
        Per-side transaction cost / per-trade slippage in basis points.
    data_source_pref:
        ``"synthetic"`` (default) or ``"auto"`` (the CLI synthetic path always
        resolves to deterministic synthetic bars).
    seed:
        Master RNG seed for the synthetic path.
    n_obs:
        Number of synthetic bars to generate.

    Returns
    -------
    PipelineResult
        The selected-config metrics, the parity report, and the PURE verdict.

    Raises
    ------
    ValidationError
        If ``signal`` is unknown, ``fast >= slow``, or a friction is invalid.
    """
    if not math.isfinite(cost_bps) or cost_bps < 0.0:
        raise ValidationError(f"cost_bps must be finite and >= 0, got {cost_bps!r}.")
    if not math.isfinite(slippage_bps) or slippage_bps < 0.0:
        raise ValidationError(f"slippage_bps must be finite and >= 0, got {slippage_bps!r}.")
    selected = _selected_spec(signal, fast, slow, lookback)

    # Deterministic synthetic bars (the deployed-default DGP, the honest null). The
    # ``polygon`` path is reachable from ``load_single_asset_bars`` in the offline
    # CLI; here the cheap default always uses the synthetic GBM-regime bars.
    bars, returns, data_source = synthetic_default_bars(
        n_obs=n_obs, seed=seed, kind="gbm_regime"
    )
    close: pd.Series = bars["close"]
    ret = np.asarray(returns.to_numpy(dtype="float64"), dtype="float64")
    n_ret = ret.size

    # Evaluate the FULL grid (honest multiplicity), collecting per-config net returns
    # and the selected config's position sequence. The selected config is also
    # appended to the grid (de-duplicated) so its DSR n_trials honestly counts it.
    grid: list[SignalSpec] = list(_CONFIG_GRID)
    if selected not in grid:
        grid.append(selected)

    net_columns: list[FloatArray] = []
    trial_sharpes: list[float] = []
    selected_net: FloatArray | None = None
    selected_positions: FloatArray | None = None
    for spec in grid:
        positions = _align_positions(build_signal(spec, close), n_ret)
        result = vectorized_backtest(
            ret, positions, cost_bps=cost_bps, slippage_bps=slippage_bps
        )
        net_columns.append(result.net_returns)
        trial_sharpes.append(_per_obs_sharpe(result.net_returns))
        if spec == selected:
            selected_net = result.net_returns
            selected_positions = result.positions
    assert selected_net is not None  # the selected spec is always in the grid.
    assert selected_positions is not None

    # The backtest<->live PARITY ORACLE on the selected config: the vectorized
    # backtest equity curve must equal the simulated paper-broker replay to 1e-10.
    parity = check_parity(
        ret,
        _align_positions(build_signal(selected, close), n_ret),
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
    )

    # Buy-and-hold baseline: a constant long position over the same path + frictions.
    buyhold = vectorized_backtest(
        ret, np.ones(n_ret, dtype="float64"), cost_bps=cost_bps, slippage_bps=slippage_bps
    )

    # OOS metrics for the selected strategy (net of costs + slippage).
    metrics = strategy_metrics(selected_net, selected_positions)
    buyhold_metrics = strategy_metrics(buyhold.net_returns, buyhold.positions)

    # Diebold-Mariano of the selected strategy vs. buy-and-hold per-bar net return.
    dm_statistic, dm_pvalue = diebold_mariano(selected_net, buyhold.net_returns)

    # Deflated Sharpe with the HONEST grid-wide n_trials and the selected config's
    # per-obs Sharpe + sample moments; PBO/CSCV over the full grid's net-return
    # matrix. The DSR is non-increasing in n_trials (the multiplicity deflation).
    n_trials = len(grid)
    sel_arr = np.asarray(selected_net, dtype="float64").ravel()
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

    # The PURE verdict: system_has_edge True iff DM-significant AND DSR > 1-alpha
    # AND PBO < 0.5, all net of costs. On the synthetic null this is False.
    verdict = system_has_edge(
        dm_statistic, dm_pvalue, dsr, pbo_result.pbo, n_trials
    )

    return PipelineResult(
        signal=signal,
        oos_sharpe=metrics.oos_sharpe,
        buyhold_sharpe=buyhold_metrics.oos_sharpe,
        dm_statistic=dm_statistic,
        dm_pvalue=dm_pvalue,
        deflated_sharpe=dsr,
        pbo=pbo_result.pbo,
        parity_max_diff=parity.max_abs_diff,
        parity_ok=parity.passed,
        turnover=metrics.turnover,
        max_drawdown=metrics.max_drawdown,
        n_trials=n_trials,
        data_source=data_source,
        verdict=verdict,
    )


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


def _format_result_lines(result: PipelineResult, *, header: str) -> list[str]:
    """Format a :class:`PipelineResult` into aligned, human-readable report lines."""
    edge = "YES" if result.verdict.system_has_edge else "NO"
    parity = "PASS" if result.parity_ok else "FAIL"
    return [
        header,
        f"  data source         : {result.data_source}",
        f"  signal              : {result.signal}",
        f"  OOS net Sharpe      : {result.oos_sharpe:+.4f}",
        f"  buy-hold Sharpe     : {result.buyhold_sharpe:+.4f}",
        f"  DM statistic        : {result.dm_statistic:+.4f}",
        f"  DM p-value          : {result.dm_pvalue:.4f}",
        f"  Deflated Sharpe     : {result.deflated_sharpe:.4f}",
        f"  PBO                 : {result.pbo:.4f}",
        f"  turnover            : {result.turnover:.4f}",
        f"  max drawdown        : {result.max_drawdown:+.4f}",
        f"  n effective trials  : {result.n_trials}",
        f"  backtest=live parity: {parity} (max-diff {result.parity_max_diff:.2e})",
        f"  system has edge     : {edge}",
    ]


def _build_app() -> typer.Typer:
    """Construct the Typer application with the backtest / paper / compare commands.

    LAZY IMPORT: ``typer`` (the ``dev`` extra) is imported inside this builder so
    importing :mod:`algosystem.cli` is cheap and side-effect-free.

    Returns
    -------
    typer.Typer
        The configured Typer app.
    """
    import typer

    app = typer.Typer(
        add_completion=False,
        help=(
            "algo-system — a leakage-free signal -> backtest -> simulated paper "
            "execution pipeline, judged by the backtest<->live parity oracle and a "
            "PURE system_has_edge verdict (honest null: no edge after costs)."
        ),
        no_args_is_help=True,
    )

    @app.command()
    def backtest(
        signal: str = typer.Option("ma_crossover", help="'ma_crossover' or 'momentum'."),
        fast: int = typer.Option(10, help="Fast MA window (ma_crossover; fast < slow)."),
        slow: int = typer.Option(50, help="Slow MA window (ma_crossover)."),
        lookback: int = typer.Option(20, help="Momentum lookback."),
        cost_bps: float = typer.Option(5.0, help="Per-side transaction cost (bps)."),
        slippage_bps: float = typer.Option(2.0, help="Per-trade slippage (bps)."),
        seed: int = typer.Option(7, help="Master RNG seed for the synthetic bars."),
    ) -> None:
        """Run the vectorized backtest of a signal and print the OOS metrics + verdict."""
        result = _safe_run(
            signal=signal,
            fast=fast,
            slow=slow,
            lookback=lookback,
            cost_bps=cost_bps,
            slippage_bps=slippage_bps,
            seed=seed,
        )
        for line in _format_result_lines(result, header="Backtest (vectorized OOS)"):
            typer.echo(line)

    @app.command()
    def paper(
        signal: str = typer.Option("ma_crossover", help="'ma_crossover' or 'momentum'."),
        fast: int = typer.Option(10, help="Fast MA window (ma_crossover; fast < slow)."),
        slow: int = typer.Option(50, help="Slow MA window (ma_crossover)."),
        lookback: int = typer.Option(20, help="Momentum lookback."),
        cost_bps: float = typer.Option(5.0, help="Per-side transaction cost (bps)."),
        slippage_bps: float = typer.Option(2.0, help="Per-trade slippage (bps)."),
        seed: int = typer.Option(7, help="Master RNG seed for the synthetic bars."),
    ) -> None:
        """Replay the signal through the simulated paper broker and print the live summary."""
        result = _safe_run(
            signal=signal,
            fast=fast,
            slow=slow,
            lookback=lookback,
            cost_bps=cost_bps,
            slippage_bps=slippage_bps,
            seed=seed,
        )
        # The paper-broker "live" curve coincides with the backtest (parity oracle),
        # so the OOS metrics it earns are identical; the summary reports the same
        # net Sharpe / turnover / drawdown alongside the parity proof.
        for line in _format_result_lines(result, header="Paper broker (simulated live)"):
            typer.echo(line)

    @app.command()
    def compare(
        signal: str = typer.Option("ma_crossover", help="'ma_crossover' or 'momentum'."),
        fast: int = typer.Option(10, help="Fast MA window (ma_crossover; fast < slow)."),
        slow: int = typer.Option(50, help="Slow MA window (ma_crossover)."),
        lookback: int = typer.Option(20, help="Momentum lookback."),
        cost_bps: float = typer.Option(5.0, help="Per-side transaction cost (bps)."),
        slippage_bps: float = typer.Option(2.0, help="Per-trade slippage (bps)."),
        seed: int = typer.Option(7, help="Master RNG seed for the synthetic bars."),
    ) -> None:
        """Run BOTH paths; print the parity max-diff, the metrics, DM/DSR/PBO, and the verdict."""
        result = _safe_run(
            signal=signal,
            fast=fast,
            slow=slow,
            lookback=lookback,
            cost_bps=cost_bps,
            slippage_bps=slippage_bps,
            seed=seed,
        )
        for line in _format_result_lines(
            result, header="Compare (backtest vs. simulated live)"
        ):
            typer.echo(line)

    return app


def _safe_run(
    *,
    signal: str,
    fast: int,
    slow: int,
    lookback: int,
    cost_bps: float,
    slippage_bps: float,
    seed: int,
) -> PipelineResult:
    """Run the pipeline, mapping a :class:`ValidationError` to a Typer ``BadParameter``.

    LAZY IMPORT of ``typer`` so the module stays side-effect-free; the validation
    failure is surfaced as a clean CLI error (non-zero exit) instead of a traceback.
    """
    try:
        return run_pipeline(
            signal=signal,
            fast=fast,
            slow=slow,
            lookback=lookback,
            cost_bps=cost_bps,
            slippage_bps=slippage_bps,
            seed=seed,
        )
    except ValidationError as exc:
        import typer

        raise typer.BadParameter(str(exc)) from exc


def main() -> None:
    """Console-script entry point (``algo-system`` -> ``algosystem.cli:main``).

    Builds the Typer app (lazy ``typer`` import) and dispatches to the requested
    sub-command. Wired to the ``[project.scripts]`` entry in ``pyproject.toml``.
    """
    app = _build_app()
    app()


if __name__ == "__main__":  # pragma: no cover - manual entry point
    main()
