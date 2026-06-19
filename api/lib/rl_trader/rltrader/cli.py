"""Typer CLI: ``train`` / ``backtest`` / ``compare`` (typer imported lazily).

The console entrypoint (``rl-trader``) exposes three commands:

- ``train``    — the OFFLINE pipeline (synthetic -> walk-forward -> PPO across N
  seeds -> ONNX policy + metrics.json); the only command that pulls in the
  ``[train]`` extra (torch + sb3 + gymnasium).
- ``backtest`` — evaluate the committed ONNX policy + the live baselines on a
  synthetic path and print the per-strategy OOS Sharpe / drawdown / turnover table
  (onnxruntime, NO torch).
- ``compare``  — the full RL-vs-buy-hold comparison with the PURE
  ``rl_beats_baseline`` verdict, derived honestly from the inference outputs.

``typer`` is imported LAZILY inside :func:`build_app` (it lives in the ``[dev]``
extra), so importing this module pulls in NO typer and has no side effects. The PPO
agent stays lazy too: ``backtest`` and ``compare`` compute the baselines with pure
numpy and only attempt the committed ONNX policy when its artifact exists
(onnxruntime, NEVER torch). The ``main`` entrypoint builds and runs the app.

This module NEVER calls :func:`rltrader.serve.run_backtest`; ``backtest`` and
``compare`` assemble the same torch-free evaluation here directly so the CLI stays
independent of the serve entrypoint.

Importing this module has no side effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from rltrader._typing import FloatArray

if TYPE_CHECKING:
    import typer

#: The baselines compared against (and reported alongside) the RL agent.
_BASELINE_NAMES: tuple[str, ...] = ("buy_hold", "flat_cash", "random_action")

#: Default per-trade slippage (bps) — kept in lock-step with the training cost model.
_SLIPPAGE_BPS: float = 1.0

#: Hyper-parameter configurations swept per seed. The honest DSR multiplicity is
#: ``#seeds x #HP configs`` — the seed lottery AND the HP grid both count as
#: selection trials (never undercount). One HP config in the shipped grid.
_N_HP_CONFIGS: int = 1


def _cli_effective_trials(n_seeds: int) -> int:
    """Return the honest DSR multiplicity ``n_seeds x #HP configs`` (computed locally).

    Counts the FULL explored configuration grid so the selection bias of trying many
    seeds x hyper-parameters is paid for in the Deflated-Sharpe benchmark. Computed
    inline (not via the still-offline ``train`` pipeline) so the CLI's verdict path
    stays torch-free and self-contained.

    Parameters
    ----------
    n_seeds:
        The number of independent training seeds (``>= 1``).

    Returns
    -------
    int
        ``max(1, n_seeds) * _N_HP_CONFIGS``.
    """
    return max(1, int(n_seeds)) * _N_HP_CONFIGS


@dataclass(frozen=True, slots=True)
class _StrategyRow:
    """One strategy's OOS line in the CLI table (net of costs + slippage)."""

    name: str
    oos_sharpe: float
    max_drawdown: float
    turnover: float
    net_pnl: float
    n_bars: int


def _build_returns(*, n_obs: int, kind: str, seed: int) -> FloatArray:
    """Build the seeded synthetic per-bar return path (torch-free, no network).

    Routes through :func:`rltrader.data.loaders.synthetic_default_path` (which
    validates ``kind`` and ``n_obs``) and materializes the returns as a contiguous
    float64 array for the env + backtester. NO torch / sb3 / gymnasium is imported.

    Parameters
    ----------
    n_obs:
        Number of synthetic bars to generate.
    kind:
        Synthetic DGP (``gbm_regime`` | ``pure_trend`` | ``pure_noise``).
    seed:
        Master RNG seed for the synthetic path.

    Returns
    -------
    FloatArray
        The per-bar return path as a 1-D float64 array.
    """
    from rltrader.data.loaders import synthetic_default_path

    _, returns, _ = synthetic_default_path(n_obs=n_obs, seed=seed, kind=kind)
    return np.asarray(returns.to_numpy(), dtype="float64")


def _baseline_rows(
    returns: FloatArray,
    *,
    cost_bps: float,
    seed: int,
) -> tuple[list[_StrategyRow], dict[str, FloatArray]]:
    """Run the live (numpy) baselines and return their table rows + per-bar net returns.

    Each baseline position sequence is scored through the SHARED vectorized
    backtester (strictly causal, cost-aware) so its accounting is identical to the
    RL agent's. Pure numpy — no torch / onnxruntime. Returns the table rows and a
    ``{name: net_returns}`` map (the buy-hold series feeds the Diebold-Mariano test).

    Parameters
    ----------
    returns:
        The per-bar return path.
    cost_bps:
        Per-side transaction cost in basis points.
    seed:
        Master RNG seed for the ``random_action`` baseline.

    Returns
    -------
    tuple[list[_StrategyRow], dict[str, FloatArray]]
        The per-baseline rows and a map of baseline name -> per-bar net returns.
    """
    from rltrader.agents.baselines import run_baseline
    from rltrader.evaluation.metrics import max_drawdown, oos_sharpe

    rows: list[_StrategyRow] = []
    nets: dict[str, FloatArray] = {}
    for name in _BASELINE_NAMES:
        result = run_baseline(
            name,
            returns,
            cost_bps=cost_bps,
            slippage_bps=_SLIPPAGE_BPS,
            seed=seed,
        )
        net = np.asarray(result.net_returns, dtype="float64")
        nets[name] = net
        rows.append(
            _StrategyRow(
                name=name,
                oos_sharpe=oos_sharpe(net),
                max_drawdown=max_drawdown(net),
                turnover=float(result.turnover),
                net_pnl=float(result.net_pnl),
                n_bars=int(result.n_bars),
            )
        )
    return rows, nets


def _observation_matrix(returns: FloatArray, *, lookback: int) -> FloatArray:
    """Build the per-bar observation matrix the env would emit (data <= t only).

    Mirrors :meth:`rltrader.env.trading_env.TradingEnv._observe`: each scored bar
    ``t in [lookback-1, N-2]`` gets the trailing return window
    ``[r_{t-lookback+1}, ..., r_t]`` concatenated with the current position. The
    position column is set to ``0.0`` (the book opens flat); this is the
    look-ahead-safe observation construction the committed ONNX policy was exported
    against. NEVER reads ``r_{t+1}`` or beyond.

    Parameters
    ----------
    returns:
        The per-bar return path.
    lookback:
        Observation look-back window length (``>= 1``).

    Returns
    -------
    FloatArray
        A ``(n_decision_bars, lookback + 1)`` observation matrix.
    """
    r = np.asarray(returns, dtype="float64").ravel()
    n = r.size
    # Scored bars are t in [0, N-2]; the first WELL-FORMED observation needs a full
    # look-back window, so decisions start at t = lookback - 1.
    start = lookback - 1
    rows: list[FloatArray] = []
    for t in range(start, n - 1):
        window = r[t - lookback + 1 : t + 1]
        rows.append(np.concatenate((window, np.array([0.0], dtype="float64"))))
    if not rows:
        return np.empty((0, lookback + 1), dtype="float64")
    return np.asarray(rows, dtype="float64")


def _rl_row(
    returns: FloatArray,
    *,
    lookback: int,
    cost_bps: float,
) -> tuple[_StrategyRow | None, FloatArray | None]:
    """Score the committed ONNX policy on the OOS path (or ``None`` if no artifact).

    Serves the committed ``artifacts/policy.onnx`` via onnxruntime (NEVER torch):
    builds the look-ahead-safe observation matrix, maps it to per-bar positions, and
    scores them through the vectorized backtester ALIGNED to the same scored window
    as the baselines (decisions begin at ``lookback - 1``; earlier bars are flat).
    When the artifact is absent or unloadable the RL row is ``None`` so the CLI
    degrades to a baselines-only table (the policy is a separate offline deliverable).

    Parameters
    ----------
    returns:
        The per-bar return path.
    lookback:
        Observation look-back window length.
    cost_bps:
        Per-side transaction cost in basis points.

    Returns
    -------
    tuple[_StrategyRow | None, FloatArray | None]
        The RL table row and its per-bar net returns, or ``(None, None)`` when the
        committed ONNX policy artifact is unavailable.
    """
    from rltrader._exceptions import ArtifactError
    from rltrader.agents.onnx_policy import OnnxPolicy, default_artifact_path
    from rltrader.env.backtester import vectorized_backtest
    from rltrader.evaluation.metrics import max_drawdown, oos_sharpe

    if not default_artifact_path().is_file():
        return None, None

    obs = _observation_matrix(returns, lookback=lookback)
    if obs.shape[0] == 0:
        return None, None

    try:
        decision_positions = OnnxPolicy().predict_positions(obs)
    except ArtifactError:
        # A missing/corrupt/mismatched artifact degrades to baselines-only honestly.
        return None, None

    r = np.asarray(returns, dtype="float64").ravel()
    # The full position sequence over [0, N-1]: flat until the first well-formed
    # decision bar (t = lookback - 1), then the policy's positions. The backtester
    # scores t in [0, N-2] (each earns r_{t+1}); the trailing element is unused.
    positions = np.zeros(r.size, dtype="float64")
    start = lookback - 1
    positions[start : start + decision_positions.size] = decision_positions

    result = vectorized_backtest(
        r,
        positions,
        cost_bps=cost_bps,
        slippage_bps=_SLIPPAGE_BPS,
        initial_position=0.0,
    )
    net = np.asarray(result.net_returns, dtype="float64")
    equity = np.asarray(result.equity_curve, dtype="float64")
    row = _StrategyRow(
        name="rl_onnx",
        oos_sharpe=oos_sharpe(net),
        max_drawdown=max_drawdown(net),
        turnover=float(result.turnover),
        net_pnl=float(equity[-1] - 1.0) if equity.size else 0.0,
        n_bars=int(result.n_bars),
    )
    return row, net


def _fmt(value: float) -> str:
    """Format a float for the CLI table (``n/a`` for NaN, fixed precision otherwise)."""
    return "n/a" if not math.isfinite(value) else f"{value:+.4f}"


def _print_table(rows: list[_StrategyRow], *, echo: object) -> None:
    """Print the per-strategy OOS table (Sharpe / drawdown / turnover / net PnL).

    Parameters
    ----------
    rows:
        The strategy rows to render.
    echo:
        A ``callable(str) -> None`` printer (``typer.echo`` at runtime).
    """
    write = echo  # typer.echo
    header = f"{'strategy':<14}{'sharpe':>11}{'max_dd':>11}{'turnover':>11}{'net_pnl':>11}"
    write(header)  # type: ignore[operator]
    write("-" * len(header))  # type: ignore[operator]
    for row in rows:
        write(  # type: ignore[operator]
            f"{row.name:<14}"
            f"{_fmt(row.oos_sharpe):>11}"
            f"{_fmt(row.max_drawdown):>11}"
            f"{row.turnover:>11.2f}"
            f"{_fmt(row.net_pnl):>11}"
        )


def build_app() -> typer.Typer:
    """Build and return the Typer application (``typer`` imported lazily here).

    Registers the ``train``, ``backtest``, and ``compare`` commands. Typer is
    imported inside this function so importing :mod:`rltrader.cli` (and the package)
    never imports typer. A fresh instance is returned on every call (no shared
    mutable state).

    Returns
    -------
    typer.Typer
        A configured ``typer.Typer`` instance.

    Raises
    ------
    ImportError
        If ``typer`` (the ``[dev]`` extra) is not installed.
    """
    import typer

    app = typer.Typer(
        add_completion=False,
        no_args_is_help=True,
        help="rl-trader — leakage-free, overfit-aware RL trading benchmark (honest NULL).",
    )

    @app.command("train")
    def _train_cmd(
        n_seeds: int = typer.Option(5, help="Number of independent training seeds."),
        lookback: int = typer.Option(32, help="Observation look-back window length."),
        cost_bps: float = typer.Option(5.0, help="Per-side transaction cost (bps)."),
        slippage_bps: float = typer.Option(1.0, help="Per-trade slippage (bps)."),
        episode_len: int = typer.Option(252, help="Max bars per training episode."),
        n_obs: int = typer.Option(2000, help="Number of synthetic bars to generate."),
        kind: str = typer.Option("gbm_regime", help="Synthetic DGP."),
        seed: int = typer.Option(7, help="Master RNG / torch seed."),
    ) -> None:
        """Run the OFFLINE training pipeline (the [train] extra)."""
        raise typer.Exit(
            code=train(
                n_seeds=n_seeds,
                lookback=lookback,
                cost_bps=cost_bps,
                slippage_bps=slippage_bps,
                episode_len=episode_len,
                n_obs=n_obs,
                kind=kind,
                seed=seed,
            )
        )

    @app.command("backtest")
    def _backtest_cmd(
        n_seeds: int = typer.Option(5, help="Seeds reflected in the seed lottery."),
        cost_bps: float = typer.Option(5.0, help="Per-side transaction cost (bps)."),
        lookback: int = typer.Option(32, help="Observation look-back window length."),
        episode_len: int = typer.Option(252, help="OOS episode length."),
        kind: str = typer.Option("gbm_regime", help="Synthetic DGP."),
        seed: int = typer.Option(7, help="Master RNG seed for the synthetic path."),
    ) -> None:
        """Evaluate the committed ONNX policy + live baselines (NO torch)."""
        raise typer.Exit(
            code=backtest(
                n_seeds=n_seeds,
                cost_bps=cost_bps,
                lookback=lookback,
                episode_len=episode_len,
                kind=kind,
                seed=seed,
            )
        )

    @app.command("compare")
    def _compare_cmd(
        n_seeds: int = typer.Option(5, help="Seeds in the seed lottery."),
        cost_bps: float = typer.Option(5.0, help="Per-side transaction cost (bps)."),
        lookback: int = typer.Option(32, help="Observation look-back window length."),
        episode_len: int = typer.Option(252, help="OOS episode length."),
        kind: str = typer.Option("gbm_regime", help="Synthetic DGP."),
        seed: int = typer.Option(7, help="Master RNG seed for the synthetic path."),
    ) -> None:
        """Run the RL-vs-buy-hold comparison + the PURE rl_beats_baseline verdict."""
        raise typer.Exit(
            code=compare(
                n_seeds=n_seeds,
                cost_bps=cost_bps,
                lookback=lookback,
                episode_len=episode_len,
                kind=kind,
                seed=seed,
            )
        )

    return app


def train(
    *,
    n_seeds: int = 5,
    lookback: int = 32,
    cost_bps: float = 5.0,
    slippage_bps: float = 1.0,
    episode_len: int = 252,
    n_obs: int = 2000,
    kind: str = "gbm_regime",
    seed: int = 7,
) -> int:
    """Run the OFFLINE training pipeline from the command line.

    Delegates to :func:`rltrader.train.train_pipeline` (which lazily imports torch /
    sb3 / gymnasium / the ONNX exporter — the ``[train]`` extra). Prints the honest
    summary: the exported policy + metrics paths, the FULL-grid effective-trial
    count, and the (expected ``False``) ``rl_beats_baseline`` verdict.

    Parameters
    ----------
    n_seeds:
        Number of independent training seeds (the seed lottery).
    lookback:
        Observation look-back window length.
    cost_bps:
        Per-side transaction cost in basis points.
    slippage_bps:
        Per-trade slippage in basis points.
    episode_len:
        Max bars per training episode.
    n_obs:
        Number of synthetic bars to generate.
    kind:
        Synthetic DGP (``gbm_regime`` | ``pure_trend`` | ``pure_noise``).
    seed:
        Master RNG / torch seed.

    Returns
    -------
    int
        Process exit code (``0`` on success, ``1`` on a library error).
    """
    import typer

    from rltrader._exceptions import RlTraderError
    from rltrader.train import train_pipeline

    try:
        result = train_pipeline(
            n_seeds=n_seeds,
            lookback=lookback,
            cost_bps=cost_bps,
            slippage_bps=slippage_bps,
            episode_len=episode_len,
            n_obs=n_obs,
            kind=kind,
            seed=seed,
        )
    except (RlTraderError, ImportError, NotImplementedError) as exc:
        typer.echo(f"train failed: {exc}")
        return 1

    typer.echo("offline training complete")
    typer.echo(f"  policy:  {result.policy_path}")
    typer.echo(f"  metrics: {result.metrics_path}")
    typer.echo(f"  n_effective_trials (seeds x HP): {result.n_effective_trials}")
    typer.echo(f"  rl_beats_baseline: {'YES' if result.rl_beats_baseline else 'NO'}")
    return 0


def backtest(
    *,
    n_seeds: int = 5,
    cost_bps: float = 5.0,
    lookback: int = 32,
    episode_len: int = 252,
    kind: str = "gbm_regime",
    seed: int = 7,
) -> int:
    """Evaluate the committed ONNX policy + the live baselines and print the OOS table.

    Builds the seeded synthetic price path, computes the buy-hold / flat / random
    baselines LIVE (numpy) through the vectorized backtester, serves the committed
    PPO policy from its ONNX artifact when present (onnxruntime, NEVER torch), and
    prints the per-strategy OOS Sharpe / drawdown / turnover table. NO torch.

    Parameters
    ----------
    n_seeds:
        Number of seeds reflected in the seed lottery.
    cost_bps:
        Per-side transaction cost in basis points.
    lookback:
        Observation look-back window length.
    episode_len:
        OOS episode length.
    kind:
        Synthetic DGP (``gbm_regime`` | ``pure_trend`` | ``pure_noise``).
    seed:
        Master RNG seed for the synthetic path.

    Returns
    -------
    int
        Process exit code (``0`` on success, ``1`` on a library error).
    """
    import typer

    from rltrader._exceptions import RlTraderError

    n_obs = _resolve_n_obs(episode_len)
    try:
        returns = _build_returns(n_obs=n_obs, kind=kind, seed=seed)
        rows, _ = _baseline_rows(returns, cost_bps=cost_bps, seed=seed)
        rl_row, _ = _rl_row(returns, lookback=lookback, cost_bps=cost_bps)
    except (RlTraderError, ValueError) as exc:
        typer.echo(f"backtest failed: {exc}")
        return 1

    table = ([rl_row] if rl_row is not None else []) + rows
    typer.echo(
        f"OOS backtest — kind={kind} seeds={n_seeds} cost_bps={cost_bps} lookback={lookback}"
    )
    _print_table(table, echo=typer.echo)
    if rl_row is None:
        typer.echo(
            "note: no committed ONNX policy artifact — baselines only (train offline first)."
        )
    return 0


def compare(
    *,
    n_seeds: int = 5,
    cost_bps: float = 5.0,
    lookback: int = 32,
    episode_len: int = 252,
    kind: str = "gbm_regime",
    seed: int = 7,
) -> int:
    """Run the RL-vs-buy-hold comparison and print the PURE ``rl_beats_baseline`` verdict.

    Computes the baselines LIVE (buy-hold / flat / random, torch-free) and serves
    the committed PPO policy from its ONNX artifact when present (onnxruntime, NEVER
    torch). Summarizes the seed lottery, runs the Diebold-Mariano test of the
    median-seed RL net return vs. buy-hold, and derives the honest verdict via
    :func:`rltrader.evaluation.verdict.derive_verdict` — ``rl_beats_baseline`` is
    ``False`` unless the median-seed beats buy-hold DM-significant AND the DSR>0 AND
    the across-seed Sharpe lower bound > 0. On the GBM-regime null it is ``False``
    by construction.

    Parameters
    ----------
    n_seeds:
        Number of seeds in the seed lottery.
    cost_bps:
        Per-side transaction cost in basis points.
    lookback:
        Observation look-back window length.
    episode_len:
        OOS episode length.
    kind:
        Synthetic DGP (``gbm_regime`` | ``pure_trend`` | ``pure_noise``).
    seed:
        Master RNG seed for the synthetic path.

    Returns
    -------
    int
        Process exit code (``0`` on success, ``1`` on a library error).
    """
    import typer

    from rltrader._exceptions import RlTraderError
    from rltrader.evaluation.diebold_mariano import diebold_mariano
    from rltrader.evaluation.seed_lottery import seed_lottery, variance_of_seed_sharpes
    from rltrader.evaluation.verdict import derive_verdict

    n_obs = _resolve_n_obs(episode_len)
    trials = _cli_effective_trials(n_seeds)
    try:
        returns = _build_returns(n_obs=n_obs, kind=kind, seed=seed)
        rows, baseline_nets = _baseline_rows(returns, cost_bps=cost_bps, seed=seed)
        rl_row, rl_net = _rl_row(returns, lookback=lookback, cost_bps=cost_bps)
    except (RlTraderError, ValueError) as exc:
        typer.echo(f"compare failed: {exc}")
        return 1

    table = ([rl_row] if rl_row is not None else []) + rows
    typer.echo(
        f"RL-vs-buy-hold — kind={kind} seeds={n_seeds} cost_bps={cost_bps} lookback={lookback}"
    )
    _print_table(table, echo=typer.echo)

    bh_sharpe = next((r.oos_sharpe for r in rows if r.name == "buy_hold"), math.nan)

    # Without a committed ONNX policy the RL comparison cannot be formed; the honest
    # verdict is then UNAVAILABLE (no narrated edge), distinct from a tested FALSE.
    if rl_row is None or rl_net is None:
        typer.echo(f"buy-hold OOS Sharpe: {_fmt(bh_sharpe)}")
        typer.echo("rl_beats_baseline: NO (no committed ONNX policy — train offline first)")
        return 0

    # The seed lottery: with one served policy we have one OOS Sharpe; the honest
    # dispersion comes from the committed metrics at serve time. Here the CLI forms a
    # single-seed lottery (the band collapses onto the served Sharpe), which CANNOT
    # clear the strictly-positive-lower-bound gate unless that Sharpe is itself > 0.
    seed_sharpes = np.array([rl_row.oos_sharpe], dtype="float64")
    if not np.isfinite(seed_sharpes).all():
        seed_sharpes = np.array([0.0], dtype="float64")
    lottery = seed_lottery(seed_sharpes, seed=seed)
    variance = variance_of_seed_sharpes(seed_sharpes) if seed_sharpes.size >= 2 else 0.0

    dm_stat, dm_pvalue = _dm_or_null(rl_net, baseline_nets["buy_hold"], diebold_mariano)
    deflated = _deflated_or_zero(
        rl_row.oos_sharpe,
        rl_net.size,
        trials,
        variance,
    )
    verdict = derive_verdict(
        dm_statistic=dm_stat,
        dm_pvalue=dm_pvalue,
        deflated_sharpe=deflated,
        seed_sharpe_lo=lottery.sharpe_lo,
        n_effective_trials=trials,
    )

    typer.echo("")
    typer.echo(f"  median-seed RL OOS Sharpe: {_fmt(lottery.median_sharpe)}")
    typer.echo(f"  buy-hold OOS Sharpe:       {_fmt(bh_sharpe)}")
    typer.echo(
        f"  seed Sharpe band:          [{_fmt(lottery.sharpe_lo)}, {_fmt(lottery.sharpe_hi)}]"
    )
    typer.echo(f"  DM p-value vs buy-hold:    {_fmt(dm_pvalue)}")
    typer.echo(f"  deflated Sharpe:           {_fmt(deflated)}")
    typer.echo(f"  n_effective_trials:        {trials}")
    typer.echo(f"rl_beats_baseline: {'YES' if verdict.rl_beats_baseline else 'NO'}")
    return 0


def _resolve_n_obs(episode_len: int) -> int:
    """Return a synthetic path length comfortably longer than one OOS episode.

    The synthetic path must hold at least one full OOS episode plus the look-back
    warm-up; a 4x multiple (floored at the synthetic minimum) gives a path with room
    for the walk-forward without being needlessly large.
    """
    return max(64, int(episode_len) * 4)


def _dm_or_null(
    rl_net: FloatArray,
    buyhold_net: FloatArray,
    dm_fn: object,
) -> tuple[float, float]:
    """Run Diebold-Mariano, returning a null ``(0.0, 1.0)`` on a degenerate differential.

    A constant (zero-dispersion) RL-vs-buy-hold differential makes the DM statistic
    undefined; rather than crash the CLI we report the honest null
    ``(statistic=0, pvalue=1)`` — no significant difference — which the verdict reads
    as "does not beat buy-hold".
    """
    from rltrader._exceptions import ValidationError

    try:
        stat, pvalue = dm_fn(rl_net, buyhold_net)  # type: ignore[operator]
    except ValidationError:
        return 0.0, 1.0
    return float(stat), float(pvalue)


def _deflated_or_zero(
    observed_sharpe: float,
    n_obs: int,
    n_trials: int,
    variance: float,
) -> float:
    """Compute the Deflated Sharpe, returning ``0.0`` when the observed Sharpe is NaN.

    A flat (zero-dispersion) RL net-return series has an undefined OOS Sharpe (NaN);
    a NaN cannot clear the ``deflated_sharpe > 0`` gate, so the honest substitute is
    ``0.0`` (the DSR floor), keeping the verdict FALSE rather than raising.
    """
    from rltrader.evaluation.dsr import deflated_sharpe_ratio

    if not math.isfinite(observed_sharpe):
        return 0.0
    # The DSR consumes a PER-OBSERVATION Sharpe; de-annualize the reported figure.
    from rltrader._constants import PERIODS_PER_YEAR

    per_obs = observed_sharpe / math.sqrt(PERIODS_PER_YEAR)
    return deflated_sharpe_ratio(
        per_obs,
        n_obs=n_obs,
        n_trials=n_trials,
        variance_of_trial_sharpes=variance,
    )


def main() -> None:
    """Console-script entrypoint: build the Typer app and run it.

    Wired to the ``rl-trader`` console script in ``pyproject.toml``. Builds the app
    via :func:`build_app` and invokes it.

    Raises
    ------
    ImportError
        If ``typer`` (the ``[dev]`` extra) is not installed.
    """
    build_app()()


if __name__ == "__main__":  # pragma: no cover - module-as-script entrypoint
    main()
