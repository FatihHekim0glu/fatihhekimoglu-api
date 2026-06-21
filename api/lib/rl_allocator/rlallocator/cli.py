"""Typer CLI: ``train`` / ``backtest`` / ``compare`` (typer imported lazily).

The console entrypoint (``rl-allocator``) exposes three commands:

- ``train``    — the OFFLINE pipeline (synthetic -> purged walk-forward -> PPO across
  N seeds -> ONNX policy + metrics.json); the only command that pulls in the
  ``[train]`` extra (torch + sb3 + gymnasium). A scaffold stub until the offline path
  is wired.
- ``backtest`` — evaluate the live baselines (and the committed ONNX policy when
  present) on a synthetic multi-asset panel inside the purged walk-forward and print
  the per-strategy OOS Sharpe / drawdown / turnover table (onnxruntime, NO torch).
- ``compare``  — the full RL-vs-baselines comparison with the PURE
  ``rl_beats_baselines`` verdict, derived honestly from the inference outputs.

``typer`` is imported LAZILY inside :func:`build_app` (it lives in the ``[dev]``
extra), so importing this module pulls in NO typer and has no side effects. The
``backtest`` / ``compare`` commands compute the baselines with pure numpy (torch-free)
through :func:`rlallocator.serve.run_allocation`. The ``main`` entrypoint builds and
runs the app.

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import typer


def build_app() -> typer.Typer:
    """Build and return the Typer application (``typer`` imported lazily here).

    Registers the ``train``, ``backtest``, and ``compare`` commands. Typer is imported
    inside this function so importing :mod:`rlallocator.cli` (and the package) never
    imports typer. A fresh instance is returned on every call (no shared mutable
    state).

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
        help="rl-allocator — leakage-free, overfit-aware multi-asset RL allocator (honest NULL).",
    )

    @app.command("train")
    def _train_cmd(
        n_assets: int = typer.Option(6, help="Number of assets in the basket."),
        n_seeds: int = typer.Option(5, help="Number of independent training seeds."),
        lookback: int = typer.Option(64, help="Observation look-back window length."),
        cost_bps: float = typer.Option(10.0, help="Per-side turnover cost (bps)."),
        kind: str = typer.Option("factor_regime", help="Synthetic DGP."),
        seed: int = typer.Option(7, help="Master RNG / torch seed."),
    ) -> None:
        """Run the OFFLINE training pipeline (the [train] extra)."""
        raise typer.Exit(
            code=train(
                n_assets=n_assets,
                n_seeds=n_seeds,
                lookback=lookback,
                cost_bps=cost_bps,
                kind=kind,
                seed=seed,
            )
        )

    @app.command("backtest")
    def _backtest_cmd(
        n_assets: int = typer.Option(6, help="Number of assets in the basket."),
        n_seeds: int = typer.Option(5, help="Seeds reflected in the seed lottery."),
        cost_bps: float = typer.Option(10.0, help="Per-side turnover cost (bps)."),
        lookback: int = typer.Option(64, help="Observation look-back window length."),
        seed: int = typer.Option(7, help="Master RNG seed for the synthetic panel."),
    ) -> None:
        """Evaluate the live baselines (+ committed ONNX policy) — NO torch."""
        raise typer.Exit(
            code=backtest(
                n_assets=n_assets,
                n_seeds=n_seeds,
                cost_bps=cost_bps,
                lookback=lookback,
                seed=seed,
            )
        )

    @app.command("compare")
    def _compare_cmd(
        n_assets: int = typer.Option(6, help="Number of assets in the basket."),
        n_seeds: int = typer.Option(5, help="Seeds in the seed lottery."),
        cost_bps: float = typer.Option(10.0, help="Per-side turnover cost (bps)."),
        lookback: int = typer.Option(64, help="Observation look-back window length."),
        seed: int = typer.Option(7, help="Master RNG seed for the synthetic panel."),
    ) -> None:
        """Run the RL-vs-baselines comparison + the PURE rl_beats_baselines verdict."""
        raise typer.Exit(
            code=compare(
                n_assets=n_assets,
                n_seeds=n_seeds,
                cost_bps=cost_bps,
                lookback=lookback,
                seed=seed,
            )
        )

    return app


def train(
    *,
    n_assets: int = 6,
    n_seeds: int = 5,
    lookback: int = 64,
    cost_bps: float = 10.0,
    kind: str = "factor_regime",
    seed: int = 7,
) -> int:
    """Run the OFFLINE training pipeline from the command line (scaffold stub).

    Delegates to :func:`rlallocator.train.train_pipeline` (which lazily imports torch /
    sb3 / gymnasium / the ONNX exporter — the ``[train]`` extra). Until that path is
    wired it surfaces the ``NotImplementedError`` as a clean, non-crashing message and
    a non-zero exit code.

    Parameters
    ----------
    n_assets, n_seeds, lookback, cost_bps, kind, seed:
        Forwarded to :func:`rlallocator.train.train_pipeline`.

    Returns
    -------
    int
        Process exit code (``0`` on success, ``1`` on a library / not-yet-implemented
        error).
    """
    import typer

    from rlallocator._exceptions import RlAllocatorError
    from rlallocator.train import train_pipeline

    try:
        result = train_pipeline(
            n_assets=n_assets,
            n_seeds=n_seeds,
            lookback=lookback,
            cost_bps=cost_bps,
            kind=kind,
            seed=seed,
        )
    except (RlAllocatorError, ImportError, NotImplementedError) as exc:
        typer.echo(f"train unavailable: {exc}")
        return 1

    typer.echo("offline training complete")
    typer.echo(f"  policy:  {result.policy_path}")
    typer.echo(f"  metrics: {result.metrics_path}")
    typer.echo(f"  n_effective_trials (seeds x HP): {result.n_effective_trials}")
    typer.echo(f"  rl_beats_baselines: {'YES' if result.rl_beats_baselines else 'NO'}")
    return 0


def backtest(
    *,
    n_assets: int = 6,
    n_seeds: int = 5,
    cost_bps: float = 10.0,
    lookback: int = 64,
    seed: int = 7,
) -> int:
    """Evaluate the live baselines (+ committed ONNX policy) and print the OOS table.

    Runs :func:`rlallocator.serve.run_allocation` (baselines LIVE through the purged
    walk-forward, RL from the committed ONNX policy when present — NO torch) and prints
    the per-strategy OOS Sharpe table.

    Parameters
    ----------
    n_assets, n_seeds, cost_bps, lookback, seed:
        Forwarded to :func:`rlallocator.serve.run_allocation`.

    Returns
    -------
    int
        Process exit code (``0`` on success, ``1`` on a library error).
    """
    import typer

    from rlallocator._exceptions import RlAllocatorError
    from rlallocator.serve import run_allocation

    try:
        run = run_allocation(
            n_assets=n_assets,
            n_seeds=n_seeds,
            cost_bps=cost_bps,
            lookback=lookback,
            seed=seed,
        )
    except RlAllocatorError as exc:
        typer.echo(f"backtest failed: {exc}")
        return 1

    s = run.summary
    typer.echo(
        f"OOS backtest — assets={n_assets} seeds={n_seeds} cost_bps={cost_bps} lookback={lookback}"
    )
    typer.echo(f"  1/N          OOS Sharpe: {_fmt(s.oos_sharpe_1n)}")
    typer.echo(f"  Markowitz    OOS Sharpe: {_fmt(s.oos_sharpe_markowitz)}")
    typer.echo(f"  risk-parity  OOS Sharpe: {_fmt(s.oos_sharpe_riskparity)}")
    typer.echo(f"  RL (median)  OOS Sharpe: {_fmt(s.oos_sharpe_rl_median)}")
    typer.echo(f"  best baseline:           {s.best_baseline}")
    typer.echo(f"  data source:             {s.data_source}")
    return 0


def compare(
    *,
    n_assets: int = 6,
    n_seeds: int = 5,
    cost_bps: float = 10.0,
    lookback: int = 64,
    seed: int = 7,
) -> int:
    """Run the RL-vs-baselines comparison and print the PURE ``rl_beats_baselines`` verdict.

    Runs :func:`rlallocator.serve.run_allocation` and prints the headline OOS Sharpes,
    the DM p-value vs. the best baseline, the Deflated Sharpe, the PBO, the seed band,
    and the honest verdict — ``rl_beats_baselines`` is ``False`` unless the median-seed
    beats the best baseline DM-significant AND the DSR > 1-alpha AND the across-seed
    Sharpe lower bound > 0 AND the PBO < 0.5. On the factor-regime null it is ``False``.

    Parameters
    ----------
    n_assets, n_seeds, cost_bps, lookback, seed:
        Forwarded to :func:`rlallocator.serve.run_allocation`.

    Returns
    -------
    int
        Process exit code (``0`` on success, ``1`` on a library error).
    """
    import typer

    from rlallocator._exceptions import RlAllocatorError
    from rlallocator.serve import run_allocation

    try:
        run = run_allocation(
            n_assets=n_assets,
            n_seeds=n_seeds,
            cost_bps=cost_bps,
            lookback=lookback,
            seed=seed,
        )
    except RlAllocatorError as exc:
        typer.echo(f"compare failed: {exc}")
        return 1

    s = run.summary
    typer.echo(f"RL-vs-baselines — assets={n_assets} seeds={n_seeds} cost_bps={cost_bps}")
    typer.echo(f"  median-seed RL OOS Sharpe: {_fmt(s.oos_sharpe_rl_median)}")
    typer.echo(f"  best baseline ({s.best_baseline}) OOS Sharpe: {_fmt(_best_sharpe(s))}")
    typer.echo(f"  seed Sharpe band:          [{_fmt(s.seed_sharpe_lo)}, {_fmt(s.seed_sharpe_hi)}]")
    typer.echo(f"  DM p-value vs best:        {_fmt(s.dm_pvalue_vs_best)}")
    typer.echo(f"  deflated Sharpe:           {_fmt(s.deflated_sharpe)}")
    typer.echo(f"  PBO:                       {_fmt(s.pbo)}")
    typer.echo(f"  n_effective_trials:        {s.n_effective_trials}")
    typer.echo(f"rl_beats_baselines: {'YES' if s.rl_beats_baselines else 'NO'}")
    return 0


def _best_sharpe(summary: object) -> float:
    """Return the best baseline's OOS Sharpe from a summary (for the CLI print)."""
    s = summary
    mapping = {
        "equal_weight": getattr(s, "oos_sharpe_1n", 0.0),
        "markowitz": getattr(s, "oos_sharpe_markowitz", 0.0),
        "risk_parity": getattr(s, "oos_sharpe_riskparity", 0.0),
    }
    return float(mapping.get(getattr(s, "best_baseline", ""), 0.0))


def _fmt(value: float) -> str:
    """Format a float for the CLI table (``n/a`` for NaN, signed fixed precision)."""
    import math

    return "n/a" if not math.isfinite(value) else f"{value:+.4f}"


def main() -> None:
    """Console-script entrypoint: build the Typer app and run it.

    Wired to the ``rl-allocator`` console script in ``pyproject.toml``. Builds the app
    via :func:`build_app` and invokes it.

    Raises
    ------
    ImportError
        If ``typer`` (the ``[dev]`` extra) is not installed.
    """
    build_app()()


if __name__ == "__main__":  # pragma: no cover - module-as-script entrypoint
    main()
