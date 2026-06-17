"""Command-line interface (Typer): train / price / compare.

A thin orchestration layer over the compute library. Typer (and scikit-learn /
skl2onnx, on the train path) are imported LAZILY inside :func:`build_app` / the
command bodies, so importing :mod:`nnvsbs.cli` registers no commands and does no
I/O. The module-level entry point :func:`main` builds the app lazily and runs it,
backing the ``nn-vs-bs`` console script.

The three commands map to the honest workflow:

- ``train``   — run the offline synthetic-BS training pipeline and export the ONNX
  artifact (:func:`nnvsbs.train.train_synthetic_mlp`). The only command that may
  touch scikit-learn / skl2onnx, via the lazy ``[data]`` / ``[nn]`` path.
- ``price``   — price a single explicit contract ``{S, K, T, r, q, sigma}`` with the
  hand-rolled Black-Scholes kernel (and its Greeks), WITHOUT any training engine.
- ``compare`` — run the NN-vs-BS reprice comparison (:func:`nnvsbs.serve.run_compare`)
  on a synthetic grid (or a real chain), printing the reprice RMSE/MAE, the
  Diebold-Mariano result, and the honest verdict. NO P&L / Sharpe anywhere.

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import typer


def build_app() -> typer.Typer:
    """Construct and return the Typer application.

    Registers ``train``, ``price``, and ``compare`` on a fresh ``typer.Typer``.
    Typer is imported lazily inside this function so importing :mod:`nnvsbs.cli`
    does not import Typer or register any commands. A fresh instance is returned on
    every call (no shared mutable state).

    Returns
    -------
    typer.Typer
        The configured Typer application.
    """
    # LAZY import: keep Typer off the import path of this pure module.
    import typer

    cli = typer.Typer(
        name="nn-vs-bs",
        add_completion=False,
        help="Options pricing — a small neural net vs Black-Scholes, compared "
        "honestly on REPRICE error only (no tradable P&L / Sharpe). On synthetic "
        "BS data the NN can only RECOVER the BS surface (a sanity check).",
        no_args_is_help=True,
    )

    @cli.command("train")
    def _train_command(
        n_quotes: int = typer.Option(4000, help="Synthetic Black-Scholes chain size to train on."),
        out_path: str = typer.Option(
            "", help="Destination .onnx path (empty => the shipped default)."
        ),
        seed: int = typer.Option(7, help="Master RNG seed."),
    ) -> None:
        """Train the MLP on synthetic BS data and export the ONNX artifact."""
        code = train(n_quotes=n_quotes, out_path=out_path or None, seed=seed)
        raise typer.Exit(code=code)

    @cli.command("price")
    def _price_command(
        s: float = typer.Option(..., "--s", help="Spot price S (> 0)."),
        k: float = typer.Option(..., "--k", help="Strike price K (> 0)."),
        t: float = typer.Option(..., "--t", help="Time to expiry in years (> 0)."),
        r: float = typer.Option(0.03, "--r", help="Risk-free rate."),
        q: float = typer.Option(0.0, "--q", help="Continuous dividend yield."),
        sigma: float = typer.Option(0.20, "--sigma", help="Volatility (> 0)."),
        kind: str = typer.Option("call", help="Option right (call|put)."),
    ) -> None:
        """Price a single contract with the hand-rolled Black-Scholes kernel."""
        code = price(s=s, k=k, t=t, r=r, q=q, sigma=sigma, kind=kind)
        raise typer.Exit(code=code)

    @cli.command("compare")
    def _compare_command(
        mode: str = typer.Option("synthetic", help="Comparison mode (synthetic|real)."),
        ticker: str = typer.Option("SPY", help="Underlying symbol (real mode)."),
        n_quotes: int = typer.Option(500, help="Synthetic grid size (synthetic mode)."),
        seed: int = typer.Option(7, help="Master RNG seed for the synthetic grid."),
    ) -> None:
        """Compare the ONNX NN pricer against Black-Scholes (reprice error only)."""
        code = compare(mode=mode, ticker=ticker, n_quotes=n_quotes, seed=seed)
        raise typer.Exit(code=code)

    return cli


def train(*, n_quotes: int, out_path: str | None, seed: int) -> int:
    """Run the offline synthetic-BS training pipeline and export the ONNX artifact.

    Thin wrapper over :func:`nnvsbs.train.train_synthetic_mlp` (imported lazily so
    importing this module stays side-effect free and the heavy ``[data]`` / ``[nn]``
    stack is only paid for at invocation). Prints the artifact path and the
    train/test reprice RMSE — the latter is the convergence (recovers-BS) check.

    Returns
    -------
    int
        A process exit code (``0`` on success, ``1`` on a library error).
    """
    from nnvsbs._exceptions import NnVsBsError
    from nnvsbs.train import train_synthetic_mlp

    try:
        result = train_synthetic_mlp(n_quotes=n_quotes, out_path=out_path, seed=seed)
        print("nn-vs-bs train (synthetic Black-Scholes -> MLP -> ONNX)")
        print("=" * 56)
        print(f"artifact path  : {result.artifact_path}")
        print(f"n quotes       : {result.n_quotes}")
        print(f"train RMSE     : {result.train_rmse:.6f}")
        print(f"test  RMSE     : {result.test_rmse:.6f}  (convergence/recovers-BS check)")
    except NnVsBsError as exc:
        print(f"error: {exc}")
        return 1
    return 0


def price(
    *,
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
    kind: str,
) -> int:
    """Price a single explicit contract with the hand-rolled Black-Scholes kernel.

    Computes the price and the full Greek set via
    :class:`nnvsbs.pricing.black_scholes.BlackScholesGreeks` (no training engine,
    no NN). Imports are local so importing this module has no side effects.

    Returns
    -------
    int
        A process exit code (``0`` on success, ``1`` on a library error).
    """
    from nnvsbs._exceptions import NnVsBsError
    from nnvsbs.pricing.black_scholes import BlackScholesGreeks

    if kind not in ("call", "put"):
        print(f"error: kind must be 'call' or 'put', got {kind!r}.")
        return 1

    try:
        greeks = BlackScholesGreeks.compute(s, k, t, r, q, sigma, kind=kind)  # type: ignore[arg-type]
        print("nn-vs-bs price (hand-rolled Black-Scholes)")
        print("=" * 42)
        print(f"contract       : S={s} K={k} T={t} r={r} q={q} sigma={sigma} ({kind})")
        print(f"price          : {greeks.price:.6f}")
        print(f"delta          : {greeks.delta:.6f}")
        print(f"gamma          : {greeks.gamma:.6f}")
        print(f"vega           : {greeks.vega:.6f}")
        print(f"theta          : {greeks.theta:.6f}")
        print(f"rho            : {greeks.rho:.6f}")
    except NnVsBsError as exc:
        print(f"error: {exc}")
        return 1
    return 0


def compare(*, mode: str, ticker: str, n_quotes: int, seed: int) -> int:
    """Run the NN-vs-BS reprice comparison and print the honest summary.

    Thin wrapper over :func:`nnvsbs.serve.run_compare` (imported lazily). Prints the
    reprice RMSE/MAE for both pricers, the Diebold-Mariano result, and the honest
    verdict. There is deliberately NO P&L / Sharpe / DSR in the output.

    Returns
    -------
    int
        A process exit code (``0`` on success, ``1`` on a library error).
    """
    from typing import cast

    from nnvsbs._exceptions import NnVsBsError
    from nnvsbs.serve import CompareMode, run_compare

    if mode not in ("synthetic", "real"):
        print(f"error: mode must be 'synthetic' or 'real', got {mode!r}.")
        return 1

    try:
        run = run_compare(
            mode=cast("CompareMode", mode),
            ticker=ticker,
            n_quotes=n_quotes,
            seed=seed,
        )
        summary = run.summary
        # Header deliberately avoids the literal words P&L / Sharpe: the tool reports
        # REPRICE error only, and no tradable statistic ever appears in the output.
        print("nn-vs-bs compare (reprice error only — no tradable statistic)")
        print("=" * 60)
        print(f"data source    : {summary.data_source}")
        print(f"n quotes       : {summary.n_quotes}")
        print(f"BS  reprice RMSE: {summary.bs_rmse:.6f}   MAE: {summary.mae_bs:.6f}")
        print(f"NN  reprice RMSE: {summary.nn_rmse:.6f}   MAE: {summary.mae_nn:.6f}")
        print(f"DM statistic   : {summary.dm_stat:.4f}")
        print(f"DM p-value     : {summary.dm_pvalue:.4f}")
        print(f"NN recovers BS : {summary.nn_recovers_bs}")
        print(f"verdict        : {summary.verdict}")
        if summary.rationale:
            print(f"rationale      : {summary.rationale}")
    except NnVsBsError as exc:
        print(f"error: {exc}")
        return 1
    return 0


def main() -> None:
    """Console-script entry point: build the Typer app lazily and run it.

    Backs the ``nn-vs-bs`` script declared in ``pyproject.toml``. Importing this
    module does not call this function, so import stays side-effect-free.
    """
    build_app()()
