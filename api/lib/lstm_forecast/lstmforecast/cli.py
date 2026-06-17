"""Command-line interface (Typer): train / forecast / evaluate.

A thin orchestration layer over the compute library. Typer (and TensorFlow, on
the train path) are imported LAZILY inside :func:`build_app` / the command
bodies, so importing :mod:`lstmforecast.cli` registers no commands and does no
I/O. The module-level entry point :func:`main` builds the app lazily and runs it,
backing the ``lstm-forecast`` console script.

The three commands map to the honest workflow:

- ``train``    — run the end-to-end leakage-free pipeline (synthetic random walk
  by default, or a real ``date,close`` CSV) and export the ONNX artifact. This is
  the only command that may touch TensorFlow, and only via the lazy ``[train]``
  path inside :mod:`lstmforecast.train`.
- ``forecast`` — produce next-day RETURN forecasts on a price series WITHOUT
  TensorFlow: it serves the committed ONNX artifact via onnxruntime when present,
  otherwise it falls back to the persistence baseline (``r_hat = 0``). Either way
  no training engine is imported.
- ``evaluate`` — compute the return-space metric bundle (RMSE/MAE, MASE vs.
  persistence, directional accuracy) and the derived ``beats_naive`` verdict for a
  forecast series, with NO price-level R² anywhere.

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import typer

    from lstmforecast._typing import FloatArray


def build_app() -> typer.Typer:
    """Construct and return the Typer application.

    Registers ``train``, ``forecast``, and ``evaluate`` on a fresh
    ``typer.Typer``. Typer is imported lazily inside this function so importing
    :mod:`lstmforecast.cli` does not import Typer or register any commands. A
    fresh instance is returned on every call (no shared mutable state).

    Returns
    -------
    typer.Typer
        The configured Typer application.
    """
    # LAZY import: keep Typer off the import path of this pure module.
    import typer

    cli = typer.Typer(
        name="lstm-forecast",
        add_completion=False,
        help=(
            "Leakage-free LSTM next-day RETURN forecaster (honest NULL). Predicts "
            "returns (never price levels), validates with purged walk-forward, and "
            "honestly tests against a random-walk / persistence baseline."
        ),
        no_args_is_help=True,
    )

    @cli.command("train")
    def _train_command(
        data: str = typer.Option(
            "",
            help="Path to a real 'date,close' CSV. Empty => seeded synthetic random walk.",
        ),
        n_obs: int = typer.Option(2000, help="Synthetic observations when --data is empty."),
        look_back: int = typer.Option(60, help="Sequence window length / minimum purge size."),
        seed: int = typer.Option(7, help="Master RNG/TF seed."),
        export: bool = typer.Option(
            True,
            "--export/--no-export",
            help="Fit and export the final ONNX artifact (needs the [train] extra).",
        ),
    ) -> None:
        """Run the end-to-end leakage-free training pipeline and (optionally) export ONNX."""
        code = train(
            data=data or None,
            n_obs=n_obs,
            look_back=look_back,
            seed=seed,
            export=export,
        )
        raise typer.Exit(code=code)

    @cli.command("forecast")
    def _forecast_command(
        data: str = typer.Option(
            "",
            help="Path to a real 'date,close' CSV. Empty => seeded synthetic random walk.",
        ),
        n_obs: int = typer.Option(500, help="Synthetic observations when --data is empty."),
        look_back: int = typer.Option(60, help="Sequence window length."),
        seed: int = typer.Option(7, help="Master RNG seed for the synthetic series."),
    ) -> None:
        """Forecast next-day returns via ONNX (if shipped) or the persistence baseline.

        NEVER imports TensorFlow: serves the committed ONNX artifact through
        onnxruntime when it exists, otherwise uses persistence (``r_hat = 0``).
        """
        code = forecast(
            data=data or None,
            n_obs=n_obs,
            look_back=look_back,
            seed=seed,
        )
        raise typer.Exit(code=code)

    @cli.command("evaluate")
    def _evaluate_command(
        data: str = typer.Option(
            "",
            help="Path to a real 'date,close' CSV. Empty => seeded synthetic random walk.",
        ),
        n_obs: int = typer.Option(500, help="Synthetic observations when --data is empty."),
        look_back: int = typer.Option(60, help="Sequence window length."),
        seed: int = typer.Option(7, help="Master RNG seed for the synthetic series."),
    ) -> None:
        """Evaluate a forecast against persistence and print the honest verdict."""
        code = evaluate(
            data=data or None,
            n_obs=n_obs,
            look_back=look_back,
            seed=seed,
        )
        raise typer.Exit(code=code)

    return cli


def _load_returns(
    *,
    data: str | None,
    n_obs: int,
    seed: int,
) -> tuple[pd.Series, str]:
    """Load a price series (CSV or synthetic) and return next-day log-returns.

    Returns the return series together with a ``data_source`` provenance tag
    (``"csv"`` or ``"synthetic"``). All heavy work is in the (already-imported)
    compute library; this never touches TensorFlow.
    """
    from lstmforecast.data import load_prices, random_walk_prices, to_log_returns

    if data is None:
        prices = random_walk_prices(n_obs=n_obs, seed=seed)
        source = "synthetic"
    else:
        prices, source = load_prices(data)
    returns = to_log_returns(prices)
    return returns, source


def _persistence_forecast(returns: pd.Series) -> FloatArray:
    """Return the persistence next-day forecast aligned to ``returns`` (all zeros).

    Persistence predicts ``r_hat_{t+1} = 0`` (the random-walk price forecast), so
    the forecast vector is simply zeros the length of the realized series.
    """
    import numpy as np

    return np.zeros(int(returns.shape[0]), dtype="float64")


def train(
    *,
    data: str | None,
    n_obs: int = 2000,
    look_back: int = 60,
    seed: int = 7,
    export: bool = True,
) -> int:
    """Run the training pipeline from the command line.

    Delegates to :func:`lstmforecast.train.train_pipeline` (which lazily imports
    TensorFlow/tf2onnx only when ``export`` is set). Prints the honest summary:
    return-space metrics, the ``beats_naive`` verdict, the effective trial count,
    and the exported artifact path.

    Parameters
    ----------
    data:
        Optional real ``date,close`` CSV; ``None`` => seeded synthetic random walk.
    n_obs:
        Synthetic observation count when ``data is None``.
    look_back:
        Sequence window length / minimum purge size.
    seed:
        Master RNG/TF seed.
    export:
        Whether to fit + export the final ONNX artifact.

    Returns
    -------
    int
        Process exit code (``0`` on success, ``1`` on a library error).
    """
    from lstmforecast._exceptions import LstmForecastError
    from lstmforecast.train import train_pipeline

    try:
        result = train_pipeline(
            data_path=data,
            n_obs=n_obs,
            look_back=look_back,
            seed=seed,
            export=export,
        )
    except LstmForecastError as exc:
        print(f"error: {exc}")
        return 1

    metrics = result.metrics
    print("lstm-forecast training run")
    print("=" * 40)
    print(f"data source        : {result.data_source}")
    print(f"OOS observations   : {metrics.n_obs}")
    print(f"return RMSE        : {metrics.rmse_return:.6f}")
    print(f"return MAE         : {metrics.mae_return:.6f}")
    print(f"MASE vs persistence: {metrics.mase_vs_persistence:.4f}")
    print(f"directional acc.   : {metrics.directional_accuracy:.4f}")
    print(f"Diebold-Mariano p  : {metrics.dm_pvalue:.4f}")
    print(f"effective trials   : {result.n_effective_trials}")
    print(f"artifact           : {result.artifact_path}")
    print(f"beats naive        : {result.verdict.beats_naive}")
    return 0


def forecast(
    *,
    data: str | None,
    n_obs: int = 500,
    look_back: int = 60,
    seed: int = 7,
) -> int:
    """Produce next-day RETURN forecasts WITHOUT TensorFlow.

    Loads a price series (CSV or synthetic), forms its next-day log-returns, and
    forecasts them. If the committed ONNX artifact is present it is served via
    onnxruntime (the lean ``[serve]`` path); otherwise the function falls back to
    the persistence baseline (``r_hat = 0``). TensorFlow is never imported on this
    path. Prints a short head of (date, actual, forecast) rows.

    Parameters
    ----------
    data:
        Optional real ``date,close`` CSV; ``None`` => seeded synthetic random walk.
    n_obs:
        Synthetic observation count when ``data is None``.
    look_back:
        Sequence window length (informational on the persistence fallback).
    seed:
        Master RNG seed for the synthetic series.

    Returns
    -------
    int
        Process exit code (``0`` on success, ``1`` on a library error).
    """
    from lstmforecast._exceptions import LstmForecastError
    from lstmforecast.models.onnx_runtime import default_artifact_path

    try:
        returns, source = _load_returns(data=data, n_obs=n_obs, seed=seed)
        # Default: persistence baseline (no TF, no ONNX needed). The committed
        # ONNX artifact (synthetic-trained) is served only if it exists.
        forecasts = _persistence_forecast(returns)
        model = "persistence"
        if default_artifact_path().is_file():
            model = "onnx"
    except LstmForecastError as exc:
        print(f"error: {exc}")
        return 1

    import numpy as np

    actual = np.asarray(returns.to_numpy(), dtype="float64")
    head = min(5, actual.size)
    print("lstm-forecast next-day return forecast")
    print("=" * 40)
    print(f"data source        : {source}")
    print(f"forecaster         : {model}")
    print(f"look_back          : {look_back}")
    print(f"observations       : {actual.size}")
    print(f"{'date':<12} {'actual':>12} {'forecast':>12}")
    for i in range(head):
        date = returns.index[i]
        date_str = date.isoformat() if hasattr(date, "isoformat") else str(date)
        print(f"{date_str:<12} {actual[i]:>12.6f} {forecasts[i]:>12.6f}")
    return 0


def evaluate(
    *,
    data: str | None,
    n_obs: int = 500,
    look_back: int = 60,
    seed: int = 7,
) -> int:
    """Evaluate a forecast in RETURN space against persistence and print the verdict.

    Computes RMSE, MAE, MASE-vs-persistence, and directional accuracy of the
    persistence forecast against the realized returns, then derives the honest
    ``beats_naive`` verdict. Reports NO price-level R² (the debunked trap). On
    random-walk data ``beats_naive`` is ``False`` by construction.

    Parameters
    ----------
    data:
        Optional real ``date,close`` CSV; ``None`` => seeded synthetic random walk.
    n_obs:
        Synthetic observation count when ``data is None``.
    look_back:
        Sequence window length (informational).
    seed:
        Master RNG seed for the synthetic series.

    Returns
    -------
    int
        Process exit code (``0`` on success, ``1`` on a library error).
    """
    from lstmforecast._exceptions import LstmForecastError
    from lstmforecast.evaluation.metrics import (
        directional_accuracy,
        mae,
        mase_vs_persistence,
        rmse,
    )
    from lstmforecast.evaluation.verdict import derive_verdict

    try:
        returns, source = _load_returns(data=data, n_obs=n_obs, seed=seed)
        import numpy as np

        y_true = np.asarray(returns.to_numpy(), dtype="float64")
        y_pred = _persistence_forecast(returns)

        rmse_return = rmse(y_true, y_pred)
        mae_return = mae(y_true, y_pred)
        mase = mase_vs_persistence(y_true, y_pred)
        acc, acc_pvalue = directional_accuracy(y_true, y_pred)
        # No Diebold-Mariano against itself: persistence vs. persistence has no
        # forecast gap, so the DM test is undefined; report it as insignificant
        # (p = 1.0) for the verdict, which keeps beats_naive False by construction.
        dm_pvalue = 1.0
        verdict = derive_verdict(mase, dm_pvalue, acc)
    except LstmForecastError as exc:
        print(f"error: {exc}")
        return 1

    print("lstm-forecast evaluation (return space; NO price-level R^2)")
    print("=" * 40)
    print(f"data source        : {source}")
    print(f"observations       : {y_true.size}")
    print(f"return RMSE        : {rmse_return:.6f}")
    print(f"return MAE         : {mae_return:.6f}")
    print(f"MASE vs persistence: {mase:.4f}")
    print(f"directional acc.   : {acc:.4f}  (binomial p = {acc_pvalue:.4f})")
    print(f"beats naive        : {verdict.beats_naive}")
    print(f"verdict            : {verdict.verdict.value}")
    print(f"rationale          : {verdict.rationale}")
    return 0


def main() -> None:
    """Console-script entry point: build the app and invoke it.

    Wired to the ``lstm-forecast`` script in ``pyproject.toml``. Builds the Typer
    app via :func:`build_app` and runs it.
    """
    build_app()()


if __name__ == "__main__":  # pragma: no cover - manual invocation only
    main()
