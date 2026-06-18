"""Typer CLI: ``train`` / ``forecast`` / ``compare`` (typer imported lazily).

The console entrypoint (``mvts-forecast``) exposes three commands:

- ``train``   — the OFFLINE pipeline (synthetic -> walk-forward -> train deep
  models -> ONNX + metrics.json); the only command that pulls in the ``[train]``
  extra (torch).
- ``forecast`` — score the live, torch-free baselines (naive + ARIMA) on a panel
  and print the per-model return-space metric table.
- ``compare``  — the full deep-vs-naive comparison with the PURE
  ``deep_beats_naive`` verdict, derived honestly from the inference outputs.

``typer`` is imported LAZILY inside :func:`build_app` (it lives in the ``[dev]``
extra), so importing this module pulls in NO typer and has no side effects. The
deep models stay lazy too: ``forecast`` and ``compare`` compute the naive/ARIMA
baselines with pure numpy/statsmodels and only attempt the committed ONNX deep
models when their artifacts exist (onnxruntime, NEVER torch). The ``main``
entrypoint builds and runs the app.

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import typer

    from mvtsforecast._typing import FloatArray

# The deep models served live (torch-free) from committed ONNX artifacts.
_DEEP_MODELS: tuple[str, ...] = ("lstm", "patchtst", "transformer")
#: Default model roster for ``compare`` (the live baselines + the deep roster).
_ALL_MODELS: tuple[str, ...] = ("naive", "arima", *_DEEP_MODELS)


def build_app() -> typer.Typer:
    """Build and return the Typer application (``typer`` imported lazily here).

    Registers the ``train``, ``forecast``, and ``compare`` commands. Typer is
    imported inside this function so importing :mod:`mvtsforecast.cli` (and the
    package) never imports typer. A fresh instance is returned on every call (no
    shared mutable state).

    Returns
    -------
    typer.Typer
        A configured ``typer.Typer`` instance.

    Raises
    ------
    ImportError
        If ``typer`` (the ``[dev]`` extra) is not installed.
    """
    # LAZY import: keep typer off the import path of this pure module.
    import typer

    cli = typer.Typer(
        name="mvts-forecast",
        add_completion=False,
        help=(
            "Leakage-free multivariate-transformer forecast benchmark (honest "
            "NULL). Benchmarks PatchTST and an interpretable transformer against "
            "LSTM, ARIMA and a naive random-walk baseline in RETURN space, with "
            "RevIN + purged walk-forward and Diebold-Mariano / Deflated-Sharpe "
            "honesty. Spoiler: deep models don't reliably beat naive."
        ),
        no_args_is_help=True,
    )

    @cli.command("train")
    def _train_command(
        basket: str = typer.Option(
            "SPY,TLT,GLD", help="Comma-separated basket tickers for the synthetic panel."
        ),
        target: str = typer.Option("SPY", help="Forecast target column (must be in --basket)."),
        lookback: int = typer.Option(60, help="Window length / minimum purge size."),
        horizon: int = typer.Option(1, help="Forecast horizon in steps (1 or 5)."),
        n_obs: int = typer.Option(1500, help="Synthetic observations to generate."),
        seed: int = typer.Option(7, help="Master RNG/torch seed."),
    ) -> None:
        """Run the OFFLINE train -> ONNX-export -> metrics pipeline (the [train] extra)."""
        code = train(
            basket=basket,
            target=target,
            lookback=lookback,
            horizon=horizon,
            n_obs=n_obs,
            seed=seed,
        )
        raise typer.Exit(code=code)

    @cli.command("forecast")
    def _forecast_command(
        basket: str = typer.Option(
            "SPY,TLT,GLD", help="Comma-separated basket tickers for the synthetic panel."
        ),
        target: str = typer.Option("SPY", help="Forecast target column (must be in --basket)."),
        lookback: int = typer.Option(60, help="Window length."),
        horizon: int = typer.Option(1, help="Forecast horizon in steps (1 or 5)."),
        n_obs: int = typer.Option(400, help="Synthetic observations to generate."),
        seed: int = typer.Option(7, help="Master RNG seed for the synthetic panel."),
    ) -> None:
        """Score the live, torch-free baselines (naive + ARIMA) on a synthetic panel."""
        code = forecast(
            basket=basket,
            target=target,
            lookback=lookback,
            horizon=horizon,
            n_obs=n_obs,
            seed=seed,
        )
        raise typer.Exit(code=code)

    @cli.command("compare")
    def _compare_command(
        basket: str = typer.Option(
            "SPY,TLT,GLD", help="Comma-separated basket tickers for the synthetic panel."
        ),
        target: str = typer.Option("SPY", help="Forecast target column (must be in --basket)."),
        models: str = typer.Option(
            "naive,arima",
            help="Comma-separated model subset of {naive,arima,lstm,patchtst,transformer}.",
        ),
        lookback: int = typer.Option(60, help="Window length."),
        horizon: int = typer.Option(1, help="Forecast horizon in steps (1 or 5)."),
        n_obs: int = typer.Option(400, help="Synthetic observations to generate."),
        seed: int = typer.Option(7, help="Master RNG seed for the synthetic panel."),
    ) -> None:
        """Run the deep-vs-naive comparison and print the PURE ``deep_beats_naive`` verdict."""
        code = compare(
            basket=basket,
            target=target,
            models=models,
            lookback=lookback,
            horizon=horizon,
            n_obs=n_obs,
            seed=seed,
        )
        raise typer.Exit(code=code)

    return cli


def _parse_list(value: str) -> list[str]:
    """Split a comma-separated option into a list of stripped, non-empty tokens."""
    return [token.strip() for token in value.split(",") if token.strip()]


def _target_returns(
    *,
    basket: str,
    target: str,
    n_obs: int,
    seed: int,
) -> tuple[pd.Series, str]:
    """Build the seeded synthetic panel and return the target's RETURN series.

    Returns the target column together with a ``data_source`` provenance tag
    (always ``"synthetic"`` here — the deployed default). Pure numpy/pandas; no
    torch / onnxruntime is touched.
    """
    from mvtsforecast._exceptions import ValidationError
    from mvtsforecast.data.synthetic import synthetic_panel

    tickers = _parse_list(basket)
    if not tickers:
        raise ValidationError("basket must contain at least one ticker.")
    if target not in tickers:
        raise ValidationError(f"target '{target}' must be one of the basket tickers {tickers}.")

    panel = synthetic_panel(tickers, n_obs=n_obs, seed=seed)
    return panel[target], "synthetic"


def _split_returns(series: pd.Series, lookback: int) -> tuple[FloatArray, FloatArray]:
    """Split a return series into a (train, test) pair using a single anchored fold.

    The first ``max(lookback, n // 2)`` observations train the baselines; the
    remainder is the OOS test set the forecasts are scored against. Keeping the
    purge >= ``lookback`` mirrors the walk-forward guard so the test window never
    straddles the (deep) input window.
    """
    import numpy as np

    arr = np.asarray(series.to_numpy(), dtype="float64").ravel()
    n = int(arr.size)
    split = max(int(lookback), n // 2)
    split = min(split, n - 1)
    return arr[:split], arr[split:]


def _baseline_forecasts(
    train: FloatArray,
    test: FloatArray,
    models: list[str],
) -> dict[str, FloatArray]:
    """Compute the live, torch-free baseline forecasts requested in ``models``.

    Only ``naive`` and ``arima`` run here (pure numpy / statsmodels). Deep models
    are handled separately via their committed ONNX artifacts. ``naive`` is always
    included so it can anchor the Diebold-Mariano comparison and the verdict.
    """
    from mvtsforecast.models.arima import arima_forecast
    from mvtsforecast.models.naive import naive_forecast

    out: dict[str, FloatArray] = {"naive": naive_forecast(test).forecast}
    if "arima" in models:
        out["arima"] = arima_forecast(train, int(test.size)).forecast
    return out


def _onnx_forecasts(
    requested: list[str],
    x: FloatArray | None,
) -> dict[str, FloatArray]:
    """Serve any requested deep models from committed ONNX artifacts (NO torch).

    Only models whose ``.onnx`` artifact is present are served; missing artifacts
    are skipped silently (the deployed default ships them, but a fresh checkout or
    a baseline-only ``compare`` need not). onnxruntime is imported lazily inside
    :func:`mvtsforecast.serve.forecast_from_onnx`; torch is never imported.
    """
    wanted = [m for m in requested if m in _DEEP_MODELS]
    if not wanted or x is None:
        return {}

    from mvtsforecast._exceptions import MvtsForecastError
    from mvtsforecast.models.onnx_runtime import default_artifact_path
    from mvtsforecast.serve import forecast_from_onnx

    out: dict[str, FloatArray] = {}
    for model in wanted:
        if not default_artifact_path(model).is_file():
            continue
        try:
            out[model] = forecast_from_onnx(model, x)
        except MvtsForecastError:
            # A corrupt / signature-mismatched artifact is non-fatal for the CLI:
            # skip the model rather than abort the whole comparison.
            continue
    return out


def train(
    *,
    basket: str = "SPY,TLT,GLD",
    target: str = "SPY",
    lookback: int = 60,
    horizon: int = 1,
    n_obs: int = 1500,
    seed: int = 7,
) -> int:
    """Run the OFFLINE training pipeline from the command line.

    Delegates to :func:`mvtsforecast.train.train_pipeline` (which lazily imports
    torch / the ONNX exporter — the ``[train]`` extra). Prints the honest summary:
    the exported artifact paths, the FULL-grid effective-trial count, and the
    (expected ``False``) ``deep_beats_naive`` verdict.

    Parameters
    ----------
    basket:
        Comma-separated basket tickers for the synthetic panel.
    target:
        The forecast target column (must be in ``basket``).
    lookback:
        Window length / minimum purge size.
    horizon:
        Forecast horizon in steps (1 or 5).
    n_obs:
        Synthetic observations to generate.
    seed:
        Master RNG/torch seed.

    Returns
    -------
    int
        Process exit code (``0`` on success, ``1`` on a library error).
    """
    from mvtsforecast._exceptions import MvtsForecastError
    from mvtsforecast.train import train_pipeline

    tickers = _parse_list(basket)
    try:
        result = train_pipeline(
            basket=tickers,
            target=target,
            lookback=lookback,
            horizon=horizon,
            n_obs=n_obs,
            seed=seed,
        )
    except MvtsForecastError as exc:
        print(f"error: {exc}")
        return 1

    print("mvts-forecast training run")
    print("=" * 48)
    print(f"target             : {target}")
    print(f"effective trials   : {result.n_effective_trials}")
    print(f"metrics            : {result.metrics_path}")
    for name, path in result.artifact_paths.items():
        print(f"artifact[{name:<11}]: {path}")
    print(f"deep beats naive   : {result.deep_beats_naive}")
    return 0


def forecast(
    *,
    basket: str = "SPY,TLT,GLD",
    target: str = "SPY",
    lookback: int = 60,
    horizon: int = 1,
    n_obs: int = 400,
    seed: int = 7,
) -> int:
    """Score the live, torch-free baselines (naive + ARIMA) and print their metrics.

    Builds the seeded synthetic panel, takes the target's RETURN series, splits it
    into an anchored (train, test) fold, computes the naive and ARIMA forecasts
    LIVE (pure numpy / statsmodels), and prints the per-model return-space metric
    table (RMSE / MAE / MASE-vs-naive / directional accuracy). NO torch, NO
    price-level R². On the synthetic null these baselines hug the OOS floor.

    Parameters
    ----------
    basket:
        Comma-separated basket tickers for the synthetic panel.
    target:
        The forecast target column (must be in ``basket``).
    lookback:
        Window length.
    horizon:
        Forecast horizon in steps (informational on the live baselines).
    n_obs:
        Synthetic observations to generate.
    seed:
        Master RNG seed for the synthetic panel.

    Returns
    -------
    int
        Process exit code (``0`` on success, ``1`` on a library error).
    """
    from mvtsforecast._exceptions import MvtsForecastError
    from mvtsforecast.evaluation.metrics import forecast_metrics

    try:
        series, source = _target_returns(basket=basket, target=target, n_obs=n_obs, seed=seed)
        train_returns, test_returns = _split_returns(series, lookback)
        forecasts = _baseline_forecasts(train_returns, test_returns, ["naive", "arima"])
    except MvtsForecastError as exc:
        print(f"error: {exc}")
        return 1

    naive = forecasts["naive"]
    print("mvts-forecast baseline forecast (return space; NO price-level R^2)")
    print("=" * 72)
    print(f"data source        : {source}")
    print(f"target             : {target}")
    print(f"horizon            : {horizon}")
    print(f"OOS observations   : {int(test_returns.size)}")
    print(f"{'model':<12} {'RMSE':>12} {'MAE':>12} {'MASE':>10} {'dir.acc':>10}")
    for model in _ordered(forecasts):
        metrics = forecast_metrics(test_returns, forecasts[model], naive)
        print(
            f"{model:<12} {metrics.rmse_return:>12.6f} {metrics.mae_return:>12.6f} "
            f"{metrics.mase_vs_naive:>10.4f} {metrics.directional_accuracy:>10.4f}"
        )
    return 0


def compare(
    *,
    basket: str = "SPY,TLT,GLD",
    target: str = "SPY",
    models: str = "naive,arima",
    lookback: int = 60,
    horizon: int = 1,
    n_obs: int = 400,
    seed: int = 7,
) -> int:
    """Run the deep-vs-naive comparison and print the PURE ``deep_beats_naive`` verdict.

    Computes the requested baselines LIVE (naive + ARIMA, torch-free) and serves
    any requested deep models from their committed ONNX artifacts (onnxruntime,
    NEVER torch) when present. Scores every model in RETURN space, runs the
    Diebold-Mariano test of each model vs. naive, and derives the honest verdict
    via :func:`mvtsforecast.evaluation.verdict.derive_verdict` — ``deep_beats_naive``
    is ``False`` unless a deep model beats naive with a DM-significant margin AND a
    positive Deflated Sharpe. On the synthetic null the verdict is ``False`` by
    construction.

    Parameters
    ----------
    basket:
        Comma-separated basket tickers for the synthetic panel.
    target:
        The forecast target column (must be in ``basket``).
    models:
        Comma-separated subset of
        ``{naive, arima, lstm, patchtst, transformer}``.
    lookback:
        Window length.
    horizon:
        Forecast horizon in steps (informational on the live baselines).
    n_obs:
        Synthetic observations to generate.
    seed:
        Master RNG seed for the synthetic panel.

    Returns
    -------
    int
        Process exit code (``0`` on success, ``1`` on a library error).
    """
    from mvtsforecast._exceptions import MvtsForecastError, ValidationError
    from mvtsforecast.evaluation.diebold_mariano import diebold_mariano
    from mvtsforecast.evaluation.metrics import directional_accuracy, rmse
    from mvtsforecast.evaluation.verdict import derive_verdict

    requested = _parse_list(models)
    unknown = [m for m in requested if m not in _ALL_MODELS]
    if unknown:
        print(f"error: unknown model(s) {unknown}; choose from {list(_ALL_MODELS)}.")
        return 1

    try:
        series, source = _target_returns(basket=basket, target=target, n_obs=n_obs, seed=seed)
        train_returns, test_returns = _split_returns(series, lookback)
        forecasts = _baseline_forecasts(train_returns, test_returns, requested)
        # Deep models are served only when their committed ONNX artifacts exist.
        # The CLI does not re-window for ONNX here (no torch); the deployed
        # backend path owns that. ``x=None`` keeps this comparison torch-free.
        forecasts.update(_onnx_forecasts(requested, None))
    except MvtsForecastError as exc:
        print(f"error: {exc}")
        return 1

    naive = forecasts["naive"]
    rmse_by_model: dict[str, float] = {}
    dir_acc_by_model: dict[str, float] = {}
    dm_pvalue_by_model: dict[str, float] = {}
    best_dm_stat = 0.0
    best_dm_pvalue = 1.0
    best_deep = ""
    for model in _ordered(forecasts):
        pred = forecasts[model]
        rmse_by_model[model] = rmse(test_returns, pred)
        dir_acc_by_model[model], _ = directional_accuracy(test_returns, pred)
        if model == "naive":
            dm_pvalue_by_model[model] = 1.0
            continue
        try:
            dm_stat, dm_pvalue = diebold_mariano(test_returns, pred, naive)
        except ValidationError:
            # A degenerate loss differential (e.g. identical forecasts) leaves the
            # DM test undefined; treat it as insignificant for the honest verdict.
            dm_stat, dm_pvalue = 0.0, 1.0
        dm_pvalue_by_model[model] = dm_pvalue
        # Track the best (most negative DM stat => model loss below naive) DEEP
        # model to drive the pure verdict.
        if model in _DEEP_MODELS and dm_stat < best_dm_stat:
            best_dm_stat, best_dm_pvalue, best_deep = dm_stat, dm_pvalue, model

    best_model = min(rmse_by_model, key=lambda m: rmse_by_model[m])
    # No deep model was served => no DSR to read; the verdict stays False (the
    # honest null). When the committed artifacts ship, the backend supplies the
    # FULL-grid DSR; the CLI keeps a conservative DSR=0.0 here.
    verdict = derive_verdict(
        best_deep or "none",
        best_dm_stat,
        best_dm_pvalue,
        deflated_sharpe=0.0,
        n_effective_trials=max(1, len(_DEEP_MODELS)),
    )

    print("mvts-forecast comparison (return space; PURE deep_beats_naive verdict)")
    print("=" * 72)
    print(f"data source        : {source}")
    print(f"target             : {target}")
    print(f"OOS observations   : {int(test_returns.size)}")
    print(f"{'model':<12} {'RMSE':>12} {'dir.acc':>10} {'DM p vs naive':>15}")
    for model in _ordered(forecasts):
        print(
            f"{model:<12} {rmse_by_model[model]:>12.6f} {dir_acc_by_model[model]:>10.4f} "
            f"{dm_pvalue_by_model[model]:>15.4f}"
        )
    print("-" * 72)
    print(f"best model         : {best_model}")
    print(f"deep beats naive   : {verdict.deep_beats_naive}")
    print(f"verdict            : {verdict.verdict.value}")
    return 0


def _ordered(forecasts: dict[str, FloatArray]) -> list[str]:
    """Return forecast model keys in a stable display order (naive first)."""
    order = [m for m in _ALL_MODELS if m in forecasts]
    extra = [m for m in forecasts if m not in order]
    return order + extra


def main() -> None:
    """Console-script entrypoint: build the Typer app and run it.

    Wired to the ``mvts-forecast`` console script in ``pyproject.toml``. Builds the
    app via :func:`build_app` and invokes it.

    Raises
    ------
    ImportError
        If ``typer`` (the ``[dev]`` extra) is not installed.
    """
    build_app()()


if __name__ == "__main__":  # pragma: no cover - module-as-script entrypoint
    main()
