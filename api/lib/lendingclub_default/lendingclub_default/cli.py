"""Command-line interface (Typer).

A thin orchestration layer over the compute library: train the model (on a real
Kaggle CSV or the synthetic generator), score a single loan application, or
evaluate a held-out panel. Typer is built on the standard library, but
constructing the app object is deferred to :func:`build_app` so importing this
module has no side effects (no command registration or I/O at import time). The
module-level ``app`` is a lazily-built singleton consumed by the
``lendingclub-default`` console-script entry point.

The ``score`` and ``evaluate`` commands load the bundle written by ``train``
through the library's public :func:`lendingclub_default.train.load_artifacts`
(a fitted pipeline + booster + calibration map) and score via
:meth:`lendingclub_default.train.ScoredArtifacts.score`, so the CLI never
re-implements the inference path.

Importing this module has no side effects.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import typer

# quantcore-candidate: new code (Typer CLI); lazy Typer import.


def build_app() -> typer.Typer:
    """Construct and return the Typer application.

    Registers the ``train``, ``score``, and ``evaluate`` commands on a fresh
    ``typer.Typer`` instance. Typer is imported lazily inside this function so
    that importing :mod:`lendingclub_default.cli` does not import Typer or
    register any commands.

    Returns
    -------
    typer.Typer
        The configured Typer application.
    """
    # LAZY import: keep Typer off the import path of this pure module.
    import typer

    cli = typer.Typer(
        name="lendingclub-default",
        add_completion=False,
        help=(
            "Leakage-free, calibrated credit-default classifier. Ranks a loan "
            "application's probability of default at origination — honest "
            "ROC-AUC / PR-AUC / Brier, never accuracy or profit. The shipped "
            "demo model is trained on a synthetic LC-schema panel."
        ),
        no_args_is_help=True,
    )

    @cli.command("train")
    def _train_command(
        data: str | None = typer.Option(
            None,
            "--data",
            help="Path to a real LendingClub accepted-loans CSV. Omit to use the "
            "reproducible synthetic generator (the shipped-artifact default).",
        ),
        out_dir: str = typer.Option(
            "artifacts",
            "--out-dir",
            help="Directory to write the booster JSON, fitted pipeline, "
            "calibration map, and run manifest.",
        ),
        seed: int = typer.Option(20260616, "--seed", help="Master RNG seed for the run."),
    ) -> None:
        """Train the end-to-end leakage-free, temporally-split, calibrated model."""
        code = train_command(data=data, out_dir=out_dir, seed=seed)
        raise typer.Exit(code=code)

    @cli.command("score")
    def _score_command(
        artifacts_dir: str = typer.Option(
            "artifacts",
            "--artifacts-dir",
            help="Directory holding the trained bundle (from `train`).",
        ),
        application: str | None = typer.Option(
            None,
            "--application",
            help="Path to a JSON file with one loan application's "
            "application-time fields. Omit to score a built-in demo application.",
        ),
    ) -> None:
        """Score a single loan application: calibrated PD, risk decile, decision."""
        code = score_command(artifacts_dir=artifacts_dir, application=application)
        raise typer.Exit(code=code)

    @cli.command("evaluate")
    def _evaluate_command(
        artifacts_dir: str = typer.Option(
            "artifacts",
            "--artifacts-dir",
            help="Directory holding the trained bundle (from `train`).",
        ),
        data: str | None = typer.Option(
            None,
            "--data",
            help="Path to a panel CSV to evaluate on. Omit to evaluate on a "
            "freshly generated synthetic held-out panel.",
        ),
    ) -> None:
        """Evaluate a trained model on a held-out panel (ROC-AUC / PR-AUC / Brier)."""
        code = evaluate_command(artifacts_dir=artifacts_dir, data=data)
        raise typer.Exit(code=code)

    return cli


# A built-in, schema-complete demo application used when `score` is invoked
# without an --application file, so the command is runnable out of the box.
_DEMO_APPLICATION: dict[str, Any] = {
    "loan_amnt": 15_000.0,
    "term": "36 months",
    "int_rate": 13.5,
    "grade": "C",
    "sub_grade": "C2",
    "emp_length": 5.0,
    "home_ownership": "MORTGAGE",
    "annual_inc": 65_000.0,
    "dti": 18.4,
    "fico_range_low": 690.0,
    "fico_range_high": 694.0,
    "revol_util": 42.7,
    "open_acc": 11.0,
    "pub_rec": 0.0,
    "purpose": "debt_consolidation",
    "addr_state": "CA",
    "installment": 509.0,
    "verification_status": "Source Verified",
}


def train_command(*, data: str | None, out_dir: str, seed: int) -> int:
    """Run the training pipeline and print the headline summary.

    Parameters
    ----------
    data:
        Real LendingClub CSV path, or ``None`` for the synthetic generator.
    out_dir:
        Output directory for the booster, pipeline, calibration map, and manifest.
    seed:
        Master RNG seed.

    Returns
    -------
    int
        Process exit code (``0`` on success, ``1`` on a handled library error).
    """
    from lendingclub_default._exceptions import LendingClubDefaultError
    from lendingclub_default.train import train

    try:
        artifacts = train(
            data_path=Path(data) if data is not None else None,
            out_dir=Path(out_dir),
            seed=seed,
        )
    except LendingClubDefaultError as exc:  # pragma: no cover - defensive
        print(f"error: {exc}")
        return 1

    metrics = artifacts.metrics
    logistic = artifacts.logistic_metrics
    print("LendingClub default — training run")
    print("=" * 44)
    print(f"data source        : {artifacts.data_source}")
    print(f"booster            : {artifacts.booster_path}")
    print(f"pipeline           : {artifacts.pipeline_path}")
    print(f"held-out n         : {metrics.n}")
    print(f"base rate          : {metrics.base_rate:.4f}")
    print(f"XGB  ROC-AUC       : {metrics.roc_auc:.4f}")
    print(f"XGB  PR-AUC        : {metrics.pr_auc:.4f}")
    print(f"XGB  Brier         : {metrics.brier:.4f}")
    print(f"XGB  KS            : {metrics.ks:.4f}")
    print(f"Logit ROC-AUC      : {logistic.roc_auc:.4f}")
    print(f"base-rate AUC      : {artifacts.base_rate_auc:.4f}")
    print(f"DeLong p-value     : {artifacts.delong.p_value:.4f}")
    print(f"n_trials (recorded): {artifacts.n_trials}")
    print("note: ranks risk; does not predict individuals. Demo artifact is synthetic-trained.")
    return 0


def score_command(*, artifacts_dir: str, application: str | None) -> int:
    """Score a single loan application from a trained bundle.

    Parameters
    ----------
    artifacts_dir:
        Directory holding the bundle written by ``train``.
    application:
        JSON file with one application's application-time fields, or ``None`` to
        score the built-in demo application.

    Returns
    -------
    int
        Process exit code (``0`` on success, ``1`` on a handled error).
    """
    import json

    from lendingclub_default._exceptions import LendingClubDefaultError

    if application is not None:
        app_path = Path(application)
        if not app_path.exists():
            print(f"error: application file does not exist: {app_path}.")
            return 1
        record = json.loads(app_path.read_text())
    else:
        record = dict(_DEMO_APPLICATION)

    try:
        result = _score_application(Path(artifacts_dir), record)
    except LendingClubDefaultError as exc:
        print(f"error: {exc}")
        return 1

    print("LendingClub default — single-application score")
    print("=" * 44)
    print(f"calibrated PD      : {result['pd']:.4f}")
    print(f"risk decile        : {result['decile']} / 10 (1 = safest)")
    print(f"predicted label    : {result['predicted_label']}")
    print(f"threshold          : {result['threshold']:.4f}")
    print(f"model ROC-AUC      : {result['model_auc']:.4f}")
    print(f"base rate          : {result['base_rate']:.4f}")
    reason_codes = result.get("reason_codes", [])
    if reason_codes:
        print("reason codes (top contributors):")
        for code in reason_codes:
            print(f"  - {code['feature']:<24} {code['direction']:<10} {code['contribution']:+.4f}")
    print("note: ranks risk; does not predict whether this individual defaults.")
    return 0


def evaluate_command(*, artifacts_dir: str, data: str | None) -> int:
    """Evaluate a trained model on a held-out panel; print the metric bundle.

    Parameters
    ----------
    artifacts_dir:
        Directory holding the bundle written by ``train``.
    data:
        Panel CSV to evaluate on, or ``None`` for a fresh synthetic panel.

    Returns
    -------
    int
        Process exit code (``0`` on success, ``1`` on a handled error).
    """
    from lendingclub_default._exceptions import LendingClubDefaultError

    try:
        metrics = _evaluate_panel(Path(artifacts_dir), Path(data) if data is not None else None)
    except LendingClubDefaultError as exc:
        print(f"error: {exc}")
        return 1

    print("LendingClub default — held-out evaluation")
    print("=" * 44)
    print(f"held-out n         : {metrics.n}")
    print(f"base rate          : {metrics.base_rate:.4f}")
    print(f"ROC-AUC            : {metrics.roc_auc:.4f}")
    print(f"PR-AUC             : {metrics.pr_auc:.4f}")
    print(f"Brier              : {metrics.brier:.4f}")
    print(f"log-loss           : {metrics.log_loss:.4f}")
    print(f"KS                 : {metrics.ks:.4f}")
    print("note: out-of-time (vintage) evaluation; ranks risk, no profit claim.")
    return 0


def _score_application(artifacts_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Score one raw application record into a calibrated PD + risk decile + label.

    Delegates loading and scoring to the library's public scoring entrypoint
    :meth:`lendingclub_default.train.ScoredArtifacts.score_one`, so the inference
    path (and the reason codes) is never re-implemented in the CLI.

    Raises
    ------
    ArtifactError
        If the trained bundle is missing or malformed.
    """
    from lendingclub_default.train import load_artifacts

    scorer = load_artifacts(artifacts_dir)
    return scorer.score_one(record)


def _evaluate_panel(artifacts_dir: Path, data_path: Path | None) -> Any:
    """Score a held-out panel with the trained bundle and return its MetricBundle.

    Raises
    ------
    ArtifactError
        If the trained bundle is missing or malformed.
    """
    from lendingclub_default.data.labels import build_labels
    from lendingclub_default.data.load import load_panel
    from lendingclub_default.evaluation.metrics import compute_metrics
    from lendingclub_default.train import load_artifacts

    scorer = load_artifacts(artifacts_dir)
    panel = load_panel(data_path)
    labelled = build_labels(panel)
    probs = scorer.score(labelled.panel)
    return compute_metrics(labelled.y, probs)


def app() -> None:
    """Console-script entry point: build the Typer app and invoke it.

    Wraps :func:`build_app` so ``lendingclub-default ...`` on the command line
    runs the CLI. Kept as a function (not a module-level Typer instance) to
    preserve import purity.
    """
    build_app()()
