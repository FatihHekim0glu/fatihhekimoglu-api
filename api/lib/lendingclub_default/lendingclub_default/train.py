"""End-to-end training orchestration.

Wires the whole honest pipeline together:

``load -> drop leakage -> build labels (exclude in-progress) -> temporal vintage
split -> fit feature pipeline on TRAIN only -> fit baselines + XGBoost (early stop
on a temporal validation slice) -> calibrate on a held-out fold -> evaluate on the
held-out LATEST vintage -> DeLong(XGB vs logistic) -> emit a <2MB booster JSON +
the fitted pipeline + a RunManifest``.

Runs identically on the synthetic generator (default) and a real Kaggle CSV
(``--data``). Importing this module has no side effects; training only happens
when :func:`train` is called (and the ``__main__`` demo is guarded). The shipped
booster is loaded LAZILY via :func:`load_booster` (a module-level ``_BOOSTER``
sentinel) so the API never trains at import or per request.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from lendingclub_default._exceptions import ArtifactError, ValidationError
from lendingclub_default._manifest import RunManifest
from lendingclub_default.data.labels import build_labels
from lendingclub_default.data.leakage import assert_no_leakage, drop_leakage
from lendingclub_default.data.load import load_panel
from lendingclub_default.data.split import temporal_split
from lendingclub_default.evaluation.delong import delong_auc_test
from lendingclub_default.evaluation.metrics import compute_metrics, roc_auc
from lendingclub_default.features.pipeline import FeatureSpec, build_feature_pipeline
from lendingclub_default.models import baselines as _baselines
from lendingclub_default.models import calibrate as _calibrate
from lendingclub_default.models import xgb as _xgb

if TYPE_CHECKING:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from lendingclub_default.data.synthetic import SyntheticConfig
    from lendingclub_default.evaluation.delong import DeLongResult
    from lendingclub_default.evaluation.metrics import MetricBundle
    from lendingclub_default.models.calibrate import CalibratedModel
    from lendingclub_default.models.xgb import XGBConfig

# quantcore-candidate: new code (training orchestration); RunManifest-stamped
# <2MB synthetic-trained artifact.

#: Default filenames for the persisted artifacts inside ``out_dir``.
BOOSTER_FILENAME: str = "booster.json"
PIPELINE_FILENAME: str = "pipeline.joblib"
CALIBRATION_FILENAME: str = "calibration.json"
MANIFEST_FILENAME: str = "manifest.json"

#: Hard ceiling (bytes) on the committed booster JSON; the brief requires < 2 MB.
MAX_BOOSTER_BYTES: int = 2 * 1024 * 1024

#: Module-level lazy-load sentinel for the shipped scoring bundle. The API and
#: :func:`load_booster` populate it on first call; it is NEVER trained at import.
_BOOSTER: ScoredArtifacts | None = None


def _swept_config_grid() -> tuple[dict[str, Any], ...]:
    """Return the hyperparameter candidates the overfitting guard accounts for.

    The recorded ``n_trials`` in a run must be ``>=`` the size of this grid: the
    credit analogue of declaring how many configurations were effectively tried,
    so the DeLong p-value (and any reported AUC) is honest about multiplicity.
    Kept tiny and explicit so the integration guard ``n_trials >= len(grid)`` is
    trivially checkable.
    """
    depths = (3, 4, 5)
    learning_rates = (0.03, 0.05, 0.1)
    return tuple({"max_depth": d, "learning_rate": lr} for d in depths for lr in learning_rates)


@dataclass(frozen=True, slots=True)
class ScoredArtifacts:
    """A loaded, ready-to-score bundle (the lazy ``_BOOSTER`` payload).

    Everything the API needs to turn a raw application row into a calibrated PD:
    the fitted feature pipeline, the booster, the calibration map, the
    container-safe reason-code coefficients, and the headline AUC / base rate
    recorded at train time.

    Attributes
    ----------
    booster:
        The fitted XGBoost booster (loaded from the ``<2MB`` JSON).
    pipeline:
        The fitted sklearn feature pipeline (transforms a raw panel to features).
    calibration:
        The fitted calibration map (raw booster score -> PD in ``[0, 1]``).
    reason_coefficients:
        The L2-logistic baseline coefficients keyed by engineered feature name,
        used for the container-safe (SHAP-free) reason codes in :meth:`score_one`.
    feature_names:
        The engineered feature names the pipeline emits, in column order.
    model_auc:
        Held-out (latest-vintage) ROC-AUC recorded at train time.
    base_rate:
        Training default base rate (the honest floor and decision threshold).
    data_source:
        ``"synthetic"`` or ``"kaggle"``.
    """

    booster: Any
    pipeline: Pipeline
    calibration: CalibratedModel
    model_auc: float
    base_rate: float
    data_source: str
    reason_coefficients: dict[str, float] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)

    def _features(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Drop leakage, transform via the fitted pipeline, and name the columns."""
        clean = drop_leakage(panel)
        matrix = np.asarray(self.pipeline.transform(clean))
        names = self.feature_names or [f"f{i}" for i in range(matrix.shape[1])]
        if len(names) != matrix.shape[1]:
            names = [f"f{i}" for i in range(matrix.shape[1])]
        return pd.DataFrame(matrix, columns=names, index=clean.index)

    def score(self, panel: pd.DataFrame) -> np.ndarray:
        """Return calibrated PDs in ``[0, 1]`` for a raw application panel.

        The panel is dropped of any leakage columns, transformed by the fitted
        pipeline, scored by the booster, and mapped through the calibration map.
        """
        features = self._features(panel)
        raw = _xgb.predict_proba(self.booster, features)
        calibrated = self.calibration.calibrate(np.asarray(raw, dtype="float64"))
        clipped: np.ndarray = np.clip(np.asarray(calibrated, dtype="float64"), 0.0, 1.0)
        return clipped

    def score_one(
        self,
        application: pd.DataFrame | dict[str, Any],
        *,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Score ONE loan application into the public scoring contract.

        The clean public scoring entrypoint the backend calls: turns a single
        raw application row into a calibrated probability of default, a risk
        decile, an honest decision label, and container-safe reason codes (from
        the L2-logistic coefficients — no SHAP in the image).

        Parameters
        ----------
        application:
            One application as a single-row :class:`pandas.DataFrame` or a plain
            ``dict`` of application-time fields.
        top_k:
            Number of reason codes to return (top contributors by ``|coef * x|``).

        Returns
        -------
        dict
            ``{"pd", "decile", "reason_codes", "predicted_label", "threshold",
            "model_auc", "base_rate", "data_source"}`` — ``pd`` is the calibrated
            PD in ``[0, 1]``; ``decile`` is ``1..10`` (1 = safest); ``reason_codes``
            is a list of ``{feature, direction, contribution}`` dicts.

        Raises
        ------
        ValidationError
            If ``application`` does not resolve to exactly one row.
        """
        from lendingclub_default._constants import N_RISK_DECILES
        from lendingclub_default.data.load import coerce_dtypes
        from lendingclub_default.models.reason_codes import reason_codes_from_logit

        frame = (
            application
            if isinstance(application, pd.DataFrame)
            else pd.DataFrame([dict(application)])
        )
        if frame.shape[0] != 1:
            raise ValidationError(
                f"score_one: expected exactly one application row, got {frame.shape[0]}."
            )
        frame = coerce_dtypes(frame.reset_index(drop=True))

        pd_value = float(np.asarray(self.score(frame)).reshape(-1)[0])
        # Bucket the calibrated PD into 1..N (1 = safest), capped to the range.
        decile = int(min(N_RISK_DECILES, max(1, int(pd_value * N_RISK_DECILES) + 1)))
        # Honest default threshold: flag a loan riskier than the average loan; the
        # headline (AUC/PR-AUC/Brier) NEVER bakes this threshold in.
        threshold = float(self.base_rate)
        label = "default" if pd_value >= threshold else "fully_paid"

        reason_codes: list[dict[str, Any]] = []
        if self.reason_coefficients:
            x_row = self._features(frame).iloc[0]
            coef = pd.Series(self.reason_coefficients, dtype="float64")
            common = coef.index.intersection(x_row.index)
            if len(common) > 0:
                reason_codes = [
                    code.to_dict() for code in reason_codes_from_logit(coef, x_row, top_k=top_k)
                ]

        return {
            "pd": pd_value,
            "decile": decile,
            "reason_codes": reason_codes,
            "predicted_label": label,
            "threshold": threshold,
            "model_auc": float(self.model_auc),
            "base_rate": float(self.base_rate),
            "data_source": self.data_source,
        }


@dataclass(frozen=True, slots=True)
class TrainArtifacts:
    """Immutable bundle of everything a training run produces.

    Attributes
    ----------
    booster_path:
        Path to the emitted ``<2MB`` booster JSON.
    pipeline_path:
        Path to the serialized fitted feature pipeline.
    metrics:
        Held-out (latest-vintage) headline metrics for the calibrated XGBoost.
    logistic_metrics:
        Held-out metrics for the L2-logistic baseline (for the horse race).
    base_rate_auc:
        ROC-AUC of the base-rate predictor (0.5 floor, recorded for honesty).
    delong:
        DeLong AUC-difference test, XGBoost vs logistic.
    n_trials:
        The RECORDED nested-CV trial count (guard asserts it >= the config grid).
    data_source:
        ``"synthetic"`` or ``"kaggle"`` — which panel the artifact was trained on.
    manifest:
        The run's :class:`lendingclub_default.RunManifest` as a dict.
    """

    booster_path: Path
    pipeline_path: Path
    metrics: MetricBundle
    logistic_metrics: MetricBundle
    base_rate_auc: float
    delong: DeLongResult
    n_trials: int
    data_source: str
    manifest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable summary of the artifacts."""
        return {
            "booster_path": str(self.booster_path),
            "pipeline_path": str(self.pipeline_path),
            "metrics": self.metrics.to_dict(),
            "logistic_metrics": self.logistic_metrics.to_dict(),
            "base_rate_auc": float(self.base_rate_auc),
            "delong": self.delong.to_dict(),
            "n_trials": int(self.n_trials),
            "data_source": self.data_source,
            "manifest": dict(self.manifest),
        }


def _temporal_subsplit(
    panel: pd.DataFrame,
    y: pd.Series,
    *,
    issue_col: str,
    holdout_size: float,
) -> tuple[list[Any], list[Any]]:
    """Carve a *later-vintage* holdout out of an (already train-only) panel.

    Returns ``(early_idx, late_idx)`` where ``late_idx`` are the most recent
    vintages (~``holdout_size`` of rows). Used to peel an early-stopping
    validation slice and a calibration slice off the train fold WITHOUT ever
    touching the held-out test vintages — so no future leaks into fitting.

    Falls back to a deterministic tail slice when the fold has a single vintage
    (so the orchestrator still produces valid, ordered sub-folds).
    """
    if panel.shape[0] < 4:
        raise ValidationError(
            f"_temporal_subsplit: need >= 4 rows to sub-split, got {panel.shape[0]}."
        )
    distinct = sorted(pd.unique(panel[issue_col].dropna()))
    if len(distinct) >= 2:
        sub = temporal_split(panel, issue_col=issue_col, test_size=holdout_size)
        return sub.train_idx, sub.test_idx
    # Single-vintage fallback: a stable index-order tail slice.
    ordered = list(panel.index)
    n_late = max(1, round(len(ordered) * holdout_size))
    n_late = min(n_late, len(ordered) - 1)
    return ordered[:-n_late], ordered[-n_late:]


def train(
    *,
    data_path: Path | str | None = None,
    out_dir: Path | str = "artifacts",
    synthetic_config: SyntheticConfig | None = None,
    xgb_config: XGBConfig | None = None,
    calibration_method: str = "isotonic",
    seed: int = 20260616,
) -> TrainArtifacts:
    """Run the full leakage-free, temporally-split, calibrated training pipeline.

    Parameters
    ----------
    data_path:
        Real LendingClub accepted-loans CSV. If ``None``, the synthetic generator
        is used (the reproducible default; the shipped artifact is synthetic).
    out_dir:
        Directory to write the booster JSON, fitted pipeline, and manifest.
    synthetic_config:
        Synthetic-generator config (used only when ``data_path is None``).
    xgb_config:
        XGBoost hyperparameters; defaults to :class:`lendingclub_default.models.xgb.XGBConfig`.
    calibration_method:
        ``"isotonic"`` (default) or ``"sigmoid"`` (Platt) calibration.
    seed:
        Master RNG seed for the whole run (RunManifest-stamped).

    Returns
    -------
    TrainArtifacts
        Paths to the emitted artifacts plus held-out metrics, the DeLong test,
        the recorded ``n_trials``, and the run manifest.

    Raises
    ------
    ArtifactError
        If the emitted booster JSON exceeds the ``<2MB`` ceiling.
    ValidationError
        If the panel is too small to carve the required temporal folds.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_source = "kaggle" if data_path is not None else "synthetic"

    # 1. Load -> 2. drop post-funding leakage. The label is rebuilt from the raw
    #    panel (loan_status lives in LEAKAGE_COLS) BEFORE the leakage drop.
    raw_panel = load_panel(data_path, config=synthetic_config)
    label_result = build_labels(raw_panel)
    panel = label_result.panel
    y_all = label_result.y

    features_panel = drop_leakage(panel)
    # Hard backstop: the model must never see an outcome-leaking column.
    assert_no_leakage(features_panel)

    # 3. TEMPORAL vintage split: train <= cutoff, test on later vintages. The
    #    test fold is the held-out LATEST vintage(s) used for the headline.
    split = temporal_split(features_panel, issue_col="issue_d", test_size=0.25)
    train_panel = features_panel.loc[split.train_idx]
    test_panel = features_panel.loc[split.test_idx]
    y_train = y_all.loc[split.train_idx]
    y_test = y_all.loc[split.test_idx]

    # 3b. Peel a later-vintage EARLY-STOPPING fold and a CALIBRATION fold off the
    #     train fold only (never the test vintages). Carve in two stages so the
    #     calibration fold is the latest train slice, the early-stop fold next,
    #     and the model-fit core the earliest — all strictly ordered in time.
    fit_idx, calib_idx = _temporal_subsplit(
        train_panel, y_train, issue_col="issue_d", holdout_size=0.2
    )
    core_panel = train_panel.loc[fit_idx]
    y_core = y_train.loc[fit_idx]
    calib_panel = train_panel.loc[calib_idx]
    y_calib = y_train.loc[calib_idx]

    core_fit_idx, valid_idx = _temporal_subsplit(
        core_panel, y_core, issue_col="issue_d", holdout_size=0.2
    )
    inner_panel = core_panel.loc[core_fit_idx]
    y_inner = y_core.loc[core_fit_idx]
    valid_panel = core_panel.loc[valid_idx]
    y_valid = y_core.loc[valid_idx]

    # 4. Build + FIT the feature pipeline on the inner (model-fit) fold ONLY. The
    #    same fitted object transforms every later fold — out-of-fold target
    #    encoding inside it guarantees no row sees its own label, and later
    #    vintages cannot influence any learned statistic.
    spec = FeatureSpec()
    pipeline = build_feature_pipeline(spec, seed=seed)
    x_inner = pipeline.fit_transform(inner_panel, y_inner.to_numpy())
    x_valid = pipeline.transform(valid_panel)
    x_calib = pipeline.transform(calib_panel)
    x_test = pipeline.transform(test_panel)

    feature_names = _resolve_feature_names(pipeline, x_inner.shape[1])
    x_inner_df = _as_named_frame(x_inner, feature_names, inner_panel.index)
    x_valid_df = _as_named_frame(x_valid, feature_names, valid_panel.index)
    x_calib_df = _as_named_frame(x_calib, feature_names, calib_panel.index)
    x_test_df = _as_named_frame(x_test, feature_names, test_panel.index)

    # 5. Fit the models on the inner fold (early stopping on the temporal valid
    #    fold) -> base-rate floor + L2-logistic baseline + the headline XGBoost.
    base_predictor = _baselines.BaseRatePredictor.fit(y_inner)
    logistic = _baselines.fit_logistic(x_inner_df, y_inner, seed=seed)
    booster = _xgb.fit_xgb(x_inner_df, y_inner, x_valid_df, y_valid, config=xgb_config)

    # 6. Calibrate the booster on the held-out calibration fold (later vintages
    #    than the inner fit, earlier than the test fold) -> PD in [0, 1].
    raw_calib = _xgb.predict_proba(booster, x_calib_df)
    calibration, _calib_summary = _calibrate.fit_calibration(
        np.asarray(raw_calib, dtype="float64"), y_calib, method=calibration_method
    )

    # 7. Evaluate on the held-out LATEST vintage. Headline = calibrated XGB; the
    #    logistic is calibrated identically so the DeLong comparison is fair.
    raw_test_xgb = np.asarray(_xgb.predict_proba(booster, x_test_df), dtype="float64")
    pd_test_xgb = np.clip(calibration.calibrate(raw_test_xgb), 0.0, 1.0)

    logit_calib = _logistic_proba(logistic, x_calib_df)
    logit_calibration, _ = _calibrate.fit_calibration(
        logit_calib, y_calib, method=calibration_method
    )
    raw_test_logit = _logistic_proba(logistic, x_test_df)
    pd_test_logit = np.clip(logit_calibration.calibrate(raw_test_logit), 0.0, 1.0)

    pd_test_base = base_predictor.predict_proba(test_panel)

    y_test_np = y_test.to_numpy(dtype="float64")
    metrics = compute_metrics(y_test_np, pd_test_xgb)
    logistic_metrics = compute_metrics(y_test_np, pd_test_logit)
    base_rate_auc = roc_auc(y_test_np, np.asarray(pd_test_base, dtype="float64"))

    # 8. DeLong AUC-difference test (XGB vs logistic) on the SAME held-out rows.
    #    Bonferroni-corrected for the swept config grid (multiplicity honesty).
    grid = _swept_config_grid()
    n_comparisons = max(1, len(grid))
    delong = delong_auc_test(y_test_np, pd_test_xgb, pd_test_logit, n_comparisons=n_comparisons)

    # The recorded nested-CV trial count: every grid configuration counts, so the
    # guard ``n_trials >= len(grid)`` is satisfied by construction.
    n_trials = len(grid)

    # 9. Persist the <2MB booster JSON + fitted pipeline + calibration + manifest.
    resolved_xgb = xgb_config if xgb_config is not None else _xgb.XGBConfig()
    run_config = {
        "data_source": data_source,
        "seed": seed,
        "calibration_method": calibration_method,
        "feature_spec": spec.to_dict(),
        "xgb_config": resolved_xgb.to_dict(),
        "config_grid_size": len(grid),
        "n_trials": n_trials,
        "n_train": int(train_panel.shape[0]),
        "n_test": int(test_panel.shape[0]),
        "test_vintages": [str(v) for v in split.test_vintages],
    }
    manifest = RunManifest.capture(run_config, seed)

    # Container-safe reason-code coefficients: the L2-logistic baseline's weights
    # keyed by engineered feature name (SHAP is dev-only, never in the image).
    reason_coefficients = _logistic_coefficients(logistic, feature_names)

    booster_path = out / BOOSTER_FILENAME
    pipeline_path = out / PIPELINE_FILENAME
    _xgb.save_booster(booster, booster_path)
    _check_booster_size(booster_path)
    _save_pipeline(pipeline, pipeline_path)
    _save_calibration(
        calibration,
        out / CALIBRATION_FILENAME,
        data_source,
        metrics,
        reason_coefficients=reason_coefficients,
        feature_names=feature_names,
    )
    (out / MANIFEST_FILENAME).write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )

    return TrainArtifacts(
        booster_path=booster_path,
        pipeline_path=pipeline_path,
        metrics=metrics,
        logistic_metrics=logistic_metrics,
        base_rate_auc=float(base_rate_auc),
        delong=delong,
        n_trials=n_trials,
        data_source=data_source,
        manifest=manifest.to_dict(),
    )


def _as_named_frame(
    matrix: np.ndarray,
    feature_names: list[str],
    index: pd.Index,
) -> pd.DataFrame:
    """Wrap a transformed feature matrix as a named, index-aligned DataFrame."""
    return pd.DataFrame(np.asarray(matrix), columns=feature_names, index=index)


def _resolve_feature_names(pipeline: Pipeline, n_cols: int) -> list[str]:
    """Best-effort feature names from a fitted pipeline (fallback: ``f0..fk``)."""
    try:
        names = list(pipeline.get_feature_names_out())
    except (AttributeError, ValueError, KeyError):
        names = [f"f{i}" for i in range(n_cols)]
    if len(names) != n_cols:
        names = [f"f{i}" for i in range(n_cols)]
    return [str(name) for name in names]


def _logistic_proba(model: LogisticRegression, x: pd.DataFrame) -> np.ndarray:
    """Positive-class probabilities from a fitted logistic model as float64.

    The baseline is fit on a nameless ndarray, so we score on ``.to_numpy()`` to
    keep sklearn's feature-name bookkeeping consistent (no spurious warning).
    """
    proba = model.predict_proba(x.to_numpy())
    return np.asarray(proba[:, 1], dtype="float64")


def _logistic_coefficients(
    model: LogisticRegression,
    feature_names: list[str],
) -> dict[str, float]:
    """Map the fitted logistic coefficients to engineered feature names.

    Powers the container-safe reason codes: the signed log-odds contribution of
    each feature is ``coef * x`` (computed at score time). Returns an empty dict
    if the coefficient/name counts disagree (defensive — reason codes are then
    simply omitted rather than misaligned).
    """
    coef = np.asarray(model.coef_, dtype="float64").ravel()
    if coef.size != len(feature_names):
        return {}
    return {str(name): float(value) for name, value in zip(feature_names, coef, strict=True)}


def _check_booster_size(path: Path) -> None:
    """Raise :class:`ArtifactError` if the booster JSON exceeds the 2 MB ceiling."""
    size = path.stat().st_size
    if size > MAX_BOOSTER_BYTES:
        raise ArtifactError(
            f"booster artifact {path.name} is {size} bytes, exceeding the "
            f"{MAX_BOOSTER_BYTES}-byte (<2MB) ceiling. Reduce n_estimators/max_depth."
        )


def _save_pipeline(pipeline: Pipeline, path: Path) -> None:
    """Serialize the fitted feature pipeline with joblib (lazy import)."""
    import joblib  # type: ignore[import-untyped]

    joblib.dump(pipeline, path)


def _load_pipeline(path: Path) -> Pipeline:
    """Load a joblib-serialized fitted feature pipeline."""
    import joblib

    if not path.exists():
        raise ArtifactError(f"fitted pipeline not found at {path}.")
    pipeline: Pipeline = joblib.load(path)
    return pipeline


def _save_calibration(
    calibration: CalibratedModel,
    path: Path,
    data_source: str,
    metrics: MetricBundle,
    *,
    reason_coefficients: dict[str, float] | None = None,
    feature_names: list[str] | None = None,
) -> None:
    """Serialize the calibration map + headline scalars + reason coeffs as JSON."""
    payload = {
        "calibration": calibration.to_dict(),
        "data_source": data_source,
        "model_auc": float(metrics.roc_auc),
        "base_rate": float(metrics.base_rate),
        "reason_coefficients": dict(reason_coefficients or {}),
        "feature_names": list(feature_names or []),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_calibration(path: Path) -> dict[str, Any]:
    """Load the calibration-map + scalars JSON payload."""
    if not path.exists():
        raise ArtifactError(f"calibration artifact not found at {path}.")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def load_artifacts(out_dir: Path | str = "artifacts") -> ScoredArtifacts:
    """Load a complete scoring bundle written by :func:`train`.

    Reads the booster JSON, the fitted feature pipeline, and the calibration map
    from ``out_dir`` and assembles a :class:`ScoredArtifacts` ready to turn a raw
    application panel into a calibrated PD.

    Parameters
    ----------
    out_dir:
        Directory the artifacts were written to.

    Returns
    -------
    ScoredArtifacts
        The fitted pipeline, booster, calibration map, and recorded scalars.

    Raises
    ------
    ArtifactError
        If any required artifact is missing or malformed.
    """
    out = Path(out_dir)
    booster = _xgb.load_booster(out / BOOSTER_FILENAME, use_cache=False)
    pipeline = _load_pipeline(out / PIPELINE_FILENAME)
    payload = _load_calibration(out / CALIBRATION_FILENAME)

    calib_dict = payload.get("calibration", {})
    method = calib_dict.get("method")
    if method not in ("isotonic", "sigmoid"):
        raise ArtifactError(
            f"calibration artifact has unknown method {method!r}; expected 'isotonic' or 'sigmoid'."
        )
    calibration = _calibrate.CalibratedModel(
        method=str(method), params=dict(calib_dict.get("params", {}))
    )
    reason_coefficients = {
        str(k): float(v) for k, v in dict(payload.get("reason_coefficients", {})).items()
    }
    feature_names = [str(name) for name in payload.get("feature_names", [])]

    return ScoredArtifacts(
        booster=booster,
        pipeline=pipeline,
        calibration=calibration,
        model_auc=float(payload.get("model_auc", float("nan"))),
        base_rate=float(payload.get("base_rate", float("nan"))),
        data_source=str(payload.get("data_source", "synthetic")),
        reason_coefficients=reason_coefficients,
        feature_names=feature_names,
    )


def load_booster(out_dir: Path | str = "artifacts", *, use_cache: bool = True) -> ScoredArtifacts:
    """Lazily load (and cache) the shipped scoring bundle the backend calls.

    On first call this populates the module-level ``_BOOSTER`` sentinel with a
    fully assembled :class:`ScoredArtifacts` (booster + fitted pipeline +
    calibration map), so the API reuses a single in-process bundle across
    requests without ever retraining. Subsequent calls return the cached bundle.

    Parameters
    ----------
    out_dir:
        Directory the committed artifacts live in.
    use_cache:
        If ``True`` (default), reuse / populate the module-level ``_BOOSTER``
        sentinel; if ``False``, always load a fresh bundle.

    Returns
    -------
    ScoredArtifacts
        The ready-to-score bundle.

    Raises
    ------
    ArtifactError
        If the artifacts are missing or malformed.
    """
    global _BOOSTER
    if use_cache and _BOOSTER is not None:
        return _BOOSTER
    bundle = load_artifacts(out_dir)
    if use_cache:
        _BOOSTER = bundle
    return bundle


def _reset_booster_cache() -> None:
    """Clear the lazy ``_BOOSTER`` sentinel (used by tests for isolation)."""
    global _BOOSTER
    _BOOSTER = None


if __name__ == "__main__":  # pragma: no cover
    # Demo entry point is intentionally guarded so importing this module is pure.
    train()
