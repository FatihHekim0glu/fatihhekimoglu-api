"""Anchored/expanding walk-forward engine with strict fit-on-train-only.

This is the leakage-control heart of the project and the explicit fix for the
Stock-Price-Forecast anti-pattern (fitting a scaler/model on the FULL series and
then "evaluating" out-of-sample). Here EVERY estimator — the RV scaler, the GARCH
parameters, the HAR-RV OLS, and the XGBoost booster — is fit INSIDE each
walk-forward TRAIN fold and only then asked to forecast the disjoint TEST fold.

Guards enforced and property-tested:

- **Forward-only targets** with an explicit ``gap`` (the target window is strictly
  ``> t + gap``); a **purge** removes the boundary rows whose target window
  overlaps the test fold, and an **embargo** sized to the horizon ``h`` removes
  the rows just after the train fold.
- **Future-perturbation invariance**: perturbing returns strictly after the
  forecast origin must not change a fold's forecasts (the canonical leakage
  detector), asserted by a property test.

The engine collects, per test point, the realized RV and each model's forecast,
so the evaluation layer can compute QLIKE/MSE and run DM/SPA. Importing this
module has no side effects (``arch``/``xgboost`` are imported lazily by the
fitters this engine calls).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from volforecast._exceptions import InsufficientDataError, ValidationError
from volforecast.baselines import (
    ewma_vol_forecast,
    fit_har_rv,
    random_walk_vol_forecast,
)
from volforecast.data import log_returns
from volforecast.features.har import build_har_features, har_components
from volforecast.realized.estimators import forward_rv_target, realized_volatility

#: The default model set fitted on the serve path (LSTM excluded — research only).
DEFAULT_MODELS: tuple[str, ...] = ("garch", "egarch", "har_rv", "ewma", "xgboost", "rw")

#: The GARCH-family labels routed to ``arch`` (mapped to its ``kind`` selector).
_GARCH_KINDS: dict[str, str] = {"garch": "garch", "egarch": "egarch", "gjr": "gjr"}


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    """Immutable configuration of a walk-forward run.

    Attributes
    ----------
    horizon:
        The forecast horizon ``h`` in trading days (one of 1, 5, 22).
    train_window:
        The minimum train-fold length in trading days (the warm-up).
    gap:
        The gap between the feature timestamp and the target window (``>= 0``).
    step:
        The number of test points advanced between refits (``>= 1``).
    anchored:
        If ``True``, the train fold expands (anchored at the start); else it
        rolls with fixed ``train_window`` length.
    models:
        The model labels to fit each fold (subset of :data:`DEFAULT_MODELS`).
    seed:
        Master seed for any stochastic fitter (XGBoost determinism).
    """

    horizon: int
    train_window: int = 504
    gap: int = 1
    step: int = 1
    anchored: bool = True
    models: tuple[str, ...] = DEFAULT_MODELS
    seed: int = 7

    def __post_init__(self) -> None:
        """Validate the configuration scalars.

        Raises
        ------
        ValidationError
            If ``horizon < 1``, ``train_window < 2``, ``gap < 0``, ``step < 1``,
            or ``models`` is empty / contains an unknown label.
        """
        for label, value in (
            ("horizon", self.horizon),
            ("train_window", self.train_window),
            ("gap", self.gap),
            ("step", self.step),
            ("seed", self.seed),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValidationError(f"{label} must be an int, got {type(value).__name__}.")
        if self.horizon < 1:
            raise ValidationError(f"horizon must be >= 1, got {self.horizon}.")
        if self.train_window < 2:
            raise ValidationError(f"train_window must be >= 2, got {self.train_window}.")
        if self.gap < 0:
            raise ValidationError(f"gap must be >= 0, got {self.gap}.")
        if self.step < 1:
            raise ValidationError(f"step must be >= 1, got {self.step}.")
        if self.seed < 0:
            raise ValidationError(f"seed must be >= 0, got {self.seed}.")
        if not self.models:
            raise ValidationError("models must be a non-empty tuple.")
        unknown = [m for m in self.models if m not in DEFAULT_MODELS]
        if unknown:
            raise ValidationError(
                f"unknown model label(s) {unknown}; expected a subset of {list(DEFAULT_MODELS)}."
            )

    @property
    def embargo(self) -> int:
        """The embargo length (sized to the horizon): ``gap + horizon``.

        This is the number of rows skipped after the train fold before the test
        origin. It is sized to the horizon so that the last train row's forward
        target window ``(t + gap, t + gap + horizon]`` cannot overlap the test
        origin's features — the leakage purge.
        """
        return self.gap + self.horizon

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this configuration."""
        return {
            "horizon": int(self.horizon),
            "train_window": int(self.train_window),
            "gap": int(self.gap),
            "step": int(self.step),
            "anchored": bool(self.anchored),
            "models": list(self.models),
            "seed": int(self.seed),
            "embargo": int(self.embargo),
        }


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    """Immutable result of a walk-forward run: aligned actuals and forecasts.

    Attributes
    ----------
    realized_vol:
        The realized forward volatility at each evaluated test origin (the truth).
    forecasts:
        A ``(n_test, n_models)`` frame of per-model forward-vol forecasts aligned
        to ``realized_vol.index``.
    config:
        The :class:`WalkForwardConfig` used.
    n_folds:
        The number of train/test folds evaluated.
    """

    realized_vol: pd.Series
    forecasts: pd.DataFrame
    config: WalkForwardConfig
    n_folds: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result.

        Series/frames are rendered with ISO-formatted date keys and NaN scrubbed
        to ``None`` so the result crosses the API boundary cleanly.
        """

        def _key(idx: Any) -> str:
            return idx.isoformat() if hasattr(idx, "isoformat") else str(idx)

        def _val(value: Any) -> float | None:
            out = float(value)
            return out if np.isfinite(out) else None

        realized = {_key(idx): _val(v) for idx, v in self.realized_vol.items()}
        forecasts = {
            str(col): {_key(idx): _val(v) for idx, v in self.forecasts[col].items()}
            for col in self.forecasts.columns
        }
        return {
            "realized_vol": realized,
            "forecasts": forecasts,
            "config": self.config.to_dict(),
            "n_folds": int(self.n_folds),
            "meta": {str(k): v for k, v in self.meta.items()},
        }


def _train_test_slices(
    n_obs: int,
    config: WalkForwardConfig,
) -> list[tuple[int, int, int]]:
    """Enumerate ``(train_start, train_end, test_origin)`` index triples.

    Each triple defines a fold: the train fold is ``[train_start, train_end)``,
    and the test origin is ``train_end + embargo`` (the first row whose features
    are observable and whose forward target is strictly disjoint from training).
    ``train_start`` is ``0`` when ``anchored`` else ``train_end - train_window``.

    Parameters
    ----------
    n_obs:
        The total number of observations available.
    config:
        The walk-forward configuration.

    Returns
    -------
    list[tuple[int, int, int]]
        The ordered fold triples (possibly empty if the series is too short).
    """
    embargo = config.embargo
    horizon = config.horizon
    gap = config.gap
    slices: list[tuple[int, int, int]] = []

    # The first refit happens once the warm-up train window has accrued. The test
    # origin sits ``embargo`` rows after the train end (the purge gap), and its
    # forward target needs ``gap + horizon`` more rows to be complete, so the last
    # admissible test origin is ``n_obs - 1 - (gap + horizon)``.
    last_origin = n_obs - 1 - (gap + horizon)

    train_end = config.train_window
    while True:
        test_origin = train_end + embargo
        if test_origin > last_origin:
            break
        train_start = 0 if config.anchored else max(0, train_end - config.train_window)
        slices.append((train_start, train_end, test_origin))
        # Advance the refit by ``step`` test points; the next train fold ends one
        # row before the next test origin's embargo, i.e. it grows by ``step``.
        train_end += config.step

    return slices


def run_walk_forward(
    ohlc: pd.DataFrame,
    *,
    config: WalkForwardConfig,
    rv_estimator: str = "garman_klass",
    exog: pd.DataFrame | None = None,
) -> WalkForwardResult:
    """Run the leakage-guarded walk-forward forecast over an OHLC series.

    For each fold the engine: (1) computes the daily RV proxy and the forward RV
    target with the configured ``gap``; (2) fits EVERY model in ``config.models``
    on the TRAIN fold ONLY (GARCH via ``arch``, HAR-RV by OLS, XGBoost with a
    fixed seed, EWMA/RW closed-form); (3) forecasts the ``horizon``-ahead vol at
    the test origin; (4) records the realized forward vol and each forecast. No
    estimator ever touches data at or after its forecast origin.

    Parameters
    ----------
    ohlc:
        A wide OHLC frame (``open/high/low/close``, case-insensitive) indexed by
        date.
    config:
        The :class:`WalkForwardConfig`.
    rv_estimator:
        The RV proxy used for features AND target, one of ``"close_to_close"``,
        ``"parkinson"``, ``"garman_klass"``.
    exog:
        Optional exogenous, ALREADY-LAGGED features (e.g. a VIX level) aligned to
        ``ohlc.index``.

    Returns
    -------
    WalkForwardResult
        The aligned realized-vol series and per-model forecast frame.

    Raises
    ------
    ValidationError
        If ``ohlc`` is malformed or ``rv_estimator`` is unknown.
    InsufficientDataError
        If the series is too short for even one train/test fold under ``config``.
    """
    if not isinstance(ohlc, pd.DataFrame):
        raise ValidationError("ohlc must be a pandas.DataFrame.")
    if rv_estimator not in ("close_to_close", "parkinson", "garman_klass"):
        raise ValidationError(
            f"unknown rv_estimator {rv_estimator!r}; expected one of "
            "('close_to_close', 'parkinson', 'garman_klass')."
        )

    frame = ohlc.copy()
    frame = frame.rename(columns={c: str(c).lower() for c in frame.columns})
    if "close" not in frame.columns:
        raise ValidationError("ohlc must contain a 'close' column.")
    frame = frame.sort_index()

    # --- Series computed ONCE, but all no-lookahead -------------------------- #
    # ``rv`` at row t uses only OHLC at or before t (trailing range estimator);
    # ``returns`` at row t uses only closes at or before t. The forward ``target``
    # at row t deliberately uses FUTURE RV — but it is the LABEL (the truth read
    # at the test origin), never an input to any fitter or feature. Computing them
    # once is therefore leakage-safe: per fold we only ever SLICE these.
    rv = realized_volatility(frame, estimator=rv_estimator, window=1)
    returns = log_returns(frame["close"])
    target = forward_rv_target(rv, horizon=config.horizon, gap=config.gap)

    n_obs = int(frame.shape[0])
    slices = _train_test_slices(n_obs, config)
    if not slices:
        raise InsufficientDataError(
            f"series of length {n_obs} is too short for even one walk-forward fold "
            f"under config {config.to_dict()}."
        )

    index = frame.index
    records: list[tuple[Any, float, dict[str, float]]] = []
    n_skipped = 0
    for train_start, train_end, test_origin in slices:
        origin_label = index[test_origin]
        truth = target.iloc[test_origin]
        if not np.isfinite(truth):
            # The test origin's forward window is incomplete (boundary) — skip it
            # rather than fabricate a target.
            n_skipped += 1
            continue

        # The forecast origin is ``train_end`` (exclusive end of the train slice):
        # every fitter sees ONLY rows ``[train_start, train_end)`` and forecasts the
        # disjoint, embargoed test origin. ``iloc`` slicing is the purge.
        fold_forecasts = _forecast_fold(
            train_start=train_start,
            train_end=train_end,
            rv=rv,
            returns=returns,
            exog=exog,
            config=config,
        )
        records.append((origin_label, float(truth), fold_forecasts))

    if not records:
        raise InsufficientDataError(
            "no walk-forward fold produced a complete forward target; the series is "
            "too short for the requested horizon/gap."
        )

    origins = [r[0] for r in records]
    realized_vol = pd.Series(
        [r[1] for r in records], index=pd.Index(origins), name="realized_vol", dtype="float64"
    )
    forecasts = pd.DataFrame([r[2] for r in records], index=pd.Index(origins)).reindex(
        columns=list(config.models)
    )

    meta: dict[str, Any] = {
        "rv_estimator": rv_estimator,
        "n_obs": n_obs,
        "n_skipped": int(n_skipped),
    }
    return WalkForwardResult(
        realized_vol=realized_vol,
        forecasts=forecasts,
        config=config,
        n_folds=len(records),
        meta=meta,
    )


def _forecast_fold(
    *,
    train_start: int,
    train_end: int,
    rv: pd.Series,
    returns: pd.Series,
    exog: pd.DataFrame | None,
    config: WalkForwardConfig,
) -> dict[str, float]:
    """Fit every requested model on ONE train fold and forecast the test origin.

    Every estimator here is fit on the train slice ``[train_start, train_end)``
    only — the GARCH parameters, the HAR-RV OLS, and the XGBoost booster are all
    re-estimated from scratch on this fold and never touch any row at or after
    ``train_end`` (the forecast origin). That is the explicit fix for the
    full-sample-fit anti-pattern. The closed-form RW/EWMA forecasts read the value
    at ``train_end - 1`` (the last observable row), which is itself a function of
    past data only.
    """
    horizon = config.horizon
    rv_train = rv.iloc[train_start:train_end]
    ret_train = returns.iloc[:train_end].dropna()
    # Only the returns whose timestamps fall inside the train slice's date span
    # are admissible (no future return ever enters a fit).
    ret_train = ret_train.loc[ret_train.index <= rv.index[train_end - 1]]
    ret_train = ret_train.iloc[-(train_end - train_start) :] if train_start > 0 else ret_train

    out: dict[str, float] = {}
    for model in config.models:
        try:
            if model in _GARCH_KINDS:
                out[model] = _garch_forecast(ret_train, kind=_GARCH_KINDS[model], horizon=horizon)
            elif model == "har_rv":
                out[model] = _har_forecast(rv_train, config=config)
            elif model == "xgboost":
                out[model] = _xgb_forecast(rv_train, exog=exog, config=config)
            elif model == "ewma":
                out[model] = _ewma_forecast(ret_train, horizon=horizon)
            elif model == "rw":
                out[model] = _rw_forecast(rv_train, horizon=horizon)
            else:  # pragma: no cover - guarded by config validation
                raise ValidationError(f"unknown model {model!r}.")
        except (ValidationError, InsufficientDataError):
            # A fold too short for one model (e.g. GARCH on a tiny warm-up) yields
            # a NaN forecast for that model on that fold rather than aborting the
            # whole run; the evaluation layer drops NaN-aligned pairs.
            out[model] = float("nan")
    return out


def _garch_forecast(returns: pd.Series, *, kind: str, horizon: int) -> float:
    """Fit a GARCH-family model on the train returns and forecast h-day RV."""
    from volforecast.garch.models import fit_garch, forecast_garch_vol

    fit = fit_garch(returns, kind=kind, dist="normal")
    return forecast_garch_vol(fit, returns, horizon=horizon)


def _har_forecast(rv_train: pd.Series, *, config: WalkForwardConfig) -> float:
    """Fit HAR-RV by OLS on the train fold and forecast the next-origin RV.

    The HAR components and forward target are recomputed on the TRAIN slice only,
    the OLS coefficients are fit in-sample, and the forecast reads the last
    observable (lagged) HAR component row — which carries information up to the
    forecast origin and no further.
    """
    components = har_components(rv_train)
    train_target = forward_rv_target(rv_train, horizon=config.horizon, gap=config.gap)
    bundle = build_har_features(rv_train, train_target)
    model = fit_har_rv(bundle.features[["rv_daily", "rv_weekly", "rv_monthly"]], bundle.target)

    # The last complete (non-NaN) lagged HAR row is observable at the origin.
    last_row = components.dropna(axis=0, how="any").iloc[[-1]]
    return float(model.predict(last_row).iloc[-1])


def _xgb_forecast(
    rv_train: pd.Series,
    *,
    exog: pd.DataFrame | None,
    config: WalkForwardConfig,
) -> float:
    """Fit XGBoost on HAR features (train fold only) and forecast the origin RV."""
    from volforecast.ml.xgb import fit_xgb

    train_target = forward_rv_target(rv_train, horizon=config.horizon, gap=config.gap)
    exog_train = exog.reindex(rv_train.index) if exog is not None else None
    bundle = build_har_features(rv_train, train_target, exog=exog_train)
    forecaster = fit_xgb(bundle.features, bundle.target, seed=config.seed)

    components = har_components(rv_train)
    design = components
    if exog_train is not None:
        design = components.join(exog_train.astype("float64"), how="left")
    last_row = design[list(bundle.feature_names)].dropna(axis=0, how="any").iloc[[-1]]
    return float(forecaster.predict(last_row).iloc[-1])


def _ewma_forecast(returns: pd.Series, *, horizon: int) -> float:
    """RiskMetrics EWMA forecast read at the last observable train row."""
    forecast = ewma_vol_forecast(returns, horizon=horizon)
    return float(forecast.iloc[-1])


def _rw_forecast(rv_train: pd.Series, *, horizon: int) -> float:
    """Random-walk-in-RV forecast: the last observable RV carried forward."""
    forecast = random_walk_vol_forecast(rv_train, horizon=horizon).dropna()
    if forecast.empty:
        raise InsufficientDataError("random-walk forecast needs at least two RV observations.")
    return float(forecast.iloc[-1])
