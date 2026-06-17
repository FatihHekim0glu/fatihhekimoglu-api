"""Public anomaly-scan entrypoint — the single function the backend calls.

:func:`run_anomaly_scan` is the library's one-call orchestration surface: given a
price (or return) series it engineers causal features, runs a strictly causal
WALK-FORWARD refit of the two independent detectors (Isolation Forest + the
PCA-reconstruction autoencoder), scores each disjoint out-of-sample fold, and
assembles the honest DESCRIPTIVE summary the API marshals.

WALK-FORWARD DISCIPLINE (the top project risk — full-sample leakage — is fenced
off here): on every fold the StandardScaler, both detectors, and ALL thresholds
are refitted on the TRAIN slice of that fold ONLY, then score the disjoint OOS
fold. No detector ever sees a bar at or after the day it scores; the per-day
features are themselves ``.shift(1)``-causal upstream. Concatenating the OOS
folds yields a single out-of-sample score/flag series with zero look-ahead.

The headline stays honest: the two detectors agree on a small core of stress
days, but their day-level agreement is MODEST and precision against the
transparent ``|z-return| > 3`` proxy is LOW. Flags are diagnostic, not tradable;
there is no ground-truth anomaly label, so no alpha is claimed.

Importing this module has no side effects (heavy compute is imported lazily and
the detectors import scikit-learn lazily inside their own methods).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from anomaly_detector._exceptions import InsufficientDataError, ValidationError
from anomaly_detector.detectors.result import _safe_float

if TYPE_CHECKING:
    from anomaly_detector._typing import PricesLike, ReturnsLike
    from anomaly_detector.detectors.result import AnomalyResult
    from anomaly_detector.evaluation.agreement import AgreementResult
    from anomaly_detector.plots import FigureDict

#: Which detector the summary's ``detector`` field and primary flag count refer
#: to. ``"both"`` reports the union of the two detectors' flags.
DetectorChoice = Literal["iforest", "autoencoder", "both"]

#: Default rolling feature window (trading days).
DEFAULT_WINDOW: int = 21

#: Default expected anomalous fraction (the flag-threshold quantile parameter).
DEFAULT_CONTAMINATION: float = 0.02

#: Default fraction of each fold's history used as the (initial) train slice; the
#: anchored walk-forward expands the train slice fold by fold.
_INITIAL_TRAIN_FRAC: float = 0.5

#: Default number of out-of-sample walk-forward folds.
DEFAULT_N_FOLDS: int = 4

#: Minimum OOS feature rows required to attempt a scan (so a fold is non-trivial).
_MIN_OOS_ROWS: int = 8


@dataclass(frozen=True, slots=True)
class ScanResult:
    """Immutable result of a full causal walk-forward anomaly scan.

    Attributes
    ----------
    result_iforest:
        The concatenated OOS :class:`AnomalyResult` of the Isolation Forest.
    result_autoencoder:
        The concatenated OOS :class:`AnomalyResult` of the PCA autoencoder.
    agreement:
        The descriptive :class:`AgreementResult` over the two OOS flag sets.
    oos_returns:
        The per-day OOS return series the agreement/proxy label is measured on.
    oos_prices:
        The per-day OOS price series (for the price-with-markers figure).
    detector:
        The detector choice the summary's ``detector`` field reports.
    data_source:
        Where the input came from (``"polygon"`` | ``"synthetic"``); ``None``
        when the caller passed an in-memory series directly.
    meta:
        Free-form JSON-serializable provenance (window, contamination, folds...).
    """

    result_iforest: AnomalyResult
    result_autoencoder: AnomalyResult
    agreement: AgreementResult
    oos_returns: pd.Series
    oos_prices: pd.Series
    detector: DetectorChoice
    data_source: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def _primary_flags(self) -> pd.Series:
        """Return the flag series the summary's ``n_flags`` counts.

        ``"iforest"``/``"autoencoder"`` report that detector's own flags;
        ``"both"`` reports the UNION (a day is flagged when EITHER detector
        flags it), matching the tool's overall flag set.
        """
        flags_if = self.result_iforest.flags
        flags_ae = self.result_autoencoder.flags
        if self.detector == "iforest":
            return flags_if.fillna(False).astype(bool)
        if self.detector == "autoencoder":
            return flags_ae.fillna(False).astype(bool)
        union_index = flags_if.index.union(flags_ae.index)
        a = flags_if.reindex(union_index).fillna(False).astype(bool)
        b = flags_ae.reindex(union_index).fillna(False).astype(bool)
        return (a | b).rename("anomaly_flag")

    def summary(self) -> dict[str, Any]:
        """Return the honest DESCRIPTIVE summary block the API marshals.

        Every scalar passes through :func:`_safe_float` (NaN/Inf -> ``None``) so
        the block is strictly JSON-safe. The ``detector`` field echoes the
        requested choice; ``data_source`` is included when known.

        Returns
        -------
        dict[str, Any]
            ``{n_flags, jaccard, proxy_precision, proxy_recall,
            top_anomaly_dates, detector, data_source}``.
        """
        agree = self.agreement
        return {
            "n_flags": int(self._primary_flags().sum()),
            "jaccard": _safe_float(agree.jaccard),
            "proxy_precision": _safe_float(agree.proxy_precision),
            "proxy_recall": _safe_float(agree.proxy_recall),
            "top_anomaly_dates": [str(d) for d in agree.top_anomaly_dates],
            "detector": str(self.detector),
            "data_source": self.data_source,
        }

    def figures(self) -> dict[str, FigureDict]:
        """Assemble the two Plotly ``{data, layout}`` figures for the tool.

        - ``price_figure`` — the OOS price path with markers on the (union of)
          flagged anomalous days.
        - ``score_figure`` — the PRIMARY detector's OOS anomaly-score series with
          its train-derived threshold drawn as a horizontal line. For the
          ``"both"`` choice the Isolation Forest score series is shown (its
          threshold is in the same ``-score_samples`` units).

        Plotly is imported lazily inside the figure builders, so calling this
        requires the ``viz`` extra but importing the module does not.

        Returns
        -------
        dict[str, FigureDict]
            ``{"price_figure": ..., "score_figure": ...}``.
        """
        from anomaly_detector.plots import price_anomaly_figure, score_threshold_figure

        union_flags = self._primary_flags()
        primary = self.result_autoencoder if self.detector == "autoencoder" else self.result_iforest
        price_figure = price_anomaly_figure(self.oos_prices, union_flags)
        score_figure = score_threshold_figure(primary.scores, primary.threshold)
        return {"price_figure": price_figure, "score_figure": score_figure}


def _coerce_prices(prices: PricesLike | None, returns: ReturnsLike | None) -> pd.Series:
    """Resolve a single price series from a price or return input.

    Exactly one of ``prices``/``returns`` must be supplied. A return series is
    integrated into a synthetic price path (base 100) so the same causal feature
    engineer and the price-with-markers figure work unchanged.
    """
    from anomaly_detector._validation import ensure_series

    if (prices is None) == (returns is None):
        raise ValidationError("run_anomaly_scan: pass exactly one of `prices` or `returns`.")
    if prices is not None:
        if isinstance(prices, pd.DataFrame):
            if prices.shape[1] != 1:
                raise ValidationError(
                    "run_anomaly_scan: a price DataFrame must have exactly one "
                    f"column, got {prices.shape[1]}."
                )
            prices = prices.iloc[:, 0]
        return ensure_series(prices, name="prices").rename("price")

    ret = ensure_series(returns, name="returns", allow_nan=True).fillna(0.0)
    price_path = 100.0 * (1.0 + ret).cumprod()
    return pd.Series(price_path, index=ret.index, name="price")


def _walk_forward_folds(n_rows: int, *, n_folds: int) -> list[tuple[int, int, int]]:
    """Return anchored ``(train_start, train_end, oos_end)`` fold boundaries.

    The train slice is anchored at row 0 and EXPANDS fold by fold; each OOS fold
    is the disjoint block of rows immediately after the (growing) train slice.
    ``train_end`` is exclusive of the OOS fold, so train and OOS never overlap
    (the no-lookahead boundary).

    Parameters
    ----------
    n_rows:
        Total number of feature rows.
    n_folds:
        Number of out-of-sample folds to carve.

    Returns
    -------
    list[tuple[int, int, int]]
        Per-fold ``(train_start, train_end, oos_end)`` positional indices.
    """
    initial_train = max(2, int(n_rows * _INITIAL_TRAIN_FRAC))
    oos_total = n_rows - initial_train
    if oos_total < _MIN_OOS_ROWS:
        raise InsufficientDataError(
            f"run_anomaly_scan: {n_rows} feature rows leave only {oos_total} "
            f"out-of-sample row(s); need at least {_MIN_OOS_ROWS}."
        )
    # Cap folds so each OOS block has at least one row.
    folds = max(1, min(n_folds, oos_total))
    base = oos_total // folds
    remainder = oos_total % folds

    boundaries: list[tuple[int, int, int]] = []
    cursor = initial_train
    for i in range(folds):
        block = base + (1 if i < remainder else 0)
        oos_end = cursor + block
        # Anchored/expanding train: rows [0, cursor) are TRAIN for this fold.
        boundaries.append((0, cursor, oos_end))
        cursor = oos_end
    return boundaries


def run_anomaly_scan(
    prices: PricesLike | None = None,
    *,
    returns: ReturnsLike | None = None,
    detector: DetectorChoice = "both",
    contamination: float = DEFAULT_CONTAMINATION,
    window: int = DEFAULT_WINDOW,
    seed: int = 7,
    n_folds: int = DEFAULT_N_FOLDS,
    data_source: str | None = None,
) -> ScanResult:
    """Run a strictly causal walk-forward anomaly scan over a price/return series.

    Engineers causal per-day features, then runs an ANCHORED walk-forward: on
    each fold the scaler, both detectors, and ALL thresholds are refitted on the
    (expanding) TRAIN slice ONLY and score the disjoint OOS fold. The per-fold
    OOS scores/flags are concatenated into a single out-of-sample series with
    zero look-ahead, and the descriptive agreement summary is assembled.

    Parameters
    ----------
    prices:
        A price level series (or single-column price panel), indexed by date.
        Mutually exclusive with ``returns``.
    returns:
        A per-day return series; integrated to a synthetic price path internally.
        Mutually exclusive with ``prices``.
    detector:
        Which detector the summary reports (``"iforest"``, ``"autoencoder"``, or
        ``"both"``); BOTH detectors are always fitted so the Jaccard agreement
        headline is available.
    contamination:
        Expected anomalous fraction; sets each fold's flag threshold. Must
        satisfy ``0 < contamination < 0.5``.
    window:
        Rolling feature window (trading days); must be ``>= 2``.
    seed:
        Master RNG seed (deterministic detectors).
    n_folds:
        Number of anchored walk-forward OOS folds.
    data_source:
        Optional provenance tag surfaced in the summary (``"polygon"`` |
        ``"synthetic"``).

    Returns
    -------
    ScanResult
        The concatenated OOS detector results, the descriptive agreement, the
        OOS price/return series, and a JSON-safe :meth:`ScanResult.summary`.

    Raises
    ------
    ValidationError
        If neither/both of ``prices``/``returns`` is given, or a parameter is out
        of range.
    InsufficientDataError
        If the series is too short to carve at least one walk-forward fold.
    """
    from anomaly_detector.data import compute_returns
    from anomaly_detector.detectors.autoencoder import PCAAutoencoderDetector
    from anomaly_detector.detectors.iforest import IsolationForestDetector
    from anomaly_detector.detectors.result import AnomalyResult
    from anomaly_detector.evaluation.agreement import compute_agreement
    from anomaly_detector.features.engineer import engineer_features

    if detector not in ("iforest", "autoencoder", "both"):
        raise ValidationError(
            f"run_anomaly_scan: detector must be 'iforest'|'autoencoder'|'both', got {detector!r}."
        )
    if not 0.0 < contamination < 0.5:
        raise ValidationError(
            f"run_anomaly_scan: contamination must satisfy 0 < c < 0.5, got {contamination}."
        )

    price_s = _coerce_prices(prices, returns)
    all_returns = compute_returns(price_s)
    features = engineer_features(price_s, window=window)
    if features.shape[0] < _MIN_OOS_ROWS + 2:
        raise InsufficientDataError(
            f"run_anomaly_scan: only {features.shape[0]} feature row(s) after the "
            f"causal warm-up; need at least {_MIN_OOS_ROWS + 2}."
        )

    folds = _walk_forward_folds(features.shape[0], n_folds=n_folds)

    scores_if: list[pd.Series] = []
    flags_if: list[pd.Series] = []
    scores_ae: list[pd.Series] = []
    flags_ae: list[pd.Series] = []
    n_train_total = 0
    # The final fold's train-derived thresholds back the score figure's reference
    # line — the largest anchored train slice is the most representative cutoff.
    last_threshold_if = 0.0
    last_threshold_ae = 0.0

    for train_start, train_end, oos_end in folds:
        train = features.iloc[train_start:train_end]
        oos = features.iloc[train_end:oos_end]
        if oos.empty:  # pragma: no cover - fold boundaries guarantee non-empty OOS
            continue

        det_if = IsolationForestDetector(contamination=contamination, seed=seed)
        res_if = det_if.fit(train).score(oos)
        det_ae = PCAAutoencoderDetector(contamination=contamination, seed=seed)
        res_ae = det_ae.fit(train).score(oos)

        scores_if.append(res_if.scores)
        flags_if.append(res_if.flags)
        scores_ae.append(res_ae.scores)
        flags_ae.append(res_ae.flags)
        n_train_total += int(train.shape[0])
        last_threshold_if = res_if.threshold
        last_threshold_ae = res_ae.threshold

    oos_index = pd.concat(scores_if).index
    result_if = AnomalyResult(
        scores=pd.concat(scores_if),
        flags=pd.concat(flags_if),
        threshold=last_threshold_if,
        detector="iforest",
        contamination=float(contamination),
        n_train=n_train_total,
        n_test=len(oos_index),
        meta={"window": int(window), "n_folds": len(folds), "walk_forward": True},
    )
    result_ae = AnomalyResult(
        scores=pd.concat(scores_ae),
        flags=pd.concat(flags_ae),
        threshold=last_threshold_ae,
        detector="autoencoder",
        contamination=float(contamination),
        n_train=n_train_total,
        n_test=len(oos_index),
        meta={"window": int(window), "n_folds": len(folds), "walk_forward": True},
    )

    oos_returns = all_returns.reindex(oos_index)
    oos_prices = price_s.reindex(oos_index)

    agreement = compute_agreement(result_if, result_ae, oos_returns, window=window)

    return ScanResult(
        result_iforest=result_if,
        result_autoencoder=result_ae,
        agreement=agreement,
        oos_returns=oos_returns,
        oos_prices=oos_prices,
        detector=detector,
        data_source=data_source,
        meta={
            "window": int(window),
            "contamination": float(contamination),
            "n_folds": len(folds),
            "seed": int(seed),
            "n_oos": len(oos_index),
            "n_train_total": n_train_total,
        },
    )
