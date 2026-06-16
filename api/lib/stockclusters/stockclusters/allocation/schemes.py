"""Cluster-aware allocation schemes and the diversification result bundle.

Each scheme maps a cluster labeling (and, where relevant, a covariance estimate)
to a weight vector on the simplex:

- :func:`one_over_n_weights` — naive equal weight across all assets (the honest
  benchmark the clustering must beat).
- :func:`cluster_equal_weight` — equal weight per cluster, split equally within.
- :func:`stripped_hrp_weights` — inverse-variance within cluster, equal across
  clusters ("stripped-HRP": the cross-cluster recursive bisection of HRP replaced
  by a flat equal split).

All schemes return labelled simplex weights; the walk-forward engine applies them
with ``shift(1)``.

Importing this module has no side effects.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from stockclusters._exceptions import ValidationError
from stockclusters._typing import MatrixLike, ReturnsLike

__all__ = [
    "DiversificationResult",
    "cluster_equal_weight",
    "one_over_n_weights",
    "run_diversification",
    "stripped_hrp_weights",
]


@dataclass(frozen=True, slots=True)
class DiversificationResult:
    """Immutable result of the cluster-vs-1/N diversification horse race.

    Attributes
    ----------
    one_over_n_sharpe:
        Annualized OOS Sharpe of the naive 1/N portfolio.
    cluster_ew_sharpe:
        Annualized OOS Sharpe of the equal-weight-across-cluster portfolio.
    stripped_hrp_sharpe:
        Annualized OOS Sharpe of the stripped-HRP portfolio.
    sharpe_diff_vs_1overN:
        The best cluster-aware Sharpe minus the 1/N Sharpe.
    memmel_jk_pvalue:
        Jobson-Korkie-Memmel two-sided p-value on that Sharpe gap.
    deflated_sharpe:
        Deflated Sharpe of the selected cluster-aware strategy under the FULL
        trial count.
    n_trials:
        The full DSR trial count (product of every swept axis).
    cost_bps:
        The per-side transaction cost (basis points) used in the OOS backtest.
    """

    one_over_n_sharpe: float
    cluster_ew_sharpe: float
    stripped_hrp_sharpe: float
    sharpe_diff_vs_1overN: float
    memmel_jk_pvalue: float
    deflated_sharpe: float
    n_trials: int
    cost_bps: float
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result.

        The ``meta`` block carries scalar diagnostics only; the raw OOS return
        series (stashed under ``meta["oos_curves"]`` for figure assembly) are
        dropped here so the dict crosses the API boundary without pandas types.
        """
        out = asdict(self)
        for key in (
            "one_over_n_sharpe",
            "cluster_ew_sharpe",
            "stripped_hrp_sharpe",
            "sharpe_diff_vs_1overN",
            "memmel_jk_pvalue",
            "deflated_sharpe",
            "cost_bps",
        ):
            out[key] = float(getattr(self, key))
        out["n_trials"] = int(self.n_trials)
        meta = out.get("meta")
        if isinstance(meta, dict):
            out["meta"] = {k: v for k, v in meta.items() if k != "oos_curves"}
        return out


def one_over_n_weights(assets: list[str]) -> pd.Series:
    r"""Naive 1/N equal-weight portfolio (the honest benchmark).

    Parameters
    ----------
    assets:
        The asset tickers in the universe.

    Returns
    -------
    pandas.Series
        Equal weights ``1 / N`` indexed by asset, summing to one.

    Raises
    ------
    ValidationError
        If ``assets`` is empty.
    """
    if len(assets) == 0:
        raise ValidationError("one_over_n_weights: assets must be non-empty.")
    n = len(assets)
    return pd.Series(np.full(n, 1.0 / n, dtype="float64"), index=list(assets))


def cluster_equal_weight(labels: pd.Series) -> pd.Series:
    r"""Equal weight across clusters, equal weight within each cluster.

    Allocates ``1 / K`` to each of the ``K`` clusters and splits each cluster's
    budget equally among its members.

    Parameters
    ----------
    labels:
        Integer cluster labels indexed by asset ticker.

    Returns
    -------
    pandas.Series
        Simplex weights indexed by asset, summing to one.

    Raises
    ------
    ValidationError
        If ``labels`` is empty.
    """
    if not isinstance(labels, pd.Series) or labels.empty:
        raise ValidationError("cluster_equal_weight: labels must be a non-empty Series.")

    labels_int = labels.astype(int)
    cluster_ids = pd.unique(labels_int)
    k = len(cluster_ids)
    per_cluster = 1.0 / k

    weights = pd.Series(0.0, index=labels.index, dtype="float64")
    for cid in cluster_ids:
        members = labels_int.index[labels_int.to_numpy() == cid]
        weights.loc[members] = per_cluster / len(members)
    return weights


def stripped_hrp_weights(labels: pd.Series, cov: MatrixLike) -> pd.Series:
    r"""Stripped-HRP: inverse-variance within cluster, equal across clusters.

    Allocates ``1 / K`` to each cluster and, within each cluster, weights members
    inversely to their variances (the within-cluster step of HRP) — but replaces
    HRP's cross-cluster recursive bisection with a flat equal split across
    clusters ("stripped").

    Parameters
    ----------
    labels:
        Integer cluster labels indexed by asset ticker.
    cov:
        The ``N x N`` covariance matrix (provides per-asset variances), labelled
        by asset.

    Returns
    -------
    pandas.Series
        Simplex weights indexed by asset, summing to one.

    Raises
    ------
    ValidationError
        If ``labels`` and ``cov`` labels disagree or ``cov`` has a non-positive
        diagonal.
    """
    if not isinstance(labels, pd.Series) or labels.empty:
        raise ValidationError("stripped_hrp_weights: labels must be a non-empty Series.")

    cov_df = cov if isinstance(cov, pd.DataFrame) else pd.DataFrame(cov)
    n_rows, n_cols = cov_df.shape
    if n_rows != n_cols:
        raise ValidationError(
            f"stripped_hrp_weights: cov must be square, got shape {(n_rows, n_cols)}."
        )

    labels_int = labels.astype(int)
    assets = list(labels_int.index)

    # When cov carries asset labels they must agree with the labeling; otherwise
    # the two are aligned positionally and must share the same dimension.
    cov_has_labels = not (
        cov_df.index.equals(pd.RangeIndex(n_rows)) and cov_df.columns.equals(pd.RangeIndex(n_cols))
    )
    if cov_has_labels:
        if set(cov_df.index) != set(assets) or set(cov_df.columns) != set(assets):
            raise ValidationError(
                "stripped_hrp_weights: cov labels do not match the cluster labeling."
            )
        cov_df = cov_df.reindex(index=assets, columns=assets)
    else:
        if n_rows != len(assets):
            raise ValidationError(
                "stripped_hrp_weights: unlabelled cov dimension "
                f"({n_rows}) does not match the number of assets ({len(assets)})."
            )
        cov_df = pd.DataFrame(cov_df.to_numpy(), index=assets, columns=assets)

    diag = np.diag(cov_df.to_numpy(dtype="float64"))
    if not np.all(np.isfinite(diag)) or np.any(diag <= 0.0):
        raise ValidationError(
            "stripped_hrp_weights: cov has a non-positive or non-finite diagonal."
        )

    cluster_ids = pd.unique(labels_int)
    k = len(cluster_ids)
    per_cluster = 1.0 / k

    weights = pd.Series(0.0, index=assets, dtype="float64")
    for cid in cluster_ids:
        members = [a for a in assets if labels_int.loc[a] == cid]
        # Within-cluster inverse-variance (the HRP IVP step): weight members
        # inversely to their own variance. The full-cov diagonal is already
        # validated positive + finite above, so this slice is safe.
        sub_var = np.diag(cov_df.loc[members, members].to_numpy(dtype="float64"))
        inv_var = 1.0 / sub_var
        intra = inv_var / inv_var.sum()
        weights.loc[members] = per_cluster * intra
    return weights


def run_diversification(
    returns: ReturnsLike,
    labels: pd.Series,
    *,
    lookback_window: int,
    n_trials: int,
    cost_bps: float = 5.0,
    rebalance: str = "monthly",
    embargo: int = 1,
    purge: int = 1,
) -> DiversificationResult:
    r"""Run the honest 1/N vs cluster-aware OOS horse race and fill the result.

    Backtests the three strategies (1/N, cluster-EW, stripped-HRP) through the
    SAME no-lookahead walk-forward engine — frozen ``labels`` are reused on every
    rebalance, and each window's covariance is estimated from the in-sample data
    only — then runs the inference layer on the comparison.

    IDENTICAL-INDEX REQUIREMENT: all three net OOS return series are asserted to
    share the EXACT same post-purge/embargo date index BEFORE the Jobson-Korkie-
    Memmel test is computed. A mismatch is a leakage/alignment bug and raises.

    DSR HONESTY: ``n_trials`` is supplied by the caller as the FULL product of the
    swept axes (``#linkages x #k-candidates x #weighting-schemes x #denoise-settings
    x #cost-grid-points``); the deflated Sharpe deflates the *selected* (best
    cluster-aware) strategy against that full multiplicity. Under-counting
    manufactures false significance and is guarded against in the regression suite.

    Parameters
    ----------
    returns:
        A wide panel of asset returns (rows = time, columns = asset).
    labels:
        Frozen integer cluster labels indexed by asset ticker (fit on TRAIN data
        upstream, never re-fit per OOS window here).
    lookback_window:
        The in-sample window length for each rebalance.
    n_trials:
        The FULL DSR trial count (product of every swept axis). Must be ``>= 1``.
    cost_bps:
        Per-side transaction cost in basis points charged on turnover.
    rebalance:
        Rebalance cadence (``"monthly"`` or ``"quarterly"``).
    embargo, purge:
        No-lookahead gap parameters passed to the walk-forward engine.

    Returns
    -------
    DiversificationResult
        The frozen horse-race bundle (three Sharpes, the best-vs-1/N gap, the
        Memmel-JK p-value, the deflated Sharpe, ``n_trials``, ``cost_bps``).

    Raises
    ------
    ValidationError
        If ``n_trials < 1``, ``labels`` is empty, or the three OOS return series
        do not share an identical date index.
    """
    from stockclusters.backtest.stats import sharpe_ratio
    from stockclusters.backtest.walk_forward import walk_forward_backtest
    from stockclusters.estimators.covariance import ledoit_wolf_cov
    from stockclusters.evaluation.comparison import jobson_korkie_memmel
    from stockclusters.evaluation.dsr import deflated_sharpe_ratio

    if n_trials < 1:
        raise ValidationError(f"run_diversification: n_trials must be >= 1, got {n_trials}.")
    if not isinstance(labels, pd.Series) or labels.empty:
        raise ValidationError("run_diversification: labels must be a non-empty Series.")

    frozen_labels = labels.astype(int)
    assets = list(frozen_labels.index)

    # --- Build the three allocators (frozen labels; window-local covariance) ---
    def _alloc_one_over_n(window: pd.DataFrame) -> pd.Series:
        return one_over_n_weights(list(window.columns))

    def _alloc_cluster_ew(window: pd.DataFrame) -> pd.Series:
        sub = frozen_labels.reindex(window.columns).dropna().astype(int)
        return cluster_equal_weight(sub).reindex(window.columns).fillna(0.0)

    def _alloc_stripped_hrp(window: pd.DataFrame) -> pd.Series:
        sub = frozen_labels.reindex(window.columns).dropna().astype(int)
        cov = ledoit_wolf_cov(window[list(sub.index)])
        return stripped_hrp_weights(sub, cov).reindex(window.columns).fillna(0.0)

    bt_kwargs: dict[str, Any] = {
        "lookback_window": lookback_window,
        "rebalance": rebalance,
        "cost_bps": float(cost_bps),
        "embargo": embargo,
        "purge": purge,
    }
    bt_1n = walk_forward_backtest(returns, _alloc_one_over_n, **bt_kwargs)
    bt_cew = walk_forward_backtest(returns, _alloc_cluster_ew, **bt_kwargs)
    bt_shrp = walk_forward_backtest(returns, _alloc_stripped_hrp, **bt_kwargs)

    oos_1n = bt_1n.oos_returns
    oos_cew = bt_cew.oos_returns
    oos_shrp = bt_shrp.oos_returns

    # --- IDENTICAL OOS INDEX GUARD (must pass BEFORE Memmel-JK) ----------------
    # The three strategies are driven by the same engine over the same panel, so
    # their post-purge/embargo OOS date indexes MUST be byte-identical. If they
    # are not, the comparison is mis-aligned (a leakage or rebalance bug) and we
    # refuse to compute a Sharpe-difference p-value on it.
    if not (oos_1n.index.equals(oos_cew.index) and oos_1n.index.equals(oos_shrp.index)):
        raise ValidationError(
            "run_diversification: the three strategies produced non-identical OOS "
            "date indexes; refusing to run Memmel-JK on misaligned series."
        )

    # Convert to numpy arrays for the (positional) inference calls; the indexes
    # are already asserted identical above, so positional alignment is exact.
    arr_1n = oos_1n.to_numpy(dtype="float64")
    arr_cew = oos_cew.to_numpy(dtype="float64")
    arr_shrp = oos_shrp.to_numpy(dtype="float64")

    sharpe_1n = sharpe_ratio(arr_1n)
    sharpe_cew = sharpe_ratio(arr_cew)
    sharpe_shrp = sharpe_ratio(arr_shrp)

    # The "selected" cluster-aware strategy is the better of the two by OOS Sharpe
    # (a real selection step the DSR multiplicity must account for).
    cluster_sharpes = {"cluster_ew": sharpe_cew, "stripped_hrp": sharpe_shrp}
    cluster_arrays = {"cluster_ew": arr_cew, "stripped_hrp": arr_shrp}

    def _finite(value: float) -> float:
        return value if math.isfinite(value) else float("-inf")

    selected = max(cluster_sharpes, key=lambda k: _finite(cluster_sharpes[k]))
    best_cluster_sharpe = cluster_sharpes[selected]
    sel_arr = cluster_arrays[selected]

    sharpe_diff = best_cluster_sharpe - sharpe_1n

    # --- Memmel-JK p-value on the selected-vs-1/N Sharpe gap -------------------
    memmel_p = jobson_korkie_memmel(sel_arr, arr_1n)

    # --- Deflated Sharpe of the selected strategy under the FULL trial count ---
    n_oos = sel_arr.shape[0]
    std = float(sel_arr.std(ddof=1)) if n_oos > 1 else 0.0
    per_period_sr = float(sel_arr.mean()) / std if std > 0.0 else 0.0

    # Cross-trial Sharpe dispersion: with the three realized per-period Sharpes as
    # the observed sample of the trial distribution, the variance of trial Sharpes
    # feeds the expected-maximum benchmark. Floor at a tiny positive value so a
    # degenerate (identical) trio still deflates rather than collapsing to PSR.
    trial_srs = []
    for a in (arr_1n, arr_cew, arr_shrp):
        s = float(a.std(ddof=1)) if a.shape[0] > 1 else 0.0
        trial_srs.append(float(a.mean()) / s if s > 0.0 else 0.0)
    var_trials = float(np.var(trial_srs, ddof=0))
    if var_trials <= 0.0:
        var_trials = 1.0 / max(n_oos, 2)

    skew, kurt = _sample_skew_kurtosis(sel_arr)
    deflated = (
        deflated_sharpe_ratio(
            per_period_sr,
            n_obs=n_oos,
            n_trials=n_trials,
            variance_of_trial_sharpes=var_trials,
            skew=skew,
            kurtosis=kurt,
        )
        if n_oos > 1
        else 0.0
    )

    return DiversificationResult(
        one_over_n_sharpe=_nan_safe(sharpe_1n),
        cluster_ew_sharpe=_nan_safe(sharpe_cew),
        stripped_hrp_sharpe=_nan_safe(sharpe_shrp),
        sharpe_diff_vs_1overN=_nan_safe(sharpe_diff),
        memmel_jk_pvalue=_nan_safe(memmel_p),
        deflated_sharpe=_nan_safe(deflated),
        n_trials=int(n_trials),
        cost_bps=float(cost_bps),
        meta={
            "selected_strategy": selected,
            "n_oos": int(n_oos),
            "rebalance": str(rebalance),
            "lookback_window": int(lookback_window),
            "n_assets": len(assets),
            "variance_of_trial_sharpes": var_trials,
            # Raw OOS return series for equity-curve figure assembly. Dropped from
            # to_dict() so the JSON summary stays pandas-free.
            "oos_curves": {
                "1/N": oos_1n,
                "cluster-EW": oos_cew,
                "stripped-HRP": oos_shrp,
            },
        },
    )


def _nan_safe(value: float) -> float:
    """Map a non-finite scalar to ``nan`` (rendered ``None`` downstream)."""
    return value if math.isfinite(value) else float("nan")


def _sample_skew_kurtosis(arr: np.ndarray) -> tuple[float, float]:
    """Return ``(skew, full_kurtosis)`` of ``arr`` (Gaussian -> ``(0, 3)``).

    Uses the population (biased) moments, matching the convention the DSR's PSR
    bracket expects (full, non-excess kurtosis).
    """
    a = arr.astype("float64")
    n = a.shape[0]
    if n < 2:
        return 0.0, 3.0
    mean = float(a.mean())
    centered = a - mean
    m2 = float((centered**2).mean())
    if m2 <= 0.0:
        return 0.0, 3.0
    m3 = float((centered**3).mean())
    m4 = float((centered**4).mean())
    skew = m3 / m2**1.5
    kurt = m4 / m2**2  # full (non-excess) kurtosis
    return skew, kurt
