"""Rolling-window cluster re-fitting and the StabilityResult bundle.

Re-fits clusters on a sequence of rolling in-sample windows and summarizes how
stable the assignment is over time. The frozen :class:`StabilityResult` carries
the per-window labelings, the headline adjacent-window ARI, and birth/death events.

Importing this module has no side effects.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from stockclusters._exceptions import InsufficientDataError, ValidationError
from stockclusters._typing import ReturnsLike
from stockclusters._validation import ensure_dataframe

__all__ = ["StabilityResult", "rolling_stability"]

#: A window clusterer maps a single in-sample returns window to an integer-label
#: Series indexed by asset. The default builds the standard
#: correlation -> RMT -> Mantegna-distance -> hierarchical pipeline; tests inject
#: a deterministic clusterer so the stability layer is decoupled from it.
WindowClusterer = Callable[[pd.DataFrame], "pd.Series"]


def _default_clusterer(
    *,
    n_clusters: int,
    method: str,
    denoise: bool,
) -> WindowClusterer:
    """Build the standard per-window clusterer (lazy import of the pipeline).

    NO-LOOKAHEAD: the returned callable sees only the window passed to it; it
    estimates the correlation, optionally RMT-denoises it, maps to Mantegna
    distance, and cuts the hierarchy at ``n_clusters`` — all within that window.
    """

    def _cluster(window: pd.DataFrame) -> pd.Series:
        from stockclusters.clustering.hierarchical import hierarchical_clusters
        from stockclusters.correlation.distance import mantegna_distance
        from stockclusters.correlation.estimate import correlation_matrix
        from stockclusters.correlation.rmt import marchenko_pastur_clip

        corr = correlation_matrix(window)
        if denoise:
            corr = marchenko_pastur_clip(corr, n_obs=window.shape[0])
        dist = mantegna_distance(corr)
        result = hierarchical_clusters(dist, n_clusters=n_clusters, method=method)
        return result.labels

    return _cluster


@dataclass(frozen=True, slots=True)
class StabilityResult:
    """Immutable result of a rolling-window cluster-stability analysis.

    Attributes
    ----------
    window_labels:
        Per-window aligned label Series, ordered earliest-to-latest.
    window_dates:
        The (end) date of each window, parallel to ``window_labels``.
    ari_mean:
        The headline mean adjacent-window ARI.
    ari_series:
        The per-adjacent-pair ARI values (``len(window_labels) - 1`` entries).
    n_windows:
        The number of rolling windows evaluated.
    """

    window_labels: list[pd.Series]
    window_dates: list[str]
    ari_mean: float
    ari_series: list[float]
    n_windows: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result.

        Each per-window labeling is rendered as a ``{ticker: cluster_id}`` mapping
        and ``ari_mean`` scrubbed of non-finite values downstream.
        """
        out = asdict(self)
        out["window_labels"] = [
            {str(k): int(v) for k, v in labels.items()} for labels in self.window_labels
        ]
        out["window_dates"] = [str(d) for d in self.window_dates]
        out["ari_mean"] = float(self.ari_mean)
        out["ari_series"] = [float(v) for v in self.ari_series]
        out["n_windows"] = int(self.n_windows)
        return out


def rolling_stability(
    returns: ReturnsLike,
    *,
    n_clusters: int,
    train_window: int = 504,
    step: int = 21,
    method: str = "average",
    denoise: bool = True,
    clusterer: WindowClusterer | None = None,
) -> StabilityResult:
    r"""Re-fit clusters on rolling windows and summarize temporal stability.

    Slides a ``train_window``-length in-sample window across ``returns`` in
    ``step``-sized increments, re-fitting clusters on each window (optionally
    RMT-denoised), aligning labels across windows, and computing the headline
    adjacent-window ARI.

    NO-LOOKAHEAD REQUIREMENT: each window's clustering uses only the observations
    inside that window; windows do not peek forward.

    Parameters
    ----------
    returns:
        A wide panel of asset returns (rows = time, columns = asset).
    n_clusters:
        The (fixed) number of clusters per window.
    train_window:
        The in-sample window length in observations.
    step:
        The stride between consecutive windows in observations.
    method:
        Linkage method for the per-window clustering.
    denoise:
        Whether to RMT-denoise each window's covariance before clustering.
    clusterer:
        Optional window-clusterer override mapping an in-sample window to a label
        Series. When ``None`` the standard correlation -> RMT -> Mantegna ->
        hierarchical pipeline is used. Injecting a clusterer keeps the stability
        layer decoupled from (and testable without) the clustering layer.

    Returns
    -------
    StabilityResult
        The frozen stability bundle.

    Raises
    ------
    ValidationError
        If ``step < 1`` or ``n_clusters < 1``.
    InsufficientDataError
        If ``train_window`` exceeds the available observations.
    """
    if step < 1:
        raise ValidationError(f"rolling_stability: step must be >= 1, got {step}.")
    if n_clusters < 1:
        raise ValidationError(f"rolling_stability: n_clusters must be >= 1, got {n_clusters}.")

    panel = ensure_dataframe(returns, name="returns")
    n_obs = panel.shape[0]
    if train_window < 2:
        raise ValidationError(f"rolling_stability: train_window must be >= 2, got {train_window}.")
    if train_window > n_obs:
        raise InsufficientDataError(
            f"rolling_stability: train_window ({train_window}) exceeds the "
            f"available observations ({n_obs})."
        )

    fit = clusterer or _default_clusterer(n_clusters=n_clusters, method=method, denoise=denoise)

    # Slide a fixed-length in-sample window across the panel. Each window sees ONLY
    # its own observations — no peeking forward — and the last window ends exactly
    # at the final observation when the spacing allows.
    starts = list(range(0, n_obs - train_window + 1, step))

    raw_labels: list[pd.Series] = []
    window_dates: list[str] = []
    for start in starts:
        end = start + train_window
        window = panel.iloc[start:end]
        labels = fit(window).astype(int)
        raw_labels.append(labels)
        window_dates.append(str(panel.index[end - 1]))

    # Align each window's (arbitrary) integer labels to the previous window so
    # adjacent-window ARI and births/deaths are computed on a consistent id space.
    from stockclusters.stability.align import align_labels
    from stockclusters.stability.ari import adjacent_window_ari, adjusted_rand_index

    aligned_labels: list[pd.Series] = []
    if raw_labels:
        aligned_labels.append(raw_labels[0])
        for i in range(1, len(raw_labels)):
            aligned_labels.append(align_labels(aligned_labels[i - 1], raw_labels[i]))

    ari_series = [
        adjusted_rand_index(aligned_labels[i], aligned_labels[i + 1])
        for i in range(len(aligned_labels) - 1)
    ]
    ari_mean = adjacent_window_ari(aligned_labels)

    return StabilityResult(
        window_labels=aligned_labels,
        window_dates=window_dates,
        ari_mean=ari_mean,
        ari_series=ari_series,
        n_windows=len(aligned_labels),
        meta={
            "train_window": int(train_window),
            "step": int(step),
            "method": str(method),
            "denoise": bool(denoise),
            "n_clusters": int(n_clusters),
        },
    )
