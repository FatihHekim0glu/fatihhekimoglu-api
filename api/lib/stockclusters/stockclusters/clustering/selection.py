"""Number-of-clusters selection: gap statistic vs a phase-randomized null.

The pre-registered selector is the Tibshirani (2001) gap statistic evaluated
against a **phase-randomized** null reference distribution, with the **1-SE rule**
choosing the smallest ``k`` whose gap is within one standard error of the maximum.
Silhouette and MST modularity are reported as cross-checks but do NOT override the
gap selection.

The phase-randomized null FFT-phase-scrambles each asset's return series
independently: it preserves each asset's marginal power spectrum (autocorrelation)
while destroying cross-asset correlation, giving a null with realistic single-asset
dynamics but no genuine cluster structure.

DSR REQUIREMENT: every ``k`` candidate evaluated here is a swept axis; the count of
candidates feeds the DSR ``n_trials`` product. All trials are recorded on
:class:`GapResult`.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from stockclusters._exceptions import ValidationError
from stockclusters._rng import make_rng
from stockclusters._typing import MatrixLike, ReturnsLike
from stockclusters._validation import ensure_dataframe

__all__ = ["GapResult", "phase_randomize", "pooled_within_dispersion", "select_k_gap"]


@dataclass(frozen=True, slots=True)
class GapResult:
    """Immutable result of gap-statistic cluster-number selection.

    Attributes
    ----------
    k_selected:
        The selected number of clusters under the Tibshirani 1-SE rule.
    k_candidates:
        The full list of ``k`` values evaluated (a swept DSR axis).
    gap:
        The gap statistic ``E*[log W_k] - log W_k`` at each candidate ``k``.
    gap_se:
        The standard error ``s_k`` of the gap at each candidate ``k``.
    log_wk:
        The observed ``log W_k`` (pooled within-cluster dispersion) at each ``k``.
    selection_rule:
        The selector identifier (``"tibshirani_1se"``).
    n_trials:
        The number of cluster-number candidates evaluated (``len(k_candidates)``);
        a multiplicative factor in the DSR trial count.
    """

    k_selected: int
    k_candidates: list[int]
    gap: list[float]
    gap_se: list[float]
    log_wk: list[float]
    selection_rule: str = "tibshirani_1se"
    n_trials: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        out = asdict(self)
        out["k_selected"] = int(self.k_selected)
        out["k_candidates"] = [int(k) for k in self.k_candidates]
        out["gap"] = [float(v) for v in self.gap]
        out["gap_se"] = [float(v) for v in self.gap_se]
        out["log_wk"] = [float(v) for v in self.log_wk]
        out["n_trials"] = int(self.n_trials)
        return out


def phase_randomize(returns: ReturnsLike, *, seed: int = 0) -> np.ndarray:
    r"""FFT phase-randomized surrogate of a returns panel.

    Independently phase-scrambles each asset's return series in the Fourier
    domain: the amplitude spectrum (hence the marginal autocorrelation) is
    preserved while the phases are randomized, destroying cross-asset correlation.
    This is the null reference for the gap statistic.

    REPRODUCIBILITY: phases are drawn from a seeded PCG64 generator, so a fixed
    ``seed`` reproduces the surrogate byte-for-byte.

    Parameters
    ----------
    returns:
        A wide panel of asset returns (rows = time, columns = asset).
    seed:
        Master seed for the random phases.

    Returns
    -------
    numpy.ndarray
        A surrogate returns array of the same shape, with preserved per-asset
        marginal spectra and destroyed cross-correlation.

    Raises
    ------
    ValidationError
        If ``returns`` has fewer than two observations.
    """
    frame = ensure_dataframe(returns, name="returns")
    x = frame.to_numpy(dtype=np.float64)
    n_obs, n_assets = x.shape
    if n_obs < 2:
        raise ValidationError(f"phase_randomize needs at least two observations, got {n_obs}.")

    gen = make_rng(seed)

    # Real FFT along time for every asset column at once.
    fx = np.fft.rfft(x, axis=0)
    n_freq = fx.shape[0]
    amplitude = np.abs(fx)

    # Random phases, drawn INDEPENDENTLY per asset (destroys cross-correlation
    # while each asset keeps its own amplitude spectrum => marginal ACF preserved).
    phases = gen.uniform(0.0, 2.0 * np.pi, size=(n_freq, n_assets))
    # Preserve the DC component (index 0) exactly: it carries the series mean.
    phases[0, :] = 0.0
    # For even-length series the Nyquist bin must stay real-valued.
    if n_obs % 2 == 0:
        phases[-1, :] = 0.0

    surrogate_freq = amplitude * np.exp(1j * phases)
    surrogate = np.fft.irfft(surrogate_freq, n=n_obs, axis=0)
    return np.asarray(surrogate, dtype=np.float64)


def pooled_within_dispersion(dist: np.ndarray, labels: np.ndarray) -> float:
    r"""Tibshirani pooled within-cluster dispersion ``W_k``.

    :math:`W_k = \sum_r \frac{1}{2 n_r} D_r`, where :math:`D_r` is the sum of all
    pairwise distances within cluster ``r`` and :math:`n_r` its size. Computed
    directly on a precomputed distance matrix (the Mantegna distances), which is
    the geometry the clustering is performed on.
    """
    total = 0.0
    for lab in np.unique(labels):
        idx = np.flatnonzero(labels == lab)
        n_r = idx.size
        if n_r < 2:
            continue
        sub = dist[np.ix_(idx, idx)]
        d_r = float(sub.sum())  # counts each pair twice; matches 1/(2 n_r) factor
        total += d_r / (2.0 * n_r)
    return total


def select_k_gap(
    returns: ReturnsLike,
    dist: MatrixLike,
    *,
    k_min: int = 2,
    k_max: int = 20,
    method: str = "average",
    n_references: int = 20,
    seed: int = 0,
) -> GapResult:
    r"""Select the number of clusters via the gap statistic (1-SE rule).

    For each ``k`` in ``[k_min, k_max]`` computes the observed pooled
    within-cluster dispersion ``log W_k`` and its expectation under ``n_references``
    phase-randomized surrogates, forms the gap ``E*[log W_k] - log W_k`` and its
    standard error ``s_k``, and applies the Tibshirani 1-SE rule: the smallest ``k``
    with ``gap(k) >= gap(k+1) - s_{k+1}``.

    HONESTY REQUIREMENT: the null is the phase-randomized reference (preserves
    marginal spectra, destroys cross-correlation), NOT a uniform-box null. The
    uniform null is used only to validate the gap code path against a reference,
    never to select ``k``. Silhouette / MST modularity are reported cross-checks,
    not selectors.

    DSR REQUIREMENT: ``GapResult.n_trials`` records ``len(k_candidates)`` so the
    caller can fold it into the full DSR trial-count product.

    Parameters
    ----------
    returns:
        The in-sample returns panel used to draw phase-randomized surrogates.
    dist:
        The Mantegna distance matrix the clustering is performed on.
    k_min, k_max:
        Inclusive range of cluster counts to evaluate.
    method:
        Linkage method used when computing ``W_k`` (defaults to ``"average"``).
    n_references:
        Number of phase-randomized reference draws ``B``.
    seed:
        Master seed for the reference draws.

    Returns
    -------
    GapResult
        The frozen selection bundle, including all evaluated candidates.

    Raises
    ------
    ValidationError
        If ``k_min < 1``, ``k_max < k_min``, or ``k_max`` exceeds the asset count.
    """
    # Lazy intra-package imports keep this module import-pure and avoid cycles.
    from scipy.cluster.hierarchy import linkage as _scipy_linkage
    from scipy.spatial.distance import squareform

    from stockclusters.clustering.hierarchical import cut_tree
    from stockclusters.correlation.distance import mantegna_distance
    from stockclusters.correlation.estimate import correlation_matrix

    returns_frame = ensure_dataframe(returns, name="returns")
    dist_frame = ensure_dataframe(dist, name="dist")
    n_assets = int(dist_frame.shape[1])
    if dist_frame.shape[0] != dist_frame.shape[1]:
        raise ValidationError(f"dist must be square, got shape {dist_frame.shape}.")
    if int(returns_frame.shape[1]) != n_assets:
        raise ValidationError(
            "returns and dist must describe the same assets "
            f"({returns_frame.shape[1]} vs {n_assets})."
        )

    if int(k_min) < 1:
        raise ValidationError(f"k_min must be >= 1, got {k_min}.")
    if int(k_max) < int(k_min):
        raise ValidationError(f"k_max ({k_max}) must be >= k_min ({k_min}).")
    if int(k_max) > n_assets:
        raise ValidationError(f"k_max ({k_max}) must not exceed the asset count ({n_assets}).")
    if int(n_references) < 1:
        raise ValidationError(f"n_references must be >= 1, got {n_references}.")

    k_candidates = list(range(int(k_min), int(k_max) + 1))
    asset_labels = list(dist_frame.columns.astype(str))

    def _linkage_from_dist(d: np.ndarray) -> np.ndarray:
        sym = 0.5 * (d + d.T)
        np.fill_diagonal(sym, 0.0)
        condensed = squareform(sym, checks=False)
        return np.asarray(_scipy_linkage(condensed, method=method), dtype=np.float64)

    def _logwk_curve(d: np.ndarray) -> np.ndarray:
        """log W_k for every candidate k on distance matrix ``d``."""
        link = _linkage_from_dist(d)
        out = np.empty(len(k_candidates), dtype=np.float64)
        for i, k in enumerate(k_candidates):
            labels = cut_tree(link, n_clusters=k, labels=asset_labels).to_numpy()
            w_k = pooled_within_dispersion(d, labels)
            # Floor W_k away from zero so log is finite (singletons => W_k == 0).
            out[i] = float(np.log(max(w_k, 1e-300)))
        return out

    observed_dist = dist_frame.to_numpy(dtype=np.float64)
    log_wk_obs = _logwk_curve(observed_dist)

    # Reference curves from B phase-randomized surrogates: scramble returns,
    # recompute correlation -> Mantegna distance -> log W_k.
    ref_curves = np.empty((int(n_references), len(k_candidates)), dtype=np.float64)
    for b in range(int(n_references)):
        surrogate = phase_randomize(returns_frame, seed=int(seed) + b)
        surr_df = pd.DataFrame(surrogate, columns=asset_labels)
        surr_corr = correlation_matrix(surr_df)
        surr_dist = mantegna_distance(surr_corr).to_numpy(dtype=np.float64)
        ref_curves[b, :] = _logwk_curve(surr_dist)

    ref_mean = ref_curves.mean(axis=0)
    # Gap(k) = E*[log W_k] - log W_k(observed).
    gap = ref_mean - log_wk_obs
    # Standard error s_k = sd_k * sqrt(1 + 1/B) (Tibshirani 2001).
    sd_k = ref_curves.std(axis=0, ddof=0)
    s_k = sd_k * np.sqrt(1.0 + 1.0 / float(n_references))

    # Tibshirani 1-SE rule: smallest k with Gap(k) >= Gap(k+1) - s_{k+1}.
    k_selected = k_candidates[-1]
    for i in range(len(k_candidates) - 1):
        if gap[i] >= gap[i + 1] - s_k[i + 1]:
            k_selected = k_candidates[i]
            break

    return GapResult(
        k_selected=int(k_selected),
        k_candidates=k_candidates,
        gap=[float(v) for v in gap],
        gap_se=[float(v) for v in s_k],
        log_wk=[float(v) for v in log_wk_obs],
        selection_rule="tibshirani_1se",
        n_trials=len(k_candidates),
        meta={"n_references": int(n_references), "method": method},
    )
