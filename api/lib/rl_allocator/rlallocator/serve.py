"""Serve entrypoint the backend calls (onnxruntime + numpy, NO torch / sb3 / gymnasium).

The FastAPI router calls :func:`run_allocation` to evaluate the committed PPO policy +
the live baselines on the synthetic multi-asset panel (or a real cross-asset basket
loaded via the CLI path) and return a JSON-safe summary plus two Plotly figures. The
equal-weight / Markowitz / risk-parity baselines are computed LIVE (pure numpy); the
RL allocator is served from its committed ONNX policy via onnxruntime — torch / sb3 /
gymnasium are NEVER imported. The honest ``rl_beats_baselines`` verdict is the PURE
function of the inference outputs (median-seed DM vs. the BEST baseline AND DSR >
1-alpha AND across-seed Sharpe lower bound > 0 AND PBO < 0.5).

WALK-FORWARD WIRED INTO THE SERVED PATH: :func:`run_allocation` computes the headline
OOS metrics from the CONCATENATED purged walk-forward folds
(:func:`rlallocator.walk_forward.make_folds`), NOT the full sample. A regression test
asserts the served path calls the walk-forward fn (the recurring "entrypoint bypasses
the rigorous fn" bug). The committed offline metrics (``artifacts/metrics.json``) are
read for the RL-side seed lottery / DSR / PBO; the request path NEVER trains.

Importing this module has no side effects (onnxruntime / plotly are imported lazily
inside the serve / figure paths).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rlallocator._exceptions import ValidationError
from rlallocator.walk_forward import Fold, make_folds

if TYPE_CHECKING:
    from rlallocator._typing import FloatArray

# quantcore-candidate: mirrors rl-trader:src/rltrader/serve.py, GENERALIZED to the
# multi-asset baselines + the walk-forward-wired OOS metric computation.

#: The package's committed-artifacts directory (ONNX policy + metrics.json).
_ARTIFACTS_DIR: Path = Path(__file__).resolve().parent / "artifacts"
#: Committed precomputed-metrics filename.
_METRICS_FILENAME: str = "metrics.json"
#: The hard cap on ``n_seeds`` the router enforces (mirrored here defensively).
_MAX_SEEDS: int = 16
#: The hard cap on ``n_assets`` the router enforces (mirrored here defensively).
_MAX_ASSETS: int = 20
#: A synthetic panel length comfortably longer than one OOS window (warm-up + room).
_MIN_N_OBS: int = 512
#: Number of purged walk-forward folds the served OOS metrics are concatenated over.
_N_WALK_FORWARD_FOLDS: int = 4
#: The live baselines compared against (and reported alongside) the RL allocator.
_BASELINE_NAMES: tuple[str, ...] = ("equal_weight", "markowitz", "risk_parity")


@dataclass(frozen=True, slots=True)
class RlAllocatorSummary:
    """Immutable, JSON-safe summary of the RL-vs-baselines multi-asset comparison.

    Attributes
    ----------
    oos_sharpe_rl_median:
        The MEDIAN-seed OOS net Sharpe of the PPO allocator (net of turnover costs).
    oos_sharpe_1n:
        The equal-weight (1/N) OOS net Sharpe.
    oos_sharpe_markowitz:
        The Markowitz minimum-variance OOS net Sharpe.
    oos_sharpe_riskparity:
        The risk-parity OOS net Sharpe.
    best_baseline:
        The name of the best baseline (highest OOS Sharpe) the RL agent must beat.
    seed_sharpe_lo:
        The across-seed OOS-Sharpe LOWER bound (the seed-lottery dispersion floor).
    seed_sharpe_hi:
        The across-seed OOS-Sharpe UPPER bound.
    dm_pvalue_vs_best:
        The Diebold-Mariano p-value of the median-seed RL net return vs. the best
        baseline.
    deflated_sharpe:
        The Deflated Sharpe (honest seed x HP ``n_trials``) of the median-seed RL.
    pbo:
        The CSCV Probability of Backtest Overfitting across the configurations.
    turnover:
        The median-seed RL allocator's total one-way turnover.
    max_drawdown:
        The median-seed RL allocator's worst peak-to-trough drawdown (``<= 0``).
    rl_beats_baselines:
        The PURE verdict: ``True`` iff the median-seed beats the BEST baseline
        DM-significant AND DSR > 1-alpha AND seed-lo > 0 AND PBO < 0.5, net of costs.
    n_effective_trials:
        The honest multiplicity count used for the DSR (#seeds x #HP configs).
    data_source:
        Provenance of the input panel (``"synthetic"`` / ``"polygon"`` / ``"eodhd"``).
    """

    oos_sharpe_rl_median: float
    oos_sharpe_1n: float
    oos_sharpe_markowitz: float
    oos_sharpe_riskparity: float
    best_baseline: str
    seed_sharpe_lo: float
    seed_sharpe_hi: float
    dm_pvalue_vs_best: float
    deflated_sharpe: float
    pbo: float
    turnover: float
    max_drawdown: float
    rl_beats_baselines: bool
    n_effective_trials: int
    data_source: str

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this summary."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RlAllocatorRun:
    """Immutable bundle returned to the backend: summary + two Plotly figures.

    Attributes
    ----------
    summary:
        The :class:`RlAllocatorSummary`.
    equity_figure:
        A Plotly ``{data, layout}`` dict: the RL median equity curve + the three
        baselines + the across-seed band.
    weights_figure:
        A Plotly ``{data, layout}`` dict: the RL allocation-over-time area chart.
    """

    summary: RlAllocatorSummary
    equity_figure: dict[str, Any] = field(default_factory=dict)
    weights_figure: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this run."""
        return {
            "summary": self.summary.to_dict(),
            "equity_figure": self.equity_figure,
            "weights_figure": self.weights_figure,
        }


def run_allocation(
    *,
    n_assets: int = 6,
    n_seeds: int = 5,
    cost_bps: float = 10.0,
    lookback: int = 64,
    rebalance: str = "monthly",
    data_source_pref: str = "synthetic",
    seed: int = 7,
) -> RlAllocatorRun:
    """Run the end-to-end RL-vs-baselines allocation; return a JSON-safe summary + figures.

    Builds (or loads) the multi-asset return panel, computes the equal-weight /
    Markowitz / risk-parity baselines LIVE (pure numpy, TRAIN-only covariance) on the
    CONCATENATED purged walk-forward folds, serves the committed PPO policy from its
    ONNX artifact when present (onnxruntime, NO torch), reads the committed per-seed
    OOS metrics + seed lottery + PBO from ``artifacts/metrics.json``, runs the
    Diebold-Mariano test of the median-seed RL net return vs. the BEST baseline,
    derives the PURE ``rl_beats_baselines`` verdict, and assembles the equity-curve +
    weight-allocation Plotly figures. NEVER trains on the request path.

    THE HEADLINE OOS METRICS ARE COMPUTED FROM THE PURGED WALK-FORWARD — this function
    calls :func:`rlallocator.walk_forward.make_folds` and scores each baseline on the
    CONCATENATED OOS folds (each fold's covariance estimated on its own TRAIN block),
    NOT the full sample. A regression test asserts this wiring.

    Parameters
    ----------
    n_assets:
        Number of assets in the basket (capped at 20 by the router).
    n_seeds:
        Number of training seeds reflected in the seed lottery (capped at 16).
    cost_bps:
        Per-side turnover cost in basis points.
    lookback:
        Observation look-back window length (drives the walk-forward purge).
    rebalance:
        Rebalance cadence (``'daily'`` | ``'weekly'`` | ``'monthly'``).
    data_source_pref:
        ``"synthetic"`` (default) or ``"auto"`` for the real PIT path.
    seed:
        Master RNG seed for the synthetic panel.

    Returns
    -------
    RlAllocatorRun
        The summary and figures for the backend response.

    Raises
    ------
    ValidationError
        If the request is invalid (bad lookback / n_seeds / n_assets / cost).
    ArtifactError
        If the committed ONNX policy cannot be loaded.
    """
    _validate_request(n_assets=n_assets, n_seeds=n_seeds, cost_bps=cost_bps, lookback=lookback)

    returns, data_source = _resolve_returns(
        n_assets=n_assets, data_source_pref=data_source_pref, seed=seed
    )

    # WALK-FORWARD WIRED IN: the headline OOS metrics — for the BASELINES AND the RL
    # policy — come from the CONCATENATED purged walk-forward folds (each baseline's
    # covariance estimated on its own TRAIN block; the committed ONNX policy SERVED on
    # each fold's OOS block via onnxruntime, NO torch), NOT the full sample. This is the
    # rigorous served path. The inherently multi-seed quantities (seed lottery, DSR,
    # PBO) come from the committed offline ``metrics.json``; the request NEVER trains.
    folds = make_folds(returns.shape[0], lookback=lookback, n_folds=_N_WALK_FORWARD_FOLDS)
    baseline_nets = _walk_forward_baselines(returns, folds=folds, cost_bps=cost_bps)
    served_rl = _walk_forward_rl_policy(
        returns, folds=folds, lookback=lookback, n_assets=n_assets, cost_bps=cost_bps
    )

    committed = _read_committed_metrics()
    summary = _build_summary(
        baseline_nets=baseline_nets,
        served_rl=served_rl,
        committed=committed,
        n_seeds=n_seeds,
        data_source=data_source,
    )
    equity_fig, weights_fig = _build_figures(
        baseline_nets=baseline_nets,
        committed=committed,
        n_assets=n_assets,
    )
    return RlAllocatorRun(
        summary=summary,
        equity_figure=equity_fig,
        weights_figure=weights_fig,
    )


def _validate_request(*, n_assets: int, n_seeds: int, cost_bps: float, lookback: int) -> None:
    """Validate the request scalars (the router's field validators, enforced here too)."""
    if n_assets < 2 or n_assets > _MAX_ASSETS:
        raise ValidationError(f"n_assets must be in [2, {_MAX_ASSETS}], got {n_assets}.")
    if n_seeds < 1 or n_seeds > _MAX_SEEDS:
        raise ValidationError(f"n_seeds must be in [1, {_MAX_SEEDS}], got {n_seeds}.")
    if not math.isfinite(cost_bps) or cost_bps < 0.0:
        raise ValidationError(f"cost_bps must be finite and >= 0, got {cost_bps}.")
    if lookback < 1:
        raise ValidationError(f"lookback must be >= 1, got {lookback}.")


def _resolve_returns(*, n_assets: int, data_source_pref: str, seed: int) -> tuple[FloatArray, str]:
    """Build the multi-asset return panel + provenance (synthetic default; PIT path).

    The deployed default is the seeded factor-regime synthetic panel (no key /
    network). ``data_source_pref='auto'`` attempts the real cross-asset PIT bars and
    falls straight through to the synthetic panel on any failure.
    """
    if data_source_pref == "auto":
        from datetime import date

        from rlallocator.data.loaders import load_multi_asset_panel

        tickers = [f"ASSET{i}" for i in range(n_assets)]
        _, returns_frame, source = load_multi_asset_panel(
            tickers,
            start=date(2010, 1, 1),
            end=date(2020, 1, 1),
            data_source_pref="polygon",
            seed=seed,
        )
    else:
        from rlallocator.data.loaders import synthetic_default_panel

        _, returns_frame, source = synthetic_default_panel(
            n_obs=_MIN_N_OBS, n_assets=n_assets, seed=seed, kind="factor_regime"
        )
    return np.asarray(returns_frame.to_numpy(), dtype="float64"), str(source)


def _walk_forward_baselines(
    returns: FloatArray,
    *,
    folds: list[Fold],
    cost_bps: float,
) -> dict[str, FloatArray]:
    """Score each baseline on the CONCATENATED purged OOS folds (TRAIN-only covariance).

    For each fold, every baseline's weight vector is estimated on the fold's TRAIN
    block ONLY (no look-ahead) and held across the fold's OOS block; the per-fold OOS
    net-return series are CONCATENATED into one OOS series per baseline. This is the
    walk-forward-wired headline OOS metric the served path reports — NOT a full-sample
    fit. Pure numpy (no torch / onnxruntime).
    """
    from rlallocator.agents.baselines import run_baseline

    nets: dict[str, list[FloatArray]] = {name: [] for name in _BASELINE_NAMES}
    for fold in folds:
        train = returns[fold.train_start : fold.train_end]
        oos = returns[fold.test_start : fold.test_end]
        if train.shape[0] < 2 or oos.shape[0] < 2:
            continue
        for name in _BASELINE_NAMES:
            result = run_baseline(name, train, oos, cost_bps=cost_bps)
            nets[name].append(np.asarray(result.net_returns, dtype="float64"))
    concatenated: dict[str, FloatArray] = {}
    for name in _BASELINE_NAMES:
        concatenated[name] = (
            np.concatenate(nets[name]) if nets[name] else np.zeros(1, dtype="float64")
        )
    return concatenated


@dataclass(frozen=True, slots=True)
class _ServedRl:
    """The committed ONNX policy's walk-forward OOS net returns + applied weight path."""

    net_returns: FloatArray
    weights: FloatArray


def _walk_forward_rl_policy(
    returns: FloatArray,
    *,
    folds: list[Fold],
    lookback: int,
    n_assets: int,
    cost_bps: float,
) -> _ServedRl | None:
    """Serve the committed ONNX policy on the CONCATENATED purged OOS folds (no torch).

    When the committed ``artifacts/policy.onnx`` is present, the FROZEN policy is served
    through onnxruntime (NEVER torch) on each fold's OOS block and scored through the
    SHARED vectorized backtester — the RL-side analog of :func:`_walk_forward_baselines`,
    so the headline RL OOS metrics are walk-forward-computed on the served policy, not a
    full-sample fit. Returns the CONCATENATED per-bar RL net returns + applied weight
    path, or ``None`` when the policy artifact is absent / unservable (the honest-NULL
    placeholder then applies).
    """
    from rlallocator.agents.onnx_policy import OnnxPolicy, default_artifact_path
    from rlallocator.env.backtester import vectorized_backtest

    if not default_artifact_path().is_file():
        return None
    policy = OnnxPolicy()
    nets: list[FloatArray] = []
    weights: list[FloatArray] = []
    for fold in folds:
        oos = returns[fold.test_start : fold.test_end]
        if oos.shape[0] < lookback + 1:
            continue
        obs_matrix = _rl_oos_observations(oos, lookback=lookback, n_assets=n_assets)
        if obs_matrix.shape[0] == 0:  # pragma: no cover - guarded by the fold-size check
            continue
        try:
            decision_weights = policy.predict_weights(obs_matrix)
        except Exception:  # pragma: no cover - defensive: a signature mismatch degrades to None
            return None
        n_decisions = decision_weights.shape[0]
        panel_slice = oos[lookback - 1 : lookback - 1 + n_decisions + 1]
        weight_path = np.vstack([decision_weights, decision_weights[-1:]])
        result = vectorized_backtest(panel_slice, weight_path, cost_bps=cost_bps)
        nets.append(np.asarray(result.net_returns, dtype="float64"))
        weights.append(np.asarray(result.weights, dtype="float64"))
    if not nets:
        return None
    return _ServedRl(net_returns=np.concatenate(nets), weights=np.concatenate(weights))


def _rl_oos_observations(oos: FloatArray, *, lookback: int, n_assets: int) -> FloatArray:
    """Build the per-bar OOS observation matrix for the served policy (data <= t only).

    Each row is the flattened trailing look-back window of returns at decision bar
    ``t in [lookback-1, n_oos-2]`` concatenated with a flat (all-cash) weight vector —
    the served policy is stateless across bars and the simplex projection enforces a
    valid simplex every bar. The weights at ``t`` earn ``r_{t+1}`` (strictly causal).
    """
    oos = np.asarray(oos, dtype="float64")
    n_oos = oos.shape[0]
    flat = np.zeros(n_assets, dtype="float64")
    rows = [
        np.concatenate((oos[t - lookback + 1 : t + 1].ravel(), flat))
        for t in range(lookback - 1, n_oos - 1)
    ]
    if not rows:
        return np.zeros((0, lookback * n_assets + n_assets), dtype="float64")
    return np.asarray(rows, dtype="float64")


def _build_summary(
    *,
    baseline_nets: dict[str, FloatArray],
    served_rl: _ServedRl | None,
    committed: dict[str, Any],
    n_seeds: int,
    data_source: str,
) -> RlAllocatorSummary:
    """Assemble the JSON-safe summary + the PURE ``rl_beats_baselines`` verdict.

    The baseline OOS Sharpes are computed LIVE from the walk-forward folds. When the
    committed ONNX policy is present, the headline RL OOS Sharpe + the DM test vs. the
    best baseline are computed LIVE from the SAME walk-forward folds (the served policy,
    onnxruntime, NO torch); the inherently multi-seed quantities (seed band, DSR, PBO)
    come from the committed offline ``metrics.json``, and the PURE verdict is RE-DERIVED
    from the live DM + the committed DSR / seed-lo / PBO gates. Absent the committed
    policy/metrics (the bare-scaffold state), the RL side is the honest-NULL placeholder
    (``rl_beats_baselines=False``) — the served path never re-derives an edge from a
    live single path.
    """
    from rlallocator.evaluation.metrics import max_drawdown, oos_sharpe, turnover

    sharpes = {name: _safe_float(oos_sharpe(net)) for name, net in baseline_nets.items()}
    best_baseline = max(sharpes, key=lambda k: sharpes[k])

    seed_lo = _safe_float(committed.get("seed_sharpe_lo", 0.0))
    deflated = _safe_float(committed.get("deflated_sharpe", 0.0))
    pbo = _safe_float(committed.get("pbo", 1.0))
    n_trials = int(committed.get("n_effective_trials", max(1, n_seeds)))

    if served_rl is not None and served_rl.net_returns.size >= 2:
        # WALK-FORWARD RL HEADLINE: the served policy's OOS Sharpe + the DM vs. the best
        # baseline are computed LIVE on the concatenated purged folds (torch-free). The
        # PURE verdict is re-derived from this live DM + the committed DSR / seed / PBO.
        rl_net = served_rl.net_returns
        rl_sharpe = _safe_float(oos_sharpe(rl_net))
        dm_pvalue, rl_beats = _served_rl_verdict(
            rl_net=rl_net,
            best_net=baseline_nets[best_baseline],
            deflated=deflated,
            seed_lo=seed_lo,
            pbo=pbo,
            n_trials=n_trials,
        )
        rl_turnover = _safe_float(turnover(served_rl.weights))
        rl_mdd = _safe_float(max_drawdown(rl_net))
    else:
        # Honest-NULL placeholder (no committed policy): the RL side reads the committed
        # offline scalars (or zeros) and the verdict is False.
        rl_sharpe = _safe_float(committed.get("oos_sharpe_rl_median", 0.0))
        dm_pvalue = _safe_float(committed.get("dm_pvalue_vs_best", 1.0))
        rl_beats = bool(committed.get("rl_beats_baselines", False))
        rl_turnover = _safe_float(committed.get("turnover", 0.0))
        rl_mdd = _safe_float(committed.get("max_drawdown", 0.0))

    return RlAllocatorSummary(
        oos_sharpe_rl_median=rl_sharpe,
        oos_sharpe_1n=sharpes["equal_weight"],
        oos_sharpe_markowitz=sharpes["markowitz"],
        oos_sharpe_riskparity=sharpes["risk_parity"],
        best_baseline=best_baseline,
        seed_sharpe_lo=seed_lo,
        seed_sharpe_hi=_safe_float(committed.get("seed_sharpe_hi", 0.0)),
        dm_pvalue_vs_best=dm_pvalue,
        deflated_sharpe=deflated,
        pbo=pbo,
        turnover=rl_turnover,
        max_drawdown=rl_mdd,
        rl_beats_baselines=rl_beats,
        n_effective_trials=n_trials,
        data_source=data_source,
    )


def _served_rl_verdict(
    *,
    rl_net: FloatArray,
    best_net: FloatArray,
    deflated: float,
    seed_lo: float,
    pbo: float,
    n_trials: int,
) -> tuple[float, bool]:
    """Re-derive the PURE verdict from the live walk-forward DM + committed DSR/seed/PBO.

    Runs the Diebold-Mariano test of the served RL net return vs. the best baseline on
    the CONCATENATED purged OOS folds, then feeds that live DM (statistic + p-value)
    together with the committed-offline DSR, across-seed Sharpe lower bound, and PBO
    through the PURE :func:`rlallocator.evaluation.verdict.derive_verdict`. Returns
    ``(dm_pvalue, rl_beats_baselines)``. The verdict cannot read ``True`` unless all
    four gates clear — the same honest, leakage-free rule used offline.
    """
    from rlallocator.evaluation.diebold_mariano import diebold_mariano
    from rlallocator.evaluation.verdict import derive_verdict

    common = int(min(rl_net.size, best_net.size))
    if common < 2:  # pragma: no cover - guarded by the rl_net.size >= 2 check upstream
        return 1.0, False
    dm_statistic, dm_pvalue = diebold_mariano(rl_net[:common], best_net[:common])
    result = derive_verdict(
        dm_statistic,
        dm_pvalue,
        deflated,
        seed_lo,
        pbo,
        max(1, n_trials),
    )
    return _safe_float(dm_pvalue), bool(result.rl_beats_baselines)


def _build_figures(
    *,
    baseline_nets: dict[str, FloatArray],
    committed: dict[str, Any],
    n_assets: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the equity-curve + weight-allocation Plotly figures (lazy plotly).

    The equity figure overlays the RL median equity (committed precomputed curve when
    present) + the three baselines + the across-seed band; the weights figure shows
    the RL allocation-over-time area chart (committed weight path when present). Both
    are empty ``{}`` when their inputs are unavailable.
    """
    from rlallocator.env.backtester import equity_curve
    from rlallocator.plots import equity_curve_figure, weights_area_figure

    equity_fig: dict[str, Any] = {}
    rl_equity = committed.get("rl_median_equity")
    if rl_equity:
        baseline_equities = {
            name: equity_curve(np.asarray(net, dtype="float64")).tolist()
            for name, net in baseline_nets.items()
        }
        equity_fig = equity_curve_figure(
            rl_median_equity=np.asarray(rl_equity, dtype="float64"),
            baseline_equities=baseline_equities,
            seed_band_lo=committed.get("seed_band_lo"),
            seed_band_hi=committed.get("seed_band_hi"),
        )

    weights_fig: dict[str, Any] = {}
    weight_path = committed.get("rl_weight_path")
    if weight_path:
        weights_fig = weights_area_figure(np.asarray(weight_path, dtype="float64"))

    return equity_fig, weights_fig


def _read_committed_metrics() -> dict[str, Any]:
    """Load the committed ``artifacts/metrics.json`` (the offline seed-lottery / DSR / PBO).

    Returns an empty dict when the artifact has not been committed yet (the
    honest-NULL placeholder then applies) or is unreadable.
    """
    path = _ARTIFACTS_DIR / _METRICS_FILENAME
    if not path.is_file():
        return {}
    try:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # pragma: no cover - defensive against a corrupt file
        return {}
    return payload


def _safe_float(value: Any) -> float:
    """Coerce a scalar to a finite ``float`` (NaN/Inf -> 0.0) for JSON safety."""
    out = float(value)
    return out if math.isfinite(out) else 0.0
