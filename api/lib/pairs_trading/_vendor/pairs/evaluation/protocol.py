"""End-to-end evaluation protocol.

:class:`EvaluationProtocol` orchestrates the entire credibility pipeline:
walk-forward, CPCV, deflated Sharpe ratio, PBO, Memmel, Hansen SPA,
HAC standard errors and stationary-bootstrap confidence intervals. The
:class:`~pairs.evaluation.trial_log.TrialLog` is consulted before any
work begins so that the OOS window cannot be reused for the same
``spec_hash`` without an explicit bump.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pairs._exceptions import InputError, OOSReuseError
from pairs._rng import default_rng, derive_rng

from .cpcv import cpcv_paths
from .dsr import deflated_sharpe_ratio, effective_n_trials
from .hac import newey_west_se
from .pbo import pbo_cscv
from .results import (
    CPCVResult,
    DSRResult,
    MemmelResult,
    PBOResult,
    ProtocolReport,
    SPAResult,
    WalkForwardResult,
)
from .spa import hansen_spa
from .trial_log import TrialLog
from .walk_forward import walk_forward_anchored

__all__ = ["EvaluationProtocol"]

PairSelector = Callable[[pd.DataFrame], Any]
PairBacktester = Callable[[pd.DataFrame, Any], pd.Series]


def _sharpe(returns: pd.Series) -> float:
    finite = returns.dropna()
    if finite.size < 2:
        return float("nan")
    std = float(finite.std(ddof=1))
    if std <= 0.0:
        return float("nan")
    return float(finite.mean() / std)


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationProtocol:
    """Configuration object that assembles a :class:`ProtocolReport`."""

    train_min_years: float
    purge_days: int
    embargo_pct: float
    trial_log: TrialLog
    rng_seed: int
    oos_fraction: float = 0.25
    n_groups: int = 10
    k_test: int = 2
    s_partitions: int = 16
    bootstrap_replicates: int = 1000

    def __post_init__(self) -> None:
        if not (0.0 < self.oos_fraction < 1.0):
            raise InputError("oos_fraction must lie in (0, 1)")
        if self.train_min_years <= 0:
            raise InputError("train_min_years must be positive")
        if self.purge_days < 0:
            raise InputError("purge_days must be non-negative")
        if not (0.0 <= self.embargo_pct < 1.0):
            raise InputError("embargo_pct must lie in [0, 1)")
        if self.s_partitions <= 0 or self.s_partitions % 2 != 0:
            raise InputError("s_partitions must be a positive even integer")
        if not isinstance(self.trial_log, TrialLog):
            raise InputError("trial_log must be a TrialLog instance")

    def run(
        self,
        prices: pd.DataFrame,
        pair_selector: PairSelector,
        pair_backtester: PairBacktester,
        *,
        spec_hash: str,
        trial_returns: pd.DataFrame | None = None,
        benchmark_returns: pd.Series | None = None,
    ) -> ProtocolReport:
        """Run the full evaluation pipeline for one strategy specification.

        Parameters
        ----------
        prices : pandas.DataFrame
            Wide price panel.
        pair_selector, pair_backtester : callable
            Strategy primitives. See
            :func:`pairs.evaluation.walk_forward_anchored`.
        spec_hash : str
            Stable hash of the strategy specification. Reusing the same
            hash on the same trial log raises :class:`OOSReuseError`.
        trial_returns : pandas.DataFrame, optional
            Returns of competing trial configurations used to estimate
            the effective number of independent trials and PBO. When
            ``None`` the corresponding fields are left empty.
        benchmark_returns : pandas.Series, optional
            Benchmark for Hansen's SPA. Required for the SPA section;
            otherwise omitted.

        Returns
        -------
        ProtocolReport
            Frozen aggregate report ready for serialisation.
        """
        if not spec_hash:
            raise InputError("spec_hash must be a non-empty string")
        if self.trial_log.count_for_hash(spec_hash) > 0:
            raise OOSReuseError(
                f"trial log already contains an entry for spec_hash={spec_hash!r}; "
                "bump the spec or use a new trial log to avoid OOS reuse"
            )
        trial_id = self.trial_log.start_trial(spec_hash)
        parent_rng = default_rng(self.rng_seed)
        wf_rng = derive_rng(parent_rng, "walk_forward")
        cpcv_rng = derive_rng(parent_rng, "cpcv")
        spa_rng = derive_rng(parent_rng, "spa")

        wf: WalkForwardResult = walk_forward_anchored(
            prices,
            train_min_years=self.train_min_years,
            purge_days=self.purge_days,
            embargo_pct=self.embargo_pct,
            pair_selector=pair_selector,
            pair_backtester=pair_backtester,
            rng=wf_rng,
            bootstrap_replicates=self.bootstrap_replicates,
        )
        cpcv: CPCVResult = cpcv_paths(
            prices,
            n_groups=self.n_groups,
            k_test=self.k_test,
            purge_days=self.purge_days,
            embargo_pct=self.embargo_pct,
            pair_selector=pair_selector,
            pair_backtester=pair_backtester,
            rng=cpcv_rng,
        )

        oos = wf.oos_returns
        sample_size = max(int(oos.size), 2)
        realized_sr = _sharpe(oos) if not oos.empty else float("nan")
        if trial_returns is not None and trial_returns.shape[1] >= 1:
            n_eff = float(effective_n_trials(trial_returns))
            sr_variance = float(np.nanvar(_per_column_sharpe(trial_returns), ddof=0))
            if sr_variance <= 0.0:
                sr_variance = 1e-6
        else:
            n_eff = 1.0
            sr_variance = 1e-6

        dsr: DSRResult = deflated_sharpe_ratio(
            realized_sr=realized_sr if np.isfinite(realized_sr) else 0.0,
            n_trials_eff=n_eff,
            sr_trial_variance=sr_variance,
            sample_size=sample_size,
        )

        pbo: PBOResult | None
        if (
            trial_returns is not None
            and trial_returns.shape[1] >= 2
            and trial_returns.shape[0] >= self.s_partitions
        ):
            pbo = pbo_cscv(trial_returns, s=self.s_partitions)
        else:
            pbo = None

        memmel: MemmelResult | None = None
        spa: SPAResult | None = None
        if benchmark_returns is not None and oos.size >= 8:
            from .memmel import memmel_test as _memmel  # local import to avoid cycle

            aligned_bench = benchmark_returns.reindex(oos.index).dropna()
            common = oos.reindex(aligned_bench.index).dropna()
            aligned_bench = aligned_bench.loc[common.index]
            if common.size >= 8 and aligned_bench.std(ddof=1) > 0.0 and common.std(ddof=1) > 0.0:
                memmel = _memmel(common, aligned_bench)
            if trial_returns is not None and trial_returns.shape[1] >= 1:
                aligned_trials = trial_returns.reindex(aligned_bench.index).dropna(how="any")
                if aligned_trials.shape[0] >= 8:
                    spa = hansen_spa(
                        aligned_trials,
                        aligned_bench.loc[aligned_trials.index],
                        rng=spa_rng,
                    )

        hac = newey_west_se(oos) if oos.size >= 4 and oos.std(ddof=1) > 0.0 else float("nan")
        is_oos_decay_pct = float("nan")
        if cpcv.path_sharpes:
            finite_paths = [s for s in cpcv.path_sharpes if np.isfinite(s)]
            if finite_paths and np.isfinite(wf.oos_sharpe) and wf.oos_sharpe != 0.0:
                is_oos_decay_pct = float(1.0 - (np.median(finite_paths) / wf.oos_sharpe))

        broken_pair_count = 0
        if hasattr(pair_selector, "broken_pair_count"):
            try:
                broken_pair_count = int(pair_selector.broken_pair_count)
            except Exception:  # pragma: no cover - defensive
                broken_pair_count = 0

        report = ProtocolReport(
            walk_forward=wf,
            cpcv=cpcv,
            dsr=dsr,
            pbo=pbo,
            memmel=memmel,
            spa=spa,
            hac_se=float(hac) if np.isfinite(hac) else 0.0,
            is_oos_decay_pct=float(is_oos_decay_pct)
            if np.isfinite(is_oos_decay_pct)
            else float("nan"),
            broken_pair_count=int(broken_pair_count),
            spec_hash=spec_hash,
            trial_id=int(trial_id),
            seed=int(self.rng_seed),
        )
        self.trial_log.record_result(
            trial_id,
            metrics={
                "oos_sharpe": float(wf.oos_sharpe) if np.isfinite(wf.oos_sharpe) else None,
                "dsr": float(dsr.dsr),
                "pbo": float(pbo.pbo) if pbo is not None else None,
                "hac_se": float(report.hac_se),
            },
            spec_hash=spec_hash,
        )
        return report


def _per_column_sharpe(df: pd.DataFrame) -> np.ndarray:
    arr = df.to_numpy(dtype=float, copy=True)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0, ddof=1)
    out = np.zeros_like(mean)
    mask = std > 0.0
    out[mask] = mean[mask] / std[mask]
    return out
