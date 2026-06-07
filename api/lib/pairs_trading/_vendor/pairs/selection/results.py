"""Immutable result containers for the selection sub-package.

The selection pipeline emits three frozen dataclasses:

* :class:`Candidate` -- a single pair under consideration, optionally annotated
  with sector taxonomy and liquidity metrics. The ``exclusion_reason`` tuple is
  populated by the pre-screen filter chain when a candidate is dropped, so
  callers can audit *why* a pair did not survive without re-running the
  filters.
* :class:`ScreenResult` -- aggregate cointegration screening output for a batch
  of candidates. Carries the per-pair diagnostic frame, the raw
  :class:`~pairs.cointegration.results.CointegrationResult` map, and the
  multiple-testing method that was applied.
* :class:`RWResult` -- output of the Romano-Wolf step-down procedure used to
  control family-wise error rate over a panel of out-of-sample Sharpe
  comparisons.

All dataclasses use ``frozen=True``, ``slots=True``, and ``kw_only=True`` so
that they are hashable-by-id, memory compact, and constructed unambiguously.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["Candidate", "RWResult", "ScreenResult"]


@dataclass(frozen=True, slots=True, kw_only=True)
class Candidate:
    """A single pair under consideration.

    Parameters
    ----------
    ticker_a, ticker_b
        Uppercase ticker symbols. Order is preserved verbatim from the
        upstream universe so the caller controls which leg is the
        "dependent" variable in subsequent regressions.
    sector, industry, sub_industry
        Optional GICS taxonomy. ``None`` means the taxonomy was not supplied
        for this candidate (typical for the v1 curated mode that does not
        carry sector metadata).
    adv_a, adv_b
        Optional 20-day average dollar volume for each leg, expressed in
        whole dollars. Used by the liquidity floor in
        :func:`pairs.selection.apply_pre_screen`.
    exclusion_reason
        Reasons accumulated by the pre-screen filter chain. An empty tuple
        means the candidate has not been screened or has passed every check.
        Filters append reason codes in the order they were evaluated so the
        last entry reflects the immediate cause of rejection.
    """

    ticker_a: str
    ticker_b: str
    sector: str | None = None
    industry: str | None = None
    sub_industry: str | None = None
    adv_a: float | None = None
    adv_b: float | None = None
    exclusion_reason: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.ticker_a or not self.ticker_b:
            msg = "ticker_a and ticker_b must be non-empty strings"
            raise ValueError(msg)
        if self.ticker_a == self.ticker_b:
            msg = f"self-pair is not a valid candidate: {self.ticker_a}"
            raise ValueError(msg)

    @property
    def pair_id(self) -> str:
        """Stable identifier ``"A__B"`` formed from the (uppercased) legs."""
        return f"{self.ticker_a.upper()}__{self.ticker_b.upper()}"

    def with_reason(self, reason: str) -> Candidate:
        """Return a copy with ``reason`` appended to :attr:`exclusion_reason`."""
        return Candidate(
            ticker_a=self.ticker_a,
            ticker_b=self.ticker_b,
            sector=self.sector,
            industry=self.industry,
            sub_industry=self.sub_industry,
            adv_a=self.adv_a,
            adv_b=self.adv_b,
            exclusion_reason=(*self.exclusion_reason, reason),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ScreenResult:
    """Aggregate output of a cointegration screening pass.

    Parameters
    ----------
    diagnostics
        One row per screened candidate with columns
        ``[pair_id, ticker_a, ticker_b, p_raw, hedge_ratio, half_life,
        q_value, survives_mtc]``.
    coint_results
        Mapping from ``pair_id`` to the corresponding
        :class:`~pairs.cointegration.results.CointegrationResult` (or
        :class:`~pairs.cointegration.results.PipelineResult` -- duck-typed
        so callers can inspect either layer).
    method
        Multiple-testing correction method that was applied
        (e.g. ``"fdr_bh"`` or ``"none"``).
    alpha
        Family-wise / FDR target threshold supplied to the screen.
    asof
        Timestamp marking the upper bound of the formation window. Useful
        for downstream provenance.
    """

    diagnostics: pd.DataFrame
    coint_results: dict[str, Any]
    method: str
    alpha: float
    asof: pd.Timestamp

    def __post_init__(self) -> None:
        if not isinstance(self.diagnostics, pd.DataFrame):
            msg = "diagnostics must be a pandas DataFrame"
            raise TypeError(msg)
        if not (0.0 < self.alpha < 1.0):
            msg = f"alpha must lie in (0, 1); got {self.alpha}"
            raise ValueError(msg)
        if not isinstance(self.asof, pd.Timestamp):
            msg = "asof must be a pandas Timestamp"
            raise TypeError(msg)

    @property
    def surviving_pairs(self) -> list[tuple[str, str]]:
        """Return the ``(ticker_a, ticker_b)`` tuples that survived the screen.

        Surviving means ``survives_mtc`` is true on the diagnostic row.
        Empty-frame inputs return an empty list rather than raising.
        """
        if self.diagnostics.empty or "survives_mtc" not in self.diagnostics.columns:
            return []
        mask = self.diagnostics["survives_mtc"].astype(bool)
        keep = self.diagnostics.loc[mask, ["ticker_a", "ticker_b"]]
        return [(str(row.ticker_a), str(row.ticker_b)) for row in keep.itertuples(index=False)]


@dataclass(frozen=True, slots=True, kw_only=True)
class RWResult:
    """Outcome of the Romano-Wolf step-down multiple-testing procedure.

    Parameters
    ----------
    surviving_pair_ids
        Pair identifiers whose null hypothesis was rejected at level
        :attr:`alpha` after step-down adjustment.
    adjusted_pvalues
        Step-down adjusted p-values, indexed by pair id. Values are
        monotone non-decreasing in the order they were rejected and lie
        in ``[0, 1]``.
    null_distribution
        ``(n_boot,)`` array of bootstrap maxima used to compute the
        adjusted p-values for the first (largest) statistic.
    block_length
        Stationary-bootstrap block length used to construct
        :attr:`null_distribution`. Always positive.
    n_boot
        Number of bootstrap draws.
    """

    surviving_pair_ids: list[str] = field(default_factory=list)
    adjusted_pvalues: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    null_distribution: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    block_length: int = 1
    n_boot: int = 0

    def __post_init__(self) -> None:
        if self.block_length <= 0:
            msg = f"block_length must be positive; got {self.block_length}"
            raise ValueError(msg)
        if self.n_boot < 0:
            msg = f"n_boot must be non-negative; got {self.n_boot}"
            raise ValueError(msg)
        if not isinstance(self.adjusted_pvalues, pd.Series):
            msg = "adjusted_pvalues must be a pandas Series"
            raise TypeError(msg)
        if len(self.adjusted_pvalues):
            vals = self.adjusted_pvalues.to_numpy()
            if not np.all((vals >= 0.0) & (vals <= 1.0)):
                msg = "adjusted_pvalues must lie in [0, 1]"
                raise ValueError(msg)
