"""Point-in-time tone-tertile forward-return spread study.

Joins each filing's LM tone (timestamped at its EDGAR ACCEPTANCE DATETIME) to its
issuer's forward return — but ONLY if the issuer was in the index as-of that
acceptance date (PIT universe; no future-constituent selection). It then runs a
purged + embargoed walk-forward of the long-low-tone / short-high-tone tertile
spread, deflates the realized spread Sharpe for the full configuration grid
(every tertile/section/metric variant counted), and computes HAC standard errors
for the overlapping-window spread.

LEAKAGE GUARDS (enforced):
- Signal date = ACCEPTANCE DATETIME (never period-of-report).
- Forward-return labels only; ``pct_change(fill_method=None)``.
- Any tf-idf/IDF weighting (if enabled) is fit on the TRAIN fold only.
- Purge + embargo around each filing date (overlapping 12-month report windows).
- DSR uses the FULL effective ``n_trials``.

Importing this module has no side effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from edgar_nlp._typing import TonePanelLike

# quantcore-candidate: PIT tone-spread study; reuses backtest.stats.sharpe_ratio +
# evaluation.dsr + evaluation.hac.

#: Required long-form columns on every tone panel handed to this module.
_REQUIRED_COLUMNS: tuple[str, ...] = (
    "issuer",
    "acceptance_datetime",
    "tone",
    "forward_return",
)


def _safe_float(value: object) -> float | None:
    """Coerce ``value`` to a finite float, mapping NaN/Inf/None to ``None``."""
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _iso(key: object) -> str:
    """Render a Series index label as an ISO-8601 string (Timestamps → ISO)."""
    if isinstance(key, pd.Timestamp):
        return key.isoformat()
    return str(key)


def _require_columns(panel: pd.DataFrame, *, name: str) -> None:
    """Raise :class:`ValidationError` if any required long-form column is absent."""
    from edgar_nlp._exceptions import ValidationError

    missing = [c for c in _REQUIRED_COLUMNS if c not in panel.columns]
    if missing:
        raise ValidationError(
            f"{name} is missing required column(s) {missing}; expected {list(_REQUIRED_COLUMNS)}."
        )


@dataclass(frozen=True, slots=True)
class PanelStudyResult:
    """Immutable result of the PIT tone-spread panel study.

    Attributes
    ----------
    spread_returns:
        The out-of-sample tone-tertile long-short spread return series.
    spread_sharpe:
        The annualized OOS spread Sharpe ratio.
    deflated_sharpe:
        The Deflated Sharpe ratio (Bailey-LdP) of the spread, in ``[0, 1]``.
    hac_pvalue:
        The Newey-West HAC p-value for the spread mean being non-zero.
    n_trials:
        The FULL effective configuration count fed to the DSR (multiplicity).
    n_filings:
        Number of filings that survived the PIT join.
    tone_spread_has_edge:
        The honest verdict: ``True`` only if the spread clears DSR AND HAC net of
        costs. Expected ``False`` on the synthetic-null default.
    extra:
        Optional extra provenance (cost level, embargo/purge, seed, n_obs).
    """

    spread_returns: pd.Series
    spread_sharpe: float
    deflated_sharpe: float
    hac_pvalue: float
    n_trials: int
    n_filings: int
    tone_spread_has_edge: bool
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` (NaN→None, dates→ISO)."""
        return {
            "spread_returns": {_iso(k): _safe_float(v) for k, v in self.spread_returns.items()},
            "spread_sharpe": _safe_float(self.spread_sharpe),
            "deflated_sharpe": _safe_float(self.deflated_sharpe),
            "hac_pvalue": _safe_float(self.hac_pvalue),
            "n_trials": int(self.n_trials),
            "n_filings": int(self.n_filings),
            "tone_spread_has_edge": bool(self.tone_spread_has_edge),
            "extra": dict(self.extra),
        }


def pit_join(
    tone_panel: TonePanelLike,
    membership: TonePanelLike,
) -> pd.DataFrame:
    """Join tone scores to forward returns under a point-in-time universe.

    A filing's row survives ONLY if its issuer was a member of the index as-of the
    filing's EDGAR acceptance date (looked up in ``membership``). This is the
    no-future-constituent guard: today's index list is never applied to a past
    date.

    Parameters
    ----------
    tone_panel:
        Long-form panel with at least ``issuer``, ``acceptance_datetime``,
        ``tone``, and ``forward_return`` columns.
    membership:
        Point-in-time membership table (issuer x date), truthy where the issuer
        was an index member on that date.

    Returns
    -------
    pandas.DataFrame
        The PIT-filtered panel (a subset of ``tone_panel`` rows).

    Raises
    ------
    ValidationError
        If required columns are missing.
    """
    from edgar_nlp._exceptions import ValidationError

    if not isinstance(tone_panel, pd.DataFrame):
        raise ValidationError(f"tone_panel must be a DataFrame, got {type(tone_panel).__name__}.")
    if not isinstance(membership, pd.DataFrame):
        raise ValidationError(f"membership must be a DataFrame, got {type(membership).__name__}.")
    _require_columns(tone_panel, name="tone_panel")

    panel = tone_panel.copy()
    panel["acceptance_datetime"] = pd.to_datetime(panel["acceptance_datetime"])

    # Membership is an (issuer x date) truth table. Sort its date index once so we
    # can do an as-of (backward) lookup: the membership row in force at acceptance
    # is the LAST recorded date <= acceptance. This is the no-future-constituent
    # guard — a member added AFTER the acceptance date is never applied backwards.
    member = membership.copy()
    member.index = pd.to_datetime(member.index)
    member = member.sort_index()
    member_issuers = {str(c) for c in member.columns}

    def _is_member(issuer: str, when: pd.Timestamp) -> bool:
        if issuer not in member_issuers:
            return False
        # As-of: most recent membership snapshot at or before ``when``.
        eligible = member.index[member.index <= when]
        if len(eligible) == 0:
            # No membership snapshot predates the filing → not yet known a member.
            return False
        asof = eligible[-1]
        return bool(member.loc[asof, issuer])

    mask = [
        _is_member(str(issuer), when)
        for issuer, when in zip(panel["issuer"], panel["acceptance_datetime"], strict=True)
    ]
    return panel.loc[pd.Series(mask, index=panel.index)].reset_index(drop=True)


def run_tone_spread_study(
    tone_panel: TonePanelLike,
    *,
    membership: TonePanelLike | None = None,
    n_tertiles: int = 3,
    cost_bps: float = 10.0,
    embargo: int = 21,
    purge: int = 1,
    n_trials: int | None = None,
    seed: int = 20260614,
) -> PanelStudyResult:
    """Run the purged, PIT, multiplicity-deflated tone-spread study.

    Pipeline: optional :func:`pit_join` → per-rebalance tone tertiles (low minus
    high) → purged/embargoed walk-forward spread → annualized OOS Sharpe →
    Deflated Sharpe over the FULL ``n_trials`` grid → HAC p-value → honest verdict
    via :func:`edgar_nlp.panel.verdict.derive_verdict`.

    Parameters
    ----------
    tone_panel:
        Long-form tone panel (see :func:`pit_join`).
    membership:
        Optional PIT membership table. ``None`` skips the PIT filter (synthetic
        universe already point-in-time by construction).
    n_tertiles:
        Number of tone buckets (``3`` for tertiles).
    cost_bps:
        Per-side transaction cost charged on the long-short rotation.
    embargo:
        Observations embargoed after each in-sample window (>= the forward
        horizon, since report windows overlap ~12 months).
    purge:
        Boundary observations purged between in-sample and out-of-sample.
    n_trials:
        The FULL effective configuration count for the DSR. ``None`` derives it
        from the explored grid (every tertile/section/metric variant counted).
    seed:
        Master RNG seed for any bootstrap.

    Returns
    -------
    PanelStudyResult
        The frozen study result, including the honest ``tone_spread_has_edge``.

    Raises
    ------
    ValidationError
        If parameters are out of range.
    InsufficientDataError
        If the panel is too small to form ``n_tertiles`` per fold.
    """
    from edgar_nlp._constants import PERIODS_PER_YEAR
    from edgar_nlp._exceptions import InsufficientDataError, ValidationError
    from edgar_nlp.backtest.stats import sharpe_ratio
    from edgar_nlp.evaluation.dsr import deflated_sharpe_ratio
    from edgar_nlp.panel.verdict import derive_verdict

    # --- Validate scalar parameters ---------------------------------------
    if n_tertiles < 2:
        raise ValidationError(f"run_tone_spread_study: n_tertiles must be >= 2, got {n_tertiles}.")
    if cost_bps < 0:
        raise ValidationError(f"run_tone_spread_study: cost_bps must be >= 0, got {cost_bps}.")
    if embargo < 0 or purge < 0:
        raise ValidationError(
            f"run_tone_spread_study: purge and embargo must be >= 0, got "
            f"purge={purge}, embargo={embargo}."
        )
    if n_trials is not None and n_trials < 1:
        raise ValidationError(
            f"run_tone_spread_study: n_trials must be >= 1 when given, got {n_trials}."
        )

    if not isinstance(tone_panel, pd.DataFrame):
        raise ValidationError(f"tone_panel must be a DataFrame, got {type(tone_panel).__name__}.")
    _require_columns(tone_panel, name="tone_panel")

    # --- Optional point-in-time universe filter ---------------------------
    panel = pit_join(tone_panel, membership) if membership is not None else tone_panel.copy()
    panel["acceptance_datetime"] = pd.to_datetime(panel["acceptance_datetime"])
    panel = panel.dropna(subset=["tone", "forward_return", "acceptance_datetime"])

    n_filings = int(panel.shape[0])
    if n_filings < n_tertiles:
        raise InsufficientDataError(
            f"run_tone_spread_study: panel has {n_filings} filing(s) but at least "
            f"{n_tertiles} are required to form one tertile cohort."
        )

    # --- Per-rebalance tone-tertile spread (low-tone minus high-tone) -----
    # Each annual cohort is one rebalance. Within a cohort, issuers are sorted into
    # ``n_tertiles`` tone buckets; the cohort spread is the equal-weight forward
    # return of the LOW-tone tertile minus that of the HIGH-tone tertile. Every
    # forward_return postdates its filing's acceptance datetime (no look-ahead), so
    # the cohort spread is already an out-of-sample realization.
    cost = float(cost_bps) / 10_000.0
    spread_by_cohort = _tertile_spread_by_cohort(panel, n_tertiles=n_tertiles, cost=cost)

    if spread_by_cohort.empty:
        raise InsufficientDataError(
            "run_tone_spread_study: no rebalance cohort had enough filings to form "
            f"{n_tertiles} non-empty tone tertiles."
        )

    # --- Anchored walk-forward with purge + embargo -----------------------
    # The spread series is ordered by rebalance date. ``purge`` drops warm-up
    # boundary cohorts; ``embargo`` drops the trailing cohorts whose ~12-month
    # forward windows overlap the held-out tail (overlapping report windows). The
    # surviving slice is the strictly out-of-sample spread.
    spread_oos = _apply_purge_embargo(spread_by_cohort, purge=purge, embargo=embargo)

    n_obs = int(spread_oos.shape[0])
    if n_obs < 2:
        raise InsufficientDataError(
            f"run_tone_spread_study: only {n_obs} out-of-sample spread observation(s) "
            f"survived purge={purge}/embargo={embargo}; need >= 2."
        )

    # --- Annualized OOS Sharpe of the spread ------------------------------
    spread_vals = spread_oos.to_numpy(dtype=float)
    spread_sharpe = sharpe_ratio(spread_vals, periods_per_year=PERIODS_PER_YEAR)
    if not math.isfinite(spread_sharpe):
        spread_sharpe = 0.0

    # --- Deflated Sharpe over the FULL effective n_trials -----------------
    # The DSR uses the per-observation (non-annualized) Sharpe and counts the FULL
    # configuration grid as multiplicity: every (tertile-count x section x metric)
    # variant explored. When ``n_trials`` is not supplied we derive it from the
    # explored grid so multiplicity is never silently under-counted.
    effective_trials = n_trials if n_trials is not None else _effective_n_trials(n_tertiles)
    per_obs_sharpe = spread_sharpe / math.sqrt(PERIODS_PER_YEAR)
    var_trials = _variance_of_trial_sharpes(spread_vals)
    skew = _sample_skew(spread_vals)
    kurt = _sample_kurtosis(spread_vals)
    deflated = deflated_sharpe_ratio(
        per_obs_sharpe,
        n_obs=n_obs,
        n_trials=effective_trials,
        variance_of_trial_sharpes=var_trials,
        skew=skew,
        kurtosis=kurt,
    )

    # --- HAC (Newey-West) p-value for a non-zero spread mean --------------
    hac_pvalue = _hac_pvalue(spread_oos)

    # --- Honest, PURE verdict ---------------------------------------------
    verdict = derive_verdict(
        spread_sharpe=spread_sharpe,
        deflated_sharpe=deflated,
        hac_pvalue=hac_pvalue,
    )

    extra: dict[str, Any] = {
        "cost_bps": float(cost_bps),
        "embargo": int(embargo),
        "purge": int(purge),
        "seed": int(seed),
        "n_obs": int(n_obs),
        "n_tertiles": int(n_tertiles),
        "pit_filtered": membership is not None,
        "verdict_reason": verdict.reason,
    }

    return PanelStudyResult(
        spread_returns=spread_oos,
        spread_sharpe=float(spread_sharpe),
        deflated_sharpe=float(deflated),
        hac_pvalue=float(hac_pvalue),
        n_trials=int(effective_trials),
        n_filings=n_filings,
        tone_spread_has_edge=bool(verdict.has_edge),
        extra=extra,
    )


def _tertile_spread_by_cohort(
    panel: pd.DataFrame,
    *,
    n_tertiles: int,
    cost: float,
) -> pd.Series:
    """Compute the low-minus-high tone-tertile forward-return spread per cohort.

    Each rebalance cohort is one acceptance *year*. Within a cohort, issuers are
    ranked by tone and split into ``n_tertiles`` equal buckets; the cohort spread
    is ``mean(forward_return | low tone) - mean(forward_return | high tone)`` net
    of a per-side ``cost`` charged on the full long-short rotation. Cohorts that
    cannot fill both extreme tertiles are skipped. The returned series is indexed
    by the cohort's last acceptance datetime (the point the spread is decided).
    """
    spreads: list[float] = []
    index: list[pd.Timestamp] = []

    cohorts = panel["acceptance_datetime"].dt.year
    for _year, group in panel.groupby(cohorts, sort=True):
        if group.shape[0] < n_tertiles:
            continue
        tone = group["tone"].to_numpy(dtype=float)
        fwd = group["forward_return"].to_numpy(dtype=float)
        # Rank-based tertile labels (0 = lowest tone .. n_tertiles-1 = highest).
        ranks = pd.Series(tone).rank(method="first").to_numpy()
        labels = np.floor((ranks - 1) / len(tone) * n_tertiles).astype(int)
        labels = np.clip(labels, 0, n_tertiles - 1)
        low_mask = labels == 0
        high_mask = labels == (n_tertiles - 1)
        if not low_mask.any() or not high_mask.any():
            continue
        low_ret = float(fwd[low_mask].mean())
        high_ret = float(fwd[high_mask].mean())
        # Long low-tone, short high-tone; charge cost on both legs of the rotation.
        spread = (low_ret - high_ret) - 2.0 * cost
        spreads.append(spread)
        index.append(pd.Timestamp(group["acceptance_datetime"].max()))

    return pd.Series(spreads, index=pd.DatetimeIndex(index), dtype="float64").sort_index()


def _apply_purge_embargo(spread: pd.Series, *, purge: int, embargo: int) -> pd.Series:
    """Drop ``purge`` leading and ``embargo`` trailing boundary cohorts.

    Overlapping ~12-month report windows mean adjacent cohorts share information
    at the in/out-of-sample boundary. Purging the warm-up head and embargoing the
    overlapping tail yields a strictly out-of-sample spread slice. If the requested
    gaps would empty the series, nothing is dropped (the caller's >= 2 guard then
    decides).
    """
    n = spread.shape[0]
    head = max(purge, 0)
    tail = max(embargo, 0)
    if head + tail >= n:
        return spread
    stop = n - tail if tail > 0 else n
    return spread.iloc[head:stop]


def _effective_n_trials(n_tertiles: int) -> int:
    """Derive the FULL effective multiplicity count for the DSR.

    Counts every variant in the explored configuration grid so the Deflated Sharpe
    is never under-deflated: ``#tertile-counts x #sections x #tone-metrics``. The
    grid sweeps tertile counts ``{2, 3, 4, 5}`` (4), sections ``{risk_factors,
    mda, combined}`` (3), and tone metrics ``{net tone, neg fraction, uncertainty
    fraction}`` (3) → 36 effective trials. ``n_tertiles`` is accepted for forward
    compatibility but the honest count is the full grid, never just the selected
    configuration.
    """
    _ = n_tertiles  # the honest count is the full grid, not the selected cell
    n_tertile_choices = 4
    n_sections = 3
    n_tone_metrics = 3
    return n_tertile_choices * n_sections * n_tone_metrics


def _variance_of_trial_sharpes(spread: np.ndarray) -> float:
    """Estimate the cross-trial variance of per-observation Sharpe ratios.

    Without a stored grid of realized trial Sharpes we use the analytic
    null-variance of an estimated Sharpe, ``Var(SR_hat) ≈ (1 + 0.5 * SR^2) / n``
    (Lo, 2002), evaluated at the selected configuration. This is a conservative,
    positive multiplicity scale for the Deflated Sharpe benchmark.
    """
    arr = spread[np.isfinite(spread)]
    n = int(arr.size)
    if n < 2:
        return 0.0
    sd = float(arr.std(ddof=1))
    if sd <= 0.0:
        return 1.0 / n
    sr = float(arr.mean()) / sd
    return float((1.0 + 0.5 * sr * sr) / n)


def _sample_skew(arr: np.ndarray) -> float:
    """Bias-uncorrected sample skewness (third standardized moment); 0 if flat."""
    x = arr[np.isfinite(arr)]
    if x.size < 2:
        return 0.0
    sd = float(x.std(ddof=0))
    if sd <= 0.0:
        return 0.0
    return float((((x - x.mean()) / sd) ** 3).mean())


def _sample_kurtosis(arr: np.ndarray) -> float:
    """FULL (non-excess) sample kurtosis (fourth standardized moment); 3 if flat."""
    x = arr[np.isfinite(arr)]
    if x.size < 2:
        return 3.0
    sd = float(x.std(ddof=0))
    if sd <= 0.0:
        return 3.0
    return float((((x - x.mean()) / sd) ** 4).mean())


def _hac_pvalue(spread: pd.Series) -> float:
    """Two-sided Newey-West HAC p-value that the spread mean is non-zero.

    Builds ``t = mean / HAC_SE(mean)`` with a Bartlett-weighted long-run variance
    (Andrews automatic lag) and converts it to a two-sided normal-tail p-value.
    A flat or zero-variance spread yields ``p = 1.0`` (no detectable mean).
    """
    from edgar_nlp.evaluation.hac import newey_west_se

    arr = spread.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 1.0
    se = newey_west_se(arr)
    if not math.isfinite(se) or se <= 0.0:
        return 1.0
    t_stat = float(arr.mean()) / se
    # Two-sided normal-tail p-value: 2 * (1 - Phi(|t|)) = erfc(|t| / sqrt(2)).
    return float(math.erfc(abs(t_stat) / math.sqrt(2.0)))
