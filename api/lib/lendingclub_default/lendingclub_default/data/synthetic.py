"""Synthetic LendingClub-schema data generator.

There is no real Kaggle LendingClub dump (or credentials) on this machine, so the
shipped demo model is trained on a *synthetic* panel that mimics the real LC
application-time schema. This module emits a deterministic pandas panel with:

- the real application-time columns (``loan_amnt``, ``term``, ``int_rate``,
  ``grade``, ``sub_grade``, ``emp_length``, ``home_ownership``, ``annual_inc``,
  ``dti``, ``fico_range_low/high``, ``revol_util``, ``open_acc``, ``pub_rec``,
  ``purpose``, ``addr_state``, ``installment``, ``verification_status``,
  ``issue_d``, ...);
- a realistic ``loan_status`` outcome column;
- the post-funding LEAKAGE columns (``recoveries``, ``total_pymnt*``,
  ``out_prncp*``, ``last_pymnt_*``, ``collection_recovery_fee``,
  ``total_rec_*``, ``debt_settlement_flag``, ``acc_now_delinq``,
  ``tot_coll_amt``, ``delinq_amnt``, ...) so the leakage allowlist is genuinely
  exercised end-to-end.

Design targets (so a leakage-free model lands at a *believable* AUC, not 0.5 and
not 0.99): base default rate ~15%; default probability a noisy monotone function
of ``fico``/``dti``/``int_rate``/``grade``; ``issue_d`` spread across 2015-2018
vintage cohorts with mild regime drift. All randomness flows from
:func:`lendingclub_default._rng.make_rng`, so a single seed reproduces the panel
byte-for-byte.

Importing this module has no side effects; data is only produced when
:func:`generate_synthetic_panel` is explicitly called.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from lendingclub_default._exceptions import ValidationError
from lendingclub_default._rng import make_rng

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

# quantcore-candidate: new code (synthetic LC-schema generator); seeded via _rng.

_GRADES: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G")
_PURPOSES: tuple[str, ...] = (
    "debt_consolidation",
    "credit_card",
    "home_improvement",
    "major_purchase",
    "small_business",
    "car",
    "medical",
    "house",
    "vacation",
    "moving",
    "other",
)
_STATES: tuple[str, ...] = (
    "CA",
    "NY",
    "TX",
    "FL",
    "IL",
    "PA",
    "OH",
    "GA",
    "NC",
    "MI",
    "NJ",
    "VA",
    "WA",
    "AZ",
    "MA",
)
_HOME: tuple[str, ...] = ("RENT", "MORTGAGE", "OWN")
_VERIFICATION: tuple[str, ...] = ("Verified", "Source Verified", "Not Verified")


@dataclass(frozen=True, slots=True)
class SyntheticConfig:
    """Immutable configuration for the synthetic LC panel generator.

    Attributes
    ----------
    n_loans:
        Number of loans (rows) to emit.
    base_default_rate:
        Target marginal default rate (fraction of resolved loans that default).
        The real LC accepted-loans rate is ~15%.
    vintages:
        Ordered ``issue_d`` cohort labels (``"YYYY-Qn"``) spread across the
        modelled period (2015-2018). Each loan is assigned to one.
    regime_drift:
        Per-vintage additive drift on the latent default log-odds, modelling mild
        macro regime change across cohorts (later vintages slightly riskier).
    signal_scale:
        Multiplier on the deterministic risk-driver block of the latent log-odds.
        Together with ``noise_scale`` it controls the signal-to-noise ratio, hence
        the achievable leakage-free AUC.
    noise_scale:
        Standard deviation of the idiosyncratic latent noise; tuned (with
        ``signal_scale``) so a leakage-free model achieves a believable AUC in
        ``[0.65, 0.72]`` rather than 0.5 (no signal) or 0.99 (leakage).
    seed:
        Master RNG seed (feeds :func:`lendingclub_default._rng.make_rng`).
    """

    n_loans: int = 20_000
    base_default_rate: float = 0.15
    vintages: tuple[str, ...] = (
        "2015-Q1",
        "2015-Q3",
        "2016-Q1",
        "2016-Q3",
        "2017-Q1",
        "2017-Q3",
        "2018-Q1",
        "2018-Q3",
    )
    regime_drift: float = 0.05
    signal_scale: float = 0.7
    noise_scale: float = 2.0
    seed: int = 20260616
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this config."""
        return asdict(self)


def _tune_intercept(latent: np.ndarray, target_rate: float) -> float:
    """Find the intercept ``b`` such that ``sigmoid(latent + b).mean() == target``.

    A monotone 1-D root-find by bisection on the marginal default rate. The latent
    array already carries every risk driver *except* the global intercept; shifting
    it by ``b`` only changes the marginal rate, monotonically, so bisection is exact
    to tolerance and fully deterministic (no RNG).
    """
    lo, hi = -20.0, 20.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        rate = float((1.0 / (1.0 + np.exp(-(latent + mid)))).mean())
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def generate_synthetic_panel(config: SyntheticConfig | None = None) -> pd.DataFrame:
    """Generate a deterministic synthetic LC-schema loan panel.

    The returned frame carries application-time columns, a ``loan_status``
    outcome, ``issue_d`` vintages, AND the post-funding leakage columns, so the
    full load -> drop-leakage -> label -> temporal-split pipeline is exercised on
    realistic data. The default probability is a noisy monotone function of
    ``fico``/``dti``/``int_rate``/``grade``; the marginal default rate matches
    ``config.base_default_rate``.

    Parameters
    ----------
    config:
        Generator configuration; defaults to :class:`SyntheticConfig` (a 20k-loan,
        ~15%-default, 2015-2018 panel).

    Returns
    -------
    pandas.DataFrame
        The synthetic panel, one row per loan.

    Raises
    ------
    ValidationError
        If ``config.n_loans`` is not positive or ``config.vintages`` is empty.
    """
    cfg = config if config is not None else SyntheticConfig()
    if cfg.n_loans <= 0:
        raise ValidationError(f"n_loans must be positive, got {cfg.n_loans}.")
    if len(cfg.vintages) == 0:
        raise ValidationError("config.vintages must be non-empty.")
    if not (0.0 < cfg.base_default_rate < 1.0):
        raise ValidationError(f"base_default_rate must be in (0, 1), got {cfg.base_default_rate}.")

    n = cfg.n_loans
    gen = make_rng(cfg.seed)
    vintages = cfg.vintages

    # --- vintage assignment (roughly balanced, oldest -> newest) ------------- #
    vintage_idx = gen.integers(0, len(vintages), size=n)
    issue_d = np.array([vintages[i] for i in vintage_idx], dtype=object)

    # --- grade (ordinal 0..6, A..G; worse grade is riskier) ------------------ #
    grade_ord = gen.integers(0, len(_GRADES), size=n)
    grade = np.array([_GRADES[i] for i in grade_ord], dtype=object)
    sub_n = gen.integers(1, 6, size=n)
    sub_grade = np.array(
        [f"{_GRADES[g]}{s}" for g, s in zip(grade_ord, sub_n, strict=True)],
        dtype=object,
    )

    # --- application-time numerics (loosely realistic LC ranges) ------------- #
    loan_amnt = gen.uniform(1_000, 35_000, size=n).round(-2)
    term = np.where(gen.random(n) < 0.7, "36 months", "60 months").astype(object)
    # int_rate is correlated with grade (LC prices risk into the coupon).
    int_rate = (5.0 + grade_ord * 3.0 + gen.normal(0, 1.5, size=n)).clip(5.0, 30.99)
    annual_inc = np.exp(gen.normal(11.0, 0.5, size=n)).clip(15_000, 300_000)
    dti = gen.uniform(0.0, 40.0, size=n)
    # FICO band: better FICO co-moves (mildly) with a better grade.
    fico_low = (710.0 - grade_ord * 10.0 + gen.normal(0, 25.0, size=n)).clip(660.0, 845.0).round()
    fico_high = fico_low + 4.0
    revol_util = gen.uniform(0.0, 100.0, size=n)
    open_acc = gen.integers(2, 30, size=n)
    pub_rec = gen.integers(0, 3, size=n)
    emp_length = gen.integers(0, 11, size=n)

    monthly_rate = int_rate / 100.0 / 12.0
    n_pay = np.where(term == "36 months", 36, 60)
    installment = (loan_amnt * monthly_rate / (1.0 - (1.0 + monthly_rate) ** (-n_pay))).round(2)

    home_ownership = np.array([_HOME[i] for i in gen.integers(0, len(_HOME), size=n)], dtype=object)
    purpose = np.array(
        [_PURPOSES[i] for i in gen.integers(0, len(_PURPOSES), size=n)], dtype=object
    )
    addr_state = np.array([_STATES[i] for i in gen.integers(0, len(_STATES), size=n)], dtype=object)
    verification_status = np.array(
        [_VERIFICATION[i] for i in gen.integers(0, len(_VERIFICATION), size=n)],
        dtype=object,
    )

    # --- latent default log-odds: noisy MONOTONE in the risk drivers --------- #
    # higher int_rate -> riskier; higher dti -> riskier; higher fico -> safer;
    # worse grade -> riskier; later vintage -> mild upward regime drift.
    drivers = (
        0.16 * (int_rate - 12.0)
        + 0.05 * (dti - 18.0)
        - 0.015 * (fico_low - 700.0)
        + 0.18 * grade_ord
        + cfg.regime_drift * vintage_idx
    )
    latent = cfg.signal_scale * drivers + gen.normal(0.0, cfg.noise_scale, size=n)
    # Tune the global intercept so the marginal default rate matches the target.
    intercept = _tune_intercept(np.asarray(latent, dtype="float64"), cfg.base_default_rate)
    p_default = 1.0 / (1.0 + np.exp(-(latent + intercept)))
    default = (gen.random(n) < p_default).astype(int)
    loan_status = np.where(default == 1, "Charged Off", "Fully Paid").astype(object)

    # --- post-funding LEAKAGE columns (correlated with the realised outcome) - #
    total_pymnt = np.where(default == 1, loan_amnt * 0.4, loan_amnt * 1.1)
    recoveries = np.where(default == 1, loan_amnt * 0.10, 0.0)
    out_prncp = np.where(default == 1, loan_amnt * 0.50, 0.0)
    # The three borrower-status fields LendingClub refreshes post-funding: they
    # track delinquency, so they leak on the real accepted.csv path.
    acc_now_delinq = np.where(default == 1, gen.integers(0, 3, size=n), 0).astype("int64")
    tot_coll_amt = np.where(default == 1, recoveries * gen.uniform(0.5, 2.0, n), 0.0)
    delinq_amnt = np.where(default == 1, out_prncp * gen.uniform(0.0, 0.3, n), 0.0)

    frame = pd.DataFrame(
        {
            # --- application-time features --------------------------------- #
            "loan_amnt": loan_amnt,
            "term": term,
            "int_rate": int_rate.round(2),
            "grade": grade,
            "sub_grade": sub_grade,
            "emp_length": emp_length,
            "home_ownership": home_ownership,
            "annual_inc": annual_inc.round(2),
            "dti": dti.round(2),
            "fico_range_low": fico_low,
            "fico_range_high": fico_high,
            "revol_util": revol_util.round(2),
            "open_acc": open_acc,
            "pub_rec": pub_rec,
            "purpose": purpose,
            "addr_state": addr_state,
            "installment": installment,
            "verification_status": verification_status,
            "issue_d": issue_d,
            # --- outcome --------------------------------------------------- #
            "loan_status": loan_status,
            # --- post-funding LEAKAGE columns ------------------------------ #
            "recoveries": recoveries.round(2),
            "total_pymnt": total_pymnt.round(2),
            "total_pymnt_inv": (total_pymnt * 0.99).round(2),
            "out_prncp": out_prncp.round(2),
            "out_prncp_inv": (out_prncp * 0.99).round(2),
            "last_pymnt_amnt": (installment * gen.uniform(0.5, 1.5, n)).round(2),
            "collection_recovery_fee": (recoveries * 0.10).round(2),
            "total_rec_prncp": (loan_amnt - out_prncp).round(2),
            "total_rec_int": (total_pymnt * 0.15).round(2),
            "debt_settlement_flag": np.where(default == 1, "Y", "N").astype(object),
            "funded_amnt": loan_amnt,
            # --- borrower-status fields refreshed post-funding (leak too) -- #
            "acc_now_delinq": acc_now_delinq,
            "tot_coll_amt": tot_coll_amt.round(2),
            "delinq_amnt": delinq_amnt.round(2),
        }
    )
    return frame
