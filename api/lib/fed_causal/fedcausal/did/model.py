"""Difference-in-differences on rate-sensitive vs. control names.

The DiD design contrasts the abnormal-return response of **rate-sensitive**
("treated") names with **rate-insensitive** ("control") names around **hawkish**
vs. **dovish** surprises. The interaction coefficient ``treated x post`` measures
how much MORE rate-sensitive names move than controls, conditional on the
surprise sign.

Standard errors are CLUSTERED (by event, optionally two-way by event and name) so
the inference is robust to within-event cross-sectional correlation and serial
correlation across events.

FRAMING (the honest deliverable): the DiD coefficient is reported as DESCRIPTIVE
cross-sectional HETEROGENEITY — rate-sensitive names mechanically move more — NOT
as a tradable profit claim. Whether that heterogeneity translates into a
net-of-cost tradable spread is decided (negatively, by default) by
:mod:`fedcausal.evaluation.verdict`.

The clustered covariance is the textbook (Cameron-Gelbach-Miller / White)
sandwich estimator with the standard finite-sample correction
``c = (G / (G - 1)) * ((N - 1) / (N - K))``; the one-way ``"event"`` estimator
matches ``statsmodels`` ``cov_type="cluster"`` to machine precision (parity-tested
to 1e-10). It is implemented in pure numpy so importing this module never pulls in
statsmodels.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats

from fedcausal._constants import DEFAULT_ALPHA, EPS
from fedcausal._exceptions import InsufficientDataError, ValidationError
from fedcausal._validation import validate_alpha

if TYPE_CHECKING:
    from fedcausal._typing import SurpriseLabel

#: The fixed column layout of the DiD regression panel produced by
#: :func:`build_did_panel` and consumed by :func:`estimate_did`.
_PANEL_COLUMNS: tuple[str, ...] = ("y", "treated", "post", "interaction", "event", "name")

#: The two supported clustering schemes.
_CLUSTER_EVENT: str = "event"
_CLUSTER_EVENT_NAME: str = "event+name"

#: Minimum number of distinct clusters needed for a usable clustered SE. With a
#: single cluster the finite-sample correction ``G / (G - 1)`` is undefined.
_MIN_CLUSTERS: int = 2

#: Number of regressors in the DiD design (intercept + treated + post + interaction).
_N_PARAMS: int = 4


@dataclass(frozen=True, slots=True)
class DiDResult:
    """The difference-in-differences interaction estimate (clustered SEs).

    Attributes
    ----------
    coef:
        The ``treated x post`` interaction coefficient — the descriptive
        treated-minus-control differential response.
    std_error:
        The clustered standard error of ``coef``.
    t_stat:
        ``coef / std_error``.
    p_value:
        The two-sided p-value of ``coef`` under the clustered SE.
    n_obs:
        Number of (name x period) observations in the regression.
    n_clusters:
        Number of clusters (events) the SE is based on.
    cluster:
        The clustering scheme used (``"event"`` or ``"event+name"``).
    """

    coef: float
    std_error: float
    t_stat: float
    p_value: float
    n_obs: int
    n_clusters: int
    cluster: str

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the DiD estimate."""
        return asdict(self)


def build_did_panel(
    abnormal_by_event: list[pd.DataFrame],
    surprises: list[SurpriseLabel],
    rate_sensitive: list[str],
    *,
    treated_surprise: SurpriseLabel = "hawkish",
    control_surprise: SurpriseLabel = "dovish",
) -> pd.DataFrame:
    """Assemble the long-format DiD regression panel.

    Each per-event abnormal-return matrix is cumulated to a per-(event, name)
    cumulative abnormal return (CAR) — the natural DiD observation — and stacked
    into a tidy frame with columns ``y`` (the CAR), ``treated`` (1 if
    rate-sensitive), ``post`` (1 if the surprise is the ``treated_surprise``), the
    ``interaction`` (``treated * post``), an ``event`` cluster id, and a ``name``
    id.

    Only events whose surprise is the ``treated_surprise`` or the
    ``control_surprise`` enter the panel — the DiD contrasts the two surprise
    signs, so ``neutral`` (no-change) events carry no ``post`` contrast and are
    dropped.

    Parameters
    ----------
    abnormal_by_event:
        Per-event abnormal-return matrices (rows = event-relative day, columns =
        ticker), as produced by
        :func:`fedcausal.eventstudy.abnormal.abnormal_returns`.
    surprises:
        The surprise label per event (parallel to ``abnormal_by_event``).
    rate_sensitive:
        The tickers designated rate-sensitive (treated).
    treated_surprise, control_surprise:
        Which surprise signs define the ``post`` contrast. ``treated_surprise``
        maps to ``post == 1``; ``control_surprise`` to ``post == 0``.

    Returns
    -------
    pandas.DataFrame
        The long-format DiD panel with columns ``_PANEL_COLUMNS``.

    Raises
    ------
    ValidationError
        If the inputs are misaligned, empty, or the two surprise labels coincide.
    InsufficientDataError
        If, after filtering to the two surprise signs, the panel is empty or one
        of the four DiD cells (treated/control x post/pre) is unpopulated.
    """
    if not abnormal_by_event:
        raise ValidationError("build_did_panel: abnormal_by_event must be non-empty.")
    if len(abnormal_by_event) != len(surprises):
        raise ValidationError(
            "build_did_panel: abnormal_by_event and surprises must be the same length, "
            f"got {len(abnormal_by_event)} and {len(surprises)}."
        )
    if treated_surprise == control_surprise:
        raise ValidationError(
            "build_did_panel: treated_surprise and control_surprise must differ, "
            f"both are {treated_surprise!r}."
        )

    treated_set = set(rate_sensitive)
    rows: list[dict[str, Any]] = []
    for event_id, (abnormal, surprise) in enumerate(zip(abnormal_by_event, surprises, strict=True)):
        if surprise == treated_surprise:
            post = 1
        elif surprise == control_surprise:
            post = 0
        else:
            # ``neutral`` (or any out-of-contrast) event carries no post signal.
            continue
        # CAR per name = sum of abnormal returns over the event window. A missing
        # (NaN) abnormal-return day leaves the CAR UNDEFINED rather than silently
        # zero-filled (``skipna=False``), so that name drops out below instead of
        # contributing a biased, partially-observed CAR.
        car = abnormal.sum(axis=0, skipna=False)
        for name, value in car.items():
            ticker = str(name)
            y = float(value)
            if not np.isfinite(y):
                continue
            treated = 1 if ticker in treated_set else 0
            rows.append(
                {
                    "y": y,
                    "treated": treated,
                    "post": post,
                    "interaction": treated * post,
                    "event": event_id,
                    "name": ticker,
                }
            )

    if not rows:
        raise InsufficientDataError(
            "build_did_panel: no usable observations after filtering to the "
            f"{treated_surprise!r}/{control_surprise!r} surprise contrast."
        )

    panel = pd.DataFrame(rows, columns=list(_PANEL_COLUMNS))
    panel = panel.astype(
        {
            "y": "float64",
            "treated": "int64",
            "post": "int64",
            "interaction": "int64",
            "event": "int64",
            "name": "string",
        }
    )
    _assert_did_cells_populated(panel)
    return panel


def _assert_did_cells_populated(panel: pd.DataFrame) -> None:
    """Assert all four DiD cells (treated/control x post/pre) carry observations.

    The interaction coefficient is only identified when each of the four
    treated x post cells is non-empty; otherwise the design matrix is rank
    deficient.
    """
    cells = {
        (int(t), int(p))
        for t, p in zip(panel["treated"].to_numpy(), panel["post"].to_numpy(), strict=True)
    }
    required = {(0, 0), (0, 1), (1, 0), (1, 1)}
    missing = required - cells
    if missing:
        raise InsufficientDataError(
            "build_did_panel: the DiD interaction is not identified; the following "
            f"treated/post cells are empty: {sorted(missing)}."
        )


def _design_matrix(panel: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``(X, y)`` arrays for ``y ~ 1 + treated + post + interaction``."""
    n = int(panel.shape[0])
    treated = panel["treated"].to_numpy(dtype=np.float64)
    post = panel["post"].to_numpy(dtype=np.float64)
    interaction = panel["interaction"].to_numpy(dtype=np.float64)
    x = np.column_stack([np.ones(n, dtype=np.float64), treated, post, interaction])
    y = panel["y"].to_numpy(dtype=np.float64)
    return x, y


def _cluster_meat(x: np.ndarray, resid: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Sum the cluster score outer products ``sum_g (X_g' u_g)(X_g' u_g)'``."""
    k = x.shape[1]
    meat = np.zeros((k, k), dtype=np.float64)
    for g in np.unique(codes):
        mask = codes == g
        score = x[mask].T @ resid[mask]
        meat += np.outer(score, score)
    return meat


def _cluster_codes(panel: pd.DataFrame, column: str) -> np.ndarray:
    """Return integer cluster codes for ``column`` (factorized, contiguous)."""
    codes, _ = pd.factorize(panel[column], sort=True)
    return np.asarray(codes, dtype=np.int64)


def _interaction_codes(event_codes: np.ndarray, name_codes: np.ndarray) -> np.ndarray:
    """Return integer codes for the event-and-name intersection clusters."""
    n_names = int(name_codes.max()) + 1 if name_codes.size else 1
    pair = event_codes.astype(np.int64) * n_names + name_codes.astype(np.int64)
    codes, _ = pd.factorize(pair, sort=True)
    return np.asarray(codes, dtype=np.int64)


def _clustered_cov(
    x: np.ndarray,
    resid: np.ndarray,
    xtx_inv: np.ndarray,
    panel: pd.DataFrame,
    *,
    cluster: str,
) -> tuple[np.ndarray, int]:
    """Compute the cluster-robust sandwich covariance and the cluster count.

    For one-way ``"event"`` clustering this is the textbook White/Liang-Zeger
    sandwich with the ``statsmodels`` finite-sample correction
    ``(G / (G - 1)) * ((N - 1) / (N - K))`` (parity to 1e-10). For two-way
    ``"event+name"`` clustering it is the Cameron-Gelbach-Miller combination
    ``V_event + V_name - V_(event∩name)``, with each component carrying its own
    cluster-count correction; the reported ``n_clusters`` is the smaller of the
    two cluster dimensions (the binding dimension for inference).
    """
    n, k = x.shape

    def _one_way(codes: np.ndarray) -> tuple[np.ndarray, int]:
        n_clusters = int(np.unique(codes).size)
        meat = _cluster_meat(x, resid, codes)
        cov = xtx_inv @ meat @ xtx_inv
        if n_clusters < _MIN_CLUSTERS:
            # A single cluster makes the ``G / (G - 1)`` correction undefined; the
            # caller turns this into a typed ``InsufficientDataError``.
            return cov, n_clusters
        # statsmodels default finite-sample correction (adjust_df + use_correction).
        correction = (n_clusters / (n_clusters - 1.0)) * ((n - 1.0) / (n - k))
        return cov * correction, n_clusters

    event_codes = _cluster_codes(panel, "event")
    if cluster == _CLUSTER_EVENT:
        return _one_way(event_codes)

    # Two-way (event, name): Cameron-Gelbach-Miller V_e + V_n - V_{e&n}.
    name_codes = _cluster_codes(panel, "name")
    cov_event, n_event = _one_way(event_codes)
    cov_name, n_name = _one_way(name_codes)
    inter_codes = _interaction_codes(event_codes, name_codes)
    cov_inter, _ = _one_way(inter_codes)
    cov = cov_event + cov_name - cov_inter
    return cov, min(n_event, n_name)


def estimate_did(
    panel: pd.DataFrame,
    *,
    cluster: str = "event",
    alpha: float = DEFAULT_ALPHA,
) -> DiDResult:
    """Estimate the DiD interaction with CLUSTERED standard errors.

    Fits ``y ~ treated + post + treated:post`` by OLS and computes
    cluster-robust standard errors (one-way by ``event``, or two-way by ``event``
    and ``name`` via Cameron-Gelbach-Miller) so inference accounts for
    within-event cross-sectional correlation and across-event serial correlation.
    The reported ``coef`` is the ``treated x post`` interaction — the descriptive
    treated-minus-control differential response.

    Parameters
    ----------
    panel:
        The long-format DiD panel from :func:`build_did_panel`.
    cluster:
        ``"event"`` (one-way) or ``"event+name"`` (two-way) clustering.
    alpha:
        Significance level (recorded for downstream verdict use).

    Returns
    -------
    DiDResult
        The interaction coefficient and clustered inference.

    Raises
    ------
    ValidationError
        If ``panel`` is missing required columns, ``cluster`` is unknown, or
        ``alpha`` is out of range.
    InsufficientDataError
        If there are too few observations or clusters, or the design is rank
        deficient (a DiD cell is unpopulated).
    """
    validate_alpha(alpha)
    if cluster not in (_CLUSTER_EVENT, _CLUSTER_EVENT_NAME):
        raise ValidationError(
            f"cluster must be {_CLUSTER_EVENT!r} or {_CLUSTER_EVENT_NAME!r}, got {cluster!r}."
        )
    missing_cols = [c for c in _PANEL_COLUMNS if c not in panel.columns]
    if missing_cols:
        raise ValidationError(f"estimate_did: panel is missing columns {missing_cols}.")
    if panel.shape[0] < _N_PARAMS + 1:
        raise InsufficientDataError(
            f"estimate_did: need at least {_N_PARAMS + 1} observations, got {panel.shape[0]}."
        )

    x, y = _design_matrix(panel)
    n, k = x.shape

    xtx = x.T @ x
    if np.linalg.matrix_rank(xtx) < k:
        raise InsufficientDataError(
            "estimate_did: the DiD design matrix is rank deficient (a treated/post "
            "cell is unpopulated or a regressor is collinear)."
        )
    xtx_inv = np.linalg.inv(xtx)
    beta = xtx_inv @ (x.T @ y)
    resid = y - x @ beta

    cov, n_clusters = _clustered_cov(x, resid, xtx_inv, panel, cluster=cluster)
    if n_clusters < _MIN_CLUSTERS:
        raise InsufficientDataError(
            f"estimate_did: clustered inference needs at least {_MIN_CLUSTERS} "
            f"clusters, got {n_clusters}."
        )

    # The interaction is the last regressor (column index 3).
    coef = float(beta[_N_PARAMS - 1])
    var = float(cov[_N_PARAMS - 1, _N_PARAMS - 1])
    std_error = float(np.sqrt(var)) if var > 0.0 else 0.0

    if std_error <= EPS:
        # Degenerate (zero clustered variance): no resolvable signal.
        t_stat = 0.0
        p_value = 1.0
    else:
        t_stat = coef / std_error
        # Cluster-robust inference references the t distribution with G - 1
        # degrees of freedom (the conventional cluster small-sample reference).
        df = max(n_clusters - 1, 1)
        p_value = float(2.0 * stats.t.sf(abs(t_stat), df=df))

    return DiDResult(
        coef=coef,
        std_error=std_error,
        t_stat=float(t_stat),
        p_value=float(p_value),
        n_obs=int(n),
        n_clusters=int(n_clusters),
        cluster=cluster,
    )
