"""End-to-end cointegration pipeline.

The pipeline runs each individual test in turn and aggregates the
verdict using an explicit four-cell truth table: a pair is treated as
cointegrated only when *all* of the following hold simultaneously:

1. Engle-Granger rejects the unit-root null on the spread,
2. The Johansen rank is at least one,
3. KPSS fails to reject stationarity of the spread, and
4. Both legs individually look I(1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pairs._exceptions import InputError, NonStationaryError

from ._internal.align import _inner_join_and_dropna
from .adf import unit_root_check
from .bootstrap import bootstrap_coint_pvalue
from .engle_granger import engle_granger
from .johansen import johansen
from .kpss import kpss_spread
from .results import JohansenResult, PipelineResult


def full_pipeline(
    y0: pd.Series | np.ndarray,
    y1: pd.Series | np.ndarray,
    *,
    alpha: float = 0.05,
    use_log: bool = True,
    n_boot: int = 0,
    run_johansen: bool = True,
    rng: np.random.Generator | None = None,
) -> PipelineResult:
    """Run the complete cointegration battery on a pair of series.

    Parameters
    ----------
    y0, y1 : pandas.Series or numpy.ndarray
        Aligned price (or log-price) series.
    alpha : float, default ``0.05``
        Significance level used for every constituent test.
    use_log : bool, default ``True``
        Whether to log-transform the inputs before testing.
    n_boot : int, default ``0``
        If positive, also run the stationary-bootstrap p-value.
    run_johansen : bool, default ``True``
        Set to ``False`` to skip the Johansen step (useful when one of
        the legs is not strictly stationary in differences).
    rng : numpy.random.Generator, optional
        Random source forwarded to :func:`bootstrap_coint_pvalue`.

    Returns
    -------
    PipelineResult
        Container with each test's verdict and the aggregate decision.
    """
    if not (0.0 < alpha < 1.0):
        raise InputError(f"alpha must lie in (0, 1); got {alpha}")
    if n_boot < 0:
        raise InputError(f"n_boot must be non-negative; got {n_boot}")

    a, b = _inner_join_and_dropna(y0, y1)
    if use_log:
        if (a <= 0).any() or (b <= 0).any():
            raise InputError("use_log=True requires strictly positive prices")
        a = pd.Series(np.log(a.to_numpy()), index=a.index, name=a.name)
        b = pd.Series(np.log(b.to_numpy()), index=b.index, name=b.name)

    leg0 = unit_root_check(a, alpha=alpha, leg_name=str(a.name or "y0"))
    leg1 = unit_root_check(b, alpha=alpha, leg_name=str(b.name or "y1"))

    eg = engle_granger(a, b, use_log=False)

    joh: JohansenResult | None = None
    if run_johansen:
        joh = johansen(pd.concat({"y0": a, "y1": b}, axis=1))

    try:
        kpss_res = kpss_spread(eg.residuals, alpha=alpha)
    except NonStationaryError as exc:  # pragma: no cover - defensive
        raise InputError(f"KPSS could not be evaluated: {exc}") from exc

    boot_res = None
    if n_boot > 0:
        boot_res = bootstrap_coint_pvalue(a, b, n_boot=n_boot, rng=rng)

    johansen_ok = joh is None or joh.rank >= 1
    cointegrated = bool(
        eg.pvalue < alpha and johansen_ok and kpss_res.is_stationary and leg0.is_i1 and leg1.is_i1,
    )

    return PipelineResult(
        eg=eg,
        johansen=joh,
        kpss=kpss_res,
        leg0_unit_root=leg0,
        leg1_unit_root=leg1,
        bootstrap=boot_res,
        cointegrated=cointegrated,
        alpha=alpha,
    )
