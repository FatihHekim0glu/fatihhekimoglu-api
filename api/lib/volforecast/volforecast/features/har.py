"""HAR-RV feature builder (Corsi 2009).

The Heterogeneous Autoregressive model of Realized Volatility decomposes the
RV process into trailing daily, weekly, and monthly average-RV components. These
same components are the regressors for the HAR-RV baseline and the engineered
features for the XGBoost model.

LAG SAFETY: every component is a trailing average of PAST RV and is then
``.shift()``-lagged so that the feature row at timestamp ``t`` uses only RV
observed at or before ``t`` — never ``t``'s own forward target. A property test
asserts this lag-safety. Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from volforecast._constants import (
    HAR_DAILY_WINDOW,
    HAR_MONTHLY_WINDOW,
    HAR_WEEKLY_WINDOW,
)
from volforecast._exceptions import ValidationError
from volforecast._validation import ensure_series

#: The ordered HAR component columns produced by :func:`har_components`. This is
#: the input contract shared by the HAR-RV baseline and the XGBoost model.
HAR_COMPONENT_COLUMNS: tuple[str, ...] = ("rv_daily", "rv_weekly", "rv_monthly")


@dataclass(frozen=True, slots=True)
class HARFeatures:
    """Immutable bundle of HAR-RV feature components and the aligned target.

    Attributes
    ----------
    features:
        The ``(T, k)`` feature frame with columns ``rv_daily``, ``rv_weekly``,
        ``rv_monthly`` (plus any exogenous columns such as ``vix``), all
        lagged so they are observable at the row timestamp.
    target:
        The aligned forward RV target series (NaN-dropped jointly with
        ``features``).
    feature_names:
        The ordered feature column names (the model's input contract).
    """

    features: pd.DataFrame
    target: pd.Series
    feature_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this bundle.

        Frames/series are rendered with ISO-formatted date keys so the bundle
        crosses the API boundary cleanly.
        """
        return {
            "features": {
                str(idx): {str(c): _safe_float(v) for c, v in row.items()}
                for idx, row in self.features.iterrows()
            },
            "target": {str(k): _safe_float(v) for k, v in self.target.items()},
            "feature_names": list(self.feature_names),
        }


def _safe_float(value: object) -> float | None:
    """Coerce ``value`` to a finite float, mapping NaN/Inf/None to ``None``."""
    import numpy as np

    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def har_components(rv: pd.Series) -> pd.DataFrame:
    r"""Build the three lagged HAR-RV components from a daily RV series.

    Returns a frame with columns

    - ``rv_daily``  : ``RV_{t-1}`` (yesterday's RV),
    - ``rv_weekly`` : the trailing 5-day mean of RV, lagged one day,
    - ``rv_monthly``: the trailing 22-day mean of RV, lagged one day,

    each ``.shift()``-lagged so the row at ``t`` contains only information
    available strictly before ``t`` (Corsi's cascade with no contemporaneous RV).

    Parameters
    ----------
    rv:
        A per-day realized-volatility series indexed by date.

    Returns
    -------
    pandas.DataFrame
        The lagged HAR component frame aligned to ``rv.index`` (leading rows NaN
        over the 22-day warm-up).

    Raises
    ------
    ValidationError
        If ``rv`` is empty.
    """
    series = ensure_series(rv, name="rv", allow_nan=True)

    # Trailing averages of PAST RV. ``min_periods`` equals the window so a
    # component is NaN until its full warm-up has accrued — no partial windows
    # leak a short-sample bias into early rows.
    daily = series.rolling(HAR_DAILY_WINDOW, min_periods=HAR_DAILY_WINDOW).mean()
    weekly = series.rolling(HAR_WEEKLY_WINDOW, min_periods=HAR_WEEKLY_WINDOW).mean()
    monthly = series.rolling(HAR_MONTHLY_WINDOW, min_periods=HAR_MONTHLY_WINDOW).mean()

    # ``.shift(1)`` is the leakage guard: the row at ``t`` carries the trailing
    # average computed up to and including ``t - 1`` only, so a HAR feature at
    # ``t`` never embeds ``RV_t`` (which lives in ``t``'s forward target window).
    frame = pd.DataFrame(
        {
            "rv_daily": daily.shift(1),
            "rv_weekly": weekly.shift(1),
            "rv_monthly": monthly.shift(1),
        },
        index=series.index,
    )
    return frame[list(HAR_COMPONENT_COLUMNS)]


def build_har_features(
    rv: pd.Series,
    target: pd.Series,
    *,
    exog: pd.DataFrame | None = None,
) -> HARFeatures:
    """Assemble the lagged HAR feature frame and align it to the forward target.

    Joins the lagged HAR components (and any exogenous, already-lagged columns
    such as a VIX level) with the forward RV ``target``, then drops rows with any
    NaN so the feature/target pair is complete and disjoint in time.

    Parameters
    ----------
    rv:
        A per-day realized-volatility series indexed by date.
    target:
        The forward RV target (e.g. from
        :func:`volforecast.realized.estimators.forward_rv_target`).
    exog:
        Optional exogenous feature frame (e.g. ``{"vix": ...}``). MUST already be
        lagged by the caller; this function does not re-lag exogenous inputs.

    Returns
    -------
    HARFeatures
        The aligned, NaN-free feature/target bundle.

    Raises
    ------
    ValidationError
        If ``rv``/``target`` cannot be aligned or the joined frame is empty.
    """
    components = har_components(rv)
    target_series = ensure_series(target, name="target", allow_nan=True)
    target_series = target_series.rename("rv_target")

    frame = components
    if exog is not None:
        if not isinstance(exog, pd.DataFrame):
            raise ValidationError("exog must be a pandas.DataFrame when provided.")
        if exog.shape[1] == 0:
            raise ValidationError("exog must have at least one column when provided.")
        overlap = set(frame.columns) & set(exog.columns)
        if overlap:
            raise ValidationError(f"exog columns collide with HAR components: {sorted(overlap)}.")
        # Caller-lagged exogenous features (e.g. a VIX level) are joined by index;
        # this function does NOT re-lag them.
        frame = frame.join(exog.astype("float64"), how="left")

    feature_names = tuple(str(c) for c in frame.columns)

    # Inner-join features and target on their common index, then drop any row
    # with a NaN so the surviving feature/target pair is complete AND disjoint in
    # time (features at {<= t}, target at {> t + gap}).
    joined = frame.join(target_series, how="inner").dropna(axis=0, how="any")
    if joined.empty:
        raise ValidationError(
            "build_har_features: no complete feature/target rows after alignment."
        )

    features_out = joined[list(feature_names)].astype("float64")
    target_out = joined["rv_target"].astype("float64")
    return HARFeatures(
        features=features_out,
        target=target_out,
        feature_names=feature_names,
    )
