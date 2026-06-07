"""User-facing view specifications and the :class:`Views` container.

A Black-Litterman investor expresses subjective opinions as a collection of
*views*.  Each view is either

* an **absolute view** on a single asset (``E[r_i] = q``), or
* a **relative view** that pits one basket of assets against another
  (``sum(P_long) * r - sum(P_short) * r = q``).

The :class:`Views` class collects a sequence of these dataclasses, validates
them against the asset universe, and assembles the ``(P, Q)`` matrices used
downstream by the Theil mixed-estimation form.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .exceptions import ViewValidationError

_ZERO_TOL: float = 1e-12


@dataclass(frozen=True)
class AbsoluteView:
    """A point estimate on a single asset's expected excess return."""

    asset: str
    expected_return: float
    confidence: float | None = None


@dataclass(frozen=True)
class RelativeView:
    """A spread view between a long and a short basket of assets.

    The basket weights specify how each leg is constructed: ``long_leg`` maps
    asset names to non-negative coefficients and likewise ``short_leg``.  The
    resulting pick row is ``long_leg - short_leg`` and must sum to zero (this
    is verified at construction time).
    """

    long_leg: Mapping[str, float]
    short_leg: Mapping[str, float]
    spread: float
    confidence: float | None = field(default=None)


ViewSpec = AbsoluteView | RelativeView


class Views:
    """Container that validates and serialises a list of :class:`ViewSpec`."""

    def __init__(
        self,
        views: Sequence[ViewSpec],
        assets: Sequence[str],
    ) -> None:
        self._views: tuple[ViewSpec, ...] = tuple(views)
        self._assets: tuple[str, ...] = tuple(assets)
        self._asset_index: dict[str, int] = {a: i for i, a in enumerate(self._assets)}
        self._validate()

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    def _validate(self) -> None:
        if len(self._asset_index) != len(self._assets):
            raise ViewValidationError("Asset universe contains duplicate names.")

        confidences_seen: list[bool] = []
        for k, view in enumerate(self._views):
            if isinstance(view, AbsoluteView):
                if view.asset not in self._asset_index:
                    raise ViewValidationError(f"View {k}: asset {view.asset!r} not in universe.")
            elif isinstance(view, RelativeView):
                self._validate_relative(k, view)
            else:  # pragma: no cover - defensive; dataclass union is closed
                raise ViewValidationError(f"View {k}: unsupported view type {type(view).__name__}.")

            c = view.confidence
            if c is not None and (not np.isfinite(c) or c < 0.0 or c > 1.0):
                raise ViewValidationError(f"View {k}: confidence must lie in [0, 1], got {c!r}.")
            confidences_seen.append(c is not None)

        if confidences_seen and any(confidences_seen) and not all(confidences_seen):
            raise ViewValidationError(
                "Confidences must be specified for all views or for none "
                "(mixed specification is ambiguous)."
            )

    def _validate_relative(self, k: int, view: RelativeView) -> None:
        if not view.long_leg or not view.short_leg:
            raise ViewValidationError(
                f"View {k}: relative view requires non-empty long and short legs."
            )
        row = np.zeros(len(self._assets))
        for asset, w in view.long_leg.items():
            if asset not in self._asset_index:
                raise ViewValidationError(f"View {k}: long-leg asset {asset!r} not in universe.")
            row[self._asset_index[asset]] += float(w)
        for asset, w in view.short_leg.items():
            if asset not in self._asset_index:
                raise ViewValidationError(f"View {k}: short-leg asset {asset!r} not in universe.")
            row[self._asset_index[asset]] -= float(w)
        if abs(row.sum()) > _ZERO_TOL:
            raise ViewValidationError(
                f"View {k}: relative-view pick row must sum to 0, got {row.sum():g}."
            )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def build_pq(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the ``(P, Q)`` matrices for the current view set."""
        k = len(self._views)
        n = len(self._assets)
        p_mat = np.zeros((k, n), dtype=float)
        q_vec = np.zeros(k, dtype=float)

        for i, view in enumerate(self._views):
            if isinstance(view, AbsoluteView):
                p_mat[i, self._asset_index[view.asset]] = 1.0
                q_vec[i] = float(view.expected_return)
            else:
                for asset, w in view.long_leg.items():
                    p_mat[i, self._asset_index[asset]] += float(w)
                for asset, w in view.short_leg.items():
                    p_mat[i, self._asset_index[asset]] -= float(w)
                q_vec[i] = float(view.spread)
        return p_mat, q_vec

    def build_omega_he_litterman(self, tau: float, cov: np.ndarray) -> np.ndarray:
        """Return the He-Litterman diagonal ``Omega = diag(P tau Sigma P^T)``."""
        if tau <= 0.0:
            raise ViewValidationError(f"tau must be > 0, got {tau}.")
        if cov.shape[0] != cov.shape[1] or cov.shape[0] != len(self._assets):
            raise ViewValidationError(
                f"Covariance shape {cov.shape} incompatible with universe "
                f"of size {len(self._assets)}."
            )
        p_mat, _ = self.build_pq()
        diag = np.einsum("ki,ij,kj->k", p_mat, tau * cov, p_mat)
        return np.diag(diag)

    def confidences(self) -> np.ndarray | None:
        """Return per-view confidences as an array, or ``None`` if unspecified."""
        if not self._views or self._views[0].confidence is None:
            return None
        return np.asarray([v.confidence for v in self._views], dtype=float)

    def has_confidences(self) -> bool:
        return self.confidences() is not None

    @property
    def assets(self) -> tuple[str, ...]:
        return self._assets

    @property
    def specs(self) -> tuple[ViewSpec, ...]:
        return self._views

    def __len__(self) -> int:
        return len(self._views)
