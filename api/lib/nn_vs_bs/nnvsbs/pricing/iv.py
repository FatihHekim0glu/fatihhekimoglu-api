"""Implied-volatility solver (Newton with a bracketed Brent fallback).

Inverts the Black-Scholes price for the volatility that reproduces an observed
option price. The primary path is Newton-Raphson seeded with the Brenner-Subrahmanyam
at-the-money approximation and accelerated by the analytic vega; if Newton leaves
the no-arbitrage bracket or stalls, the solver falls back to a robust bracketed
Brent root find (``scipy.optimize.brentq``). Validated to ``1e-8`` against
``py_vollib`` in the parity tests.

IMPLIED VOLATILITY IS AN ANALYSIS OUTPUT ONLY. It is a deterministic Black-Scholes
inversion of the *target price*, so it must NEVER enter the neural-net feature
matrix when the label is price — doing so would leak the label (see
:mod:`nnvsbs.features.build`). This module exists to *compute* the smile for
plotting and diagnostics, not to feed the model.

The solver carries its own minimal, self-contained Black-Scholes price/vega kernel
(``numpy`` + ``scipy.stats.norm``, the ``[data]`` extra) so it can be developed and
parity-validated independently of :mod:`nnvsbs.pricing.black_scholes`. ``scipy`` is
imported lazily inside the solver; importing this module has no side effects.
"""

from __future__ import annotations

import math

import numpy as np

from nnvsbs._exceptions import ConvergenceError, ValidationError
from nnvsbs._typing import FloatArray
from nnvsbs.pricing.black_scholes import OptionKind

#: Lower bound on the volatility search bracket used by the Brent fallback.
_SIGMA_LO = 1e-9
#: Upper bound on the volatility search bracket used by the Brent fallback.
_SIGMA_HI = 50.0
#: Volatility floor below which the analytic-vega Newton step is treated as
#: ill-conditioned and the solver defers to the bracketed Brent fallback.
_VEGA_FLOOR = 1e-12
#: Absolute slack added to the no-arbitrage bounds check (numerical headroom).
_BOUND_SLACK = 1e-9


def _as_1d(value: object, *, name: str) -> FloatArray:
    """Coerce ``value`` to a 1-D float64 array for the vectorized solver.

    Parameters
    ----------
    value:
        A scalar or array-like of option parameters / prices.
    name:
        Human-readable label used in error messages.

    Returns
    -------
    FloatArray
        ``value`` as a contiguous 1-D float64 array.

    Raises
    ------
    ValidationError
        If ``value`` is not coercible to a finite float array.
    """
    try:
        arr = np.asarray(value, dtype="float64")
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValidationError(f"{name} is not coercible to a float array: {exc!r}") from exc
    flat = np.atleast_1d(arr).reshape(-1)
    if not np.all(np.isfinite(flat)):
        raise ValidationError(f"{name} contains non-finite values.")
    return flat


def _bs_price_vega_scalar(
    sigma: float,
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    *,
    kind: OptionKind,
) -> tuple[float, float]:
    """Return the Black-Scholes-Merton ``(price, vega)`` for one contract.

    A self-contained scalar kernel (``math`` + ``scipy.stats.norm``) used by the
    Newton iteration so the solver does not depend on the (separately authored)
    :mod:`nnvsbs.pricing.black_scholes` vectorized kernels.

    Parameters
    ----------
    sigma:
        Trial volatility (must be strictly positive).
    s, k, t, r, q:
        Spot, strike, time-to-expiry (years), risk-free rate, dividend yield.
    kind:
        ``"call"`` or ``"put"``.

    Returns
    -------
    tuple[float, float]
        The option price and its vega (``dV/dsigma``, per unit vol).
    """
    from scipy.stats import norm

    sqrt_t = math.sqrt(t)
    vol_sqrt_t = sigma * sqrt_t
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    disc_q = math.exp(-q * t)
    disc_r = math.exp(-r * t)
    if kind == "call":
        price = s * disc_q * float(norm.cdf(d1)) - k * disc_r * float(norm.cdf(d2))
    else:
        price = k * disc_r * float(norm.cdf(-d2)) - s * disc_q * float(norm.cdf(-d1))
    vega = s * disc_q * float(norm.pdf(d1)) * sqrt_t
    return price, vega


def _intrinsic_and_bounds(
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    *,
    kind: OptionKind,
) -> tuple[float, float]:
    """Return the no-arbitrage ``(lower, upper)`` price bounds for a contract.

    For a European call the price lies in
    ``[max(S e^{-qT} - K e^{-rT}, 0), S e^{-qT}]``; the put bounds follow by
    symmetry. A target price outside ``[lower, upper]`` admits no implied vol.

    Parameters
    ----------
    s, k, t, r, q:
        Spot, strike, time-to-expiry (years), risk-free rate, dividend yield.
    kind:
        ``"call"`` or ``"put"``.

    Returns
    -------
    tuple[float, float]
        The lower and upper no-arbitrage price bounds.
    """
    fwd = s * math.exp(-q * t)
    disc_k = k * math.exp(-r * t)
    if kind == "call":
        lower = max(fwd - disc_k, 0.0)
        upper = fwd
    else:
        lower = max(disc_k - fwd, 0.0)
        upper = disc_k
    return lower, upper


def _solve_one(
    price: float,
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    *,
    kind: OptionKind,
    tol: float,
    max_iter: int,
) -> float:
    """Solve the implied volatility of a single observed price.

    Tries analytic-vega Newton from the Brenner-Subrahmanyam seed, then falls back
    to a bracketed Brent (``scipy.optimize.brentq``) root find on the price error
    whenever Newton is ill-conditioned (tiny vega, a non-finite/out-of-bracket
    step, or a stall). Both paths target the same absolute price tolerance ``tol``.

    To stay well-conditioned the inversion is always done on the *out-of-the-money*
    right: an in-the-money price is dominated by its (vol-independent) intrinsic
    value, so its time-value — where all the vega is — is a catastrophic
    cancellation of two large numbers. Put-call parity gives the OTM counterpart's
    price exactly, with no loss of information, and the implied vol is identical.

    Parameters
    ----------
    price:
        Observed option price to invert.
    s, k, t, r, q:
        Spot, strike, time-to-expiry (years), risk-free rate, dividend yield.
    kind:
        ``"call"`` or ``"put"``.
    tol:
        Absolute price tolerance for convergence.
    max_iter:
        Maximum Newton iterations before falling back to Brent.

    Returns
    -------
    float
        The implied volatility.

    Raises
    ------
    ConvergenceError
        If ``price`` is outside the no-arbitrage bounds, or neither Newton nor
        Brent reaches ``tol`` within the iteration budget.
    """
    lower, upper = _intrinsic_and_bounds(s, k, t, r, q, kind=kind)
    if price < lower - _BOUND_SLACK or price > upper + _BOUND_SLACK:
        raise ConvergenceError(
            f"price {price!r} is outside the no-arbitrage bounds "
            f"[{lower!r}, {upper!r}] for (S={s}, K={k}, T={t}, r={r}, q={q}, "
            f"kind={kind!r}); no implied volatility exists."
        )

    # Fold to the OTM right (the well-conditioned one). Put-call parity:
    # C - P = S e^{-qT} - K e^{-rT}, so the counterpart's price follows exactly and
    # carries the same implied vol. The OTM right is the call when K > forward.
    forward = s * math.exp((r - q) * t)
    otm_kind: OptionKind = "call" if k >= forward else "put"
    if otm_kind != kind:
        parity = s * math.exp(-q * t) - k * math.exp(-r * t)
        price = price - parity if kind == "call" else price + parity
        kind = otm_kind
        # Clamp away sub-floor round-off introduced by the parity subtraction.
        price = max(price, 0.0)

    # Degenerate sigma -> 0 limit: if the price is at or below the model price at the
    # vol floor, no strictly-positive vol reproduces it (the objective has no sign
    # change), so report the floor. This is keyed on the *model* price at ``_SIGMA_LO``
    # — NOT a slack on the intrinsic — so a legitimately tiny OTM price (still strictly
    # above the floor price) is left for Newton/Brent, which recover the true vol.
    floor_price, _ = _bs_price_vega_scalar(_SIGMA_LO, s, k, t, r, q, kind=kind)
    if price <= floor_price:
        return _SIGMA_LO

    sigma = _newton(price, s, k, t, r, q, kind=kind, tol=tol, max_iter=max_iter)
    if sigma is not None:
        return sigma
    return _brent(price, s, k, t, r, q, kind=kind, tol=tol)


def _newton(
    price: float,
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    *,
    kind: OptionKind,
    tol: float,
    max_iter: int,
) -> float | None:
    """Run analytic-vega Newton-Raphson; return ``None`` if it does not converge.

    Returns ``None`` (rather than raising) so the caller can transparently fall
    back to the bracketed Brent solver. The step is rejected — triggering the
    fallback — when vega underflows, the proposed ``sigma`` is non-positive or
    non-finite, or the iteration budget is exhausted without reaching ``tol``.

    Parameters
    ----------
    price:
        Observed option price to invert.
    s, k, t, r, q:
        Spot, strike, time-to-expiry (years), risk-free rate, dividend yield.
    kind:
        ``"call"`` or ``"put"``.
    tol:
        Absolute price tolerance for convergence.
    max_iter:
        Maximum Newton iterations.

    Returns
    -------
    float | None
        The implied volatility, or ``None`` if Newton failed to converge.
    """
    sigma = _brenner_subrahmanyam_scalar(price, s, t)
    sigma = min(max(sigma, _SIGMA_LO), _SIGMA_HI)
    for _ in range(max_iter):
        model, vega = _bs_price_vega_scalar(sigma, s, k, t, r, q, kind=kind)
        # The vega floor is checked BEFORE accepting convergence: where vega
        # underflows the price is (numerically) flat in sigma, so a small price
        # residual does NOT identify the vol (a deep-OTM/near-expiry price can sit
        # far below ``tol`` at a wildly wrong sigma). Such ill-conditioned points
        # defer to the bracketed Brent solver, which solves on sigma directly.
        if vega < _VEGA_FLOOR:
            return None
        diff = model - price
        if abs(diff) <= tol:
            return sigma
        step = diff / vega
        sigma_next = sigma - step
        if not math.isfinite(sigma_next) or sigma_next <= 0.0 or sigma_next > _SIGMA_HI:
            return None
        sigma = sigma_next
    return None


def _brent(
    price: float,
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    *,
    kind: OptionKind,
    tol: float,
) -> float:
    """Bracketed Brent root find on the price error (the robust fallback).

    Solves ``bs_price(sigma) - price == 0`` over ``[_SIGMA_LO, _SIGMA_HI]`` with
    ``scipy.optimize.brentq``. The objective is monotone increasing in ``sigma``
    (vega > 0), so a sign change is guaranteed once the target price is inside the
    no-arbitrage bounds — which the caller has already verified.

    Parameters
    ----------
    price:
        Observed option price to invert.
    s, k, t, r, q:
        Spot, strike, time-to-expiry (years), risk-free rate, dividend yield.
    kind:
        ``"call"`` or ``"put"``.
    tol:
        Absolute price tolerance; mapped to the solver's ``xtol`` on ``sigma``.

    Returns
    -------
    float
        The implied volatility.

    Raises
    ------
    ConvergenceError
        If Brent cannot bracket a root or fails to converge.
    """
    from scipy.optimize import brentq

    def objective(sigma: float) -> float:
        model, _ = _bs_price_vega_scalar(sigma, s, k, t, r, q, kind=kind)
        return model - price

    f_lo = objective(_SIGMA_LO)
    f_hi = objective(_SIGMA_HI)
    if f_lo * f_hi > 0.0:  # pragma: no cover - defensive: caller verifies no-arb bounds
        raise ConvergenceError(
            f"Brent could not bracket an implied volatility for price {price!r} "
            f"(S={s}, K={k}, T={t}, r={r}, q={q}, kind={kind!r})."
        )
    try:
        root = float(brentq(objective, _SIGMA_LO, _SIGMA_HI, xtol=1e-12, rtol=1e-15, maxiter=200))
    except (RuntimeError, ValueError) as exc:  # pragma: no cover - defensive: smooth monotone obj
        raise ConvergenceError(
            f"Brent failed to converge for price {price!r} "
            f"(S={s}, K={k}, T={t}, r={r}, q={q}, kind={kind!r}): {exc!r}"
        ) from exc

    if abs(objective(root)) > max(tol, 1e-7):  # pragma: no cover - defensive: xtol guarantees fit
        raise ConvergenceError(
            f"Brent converged to sigma={root!r} but the price residual exceeds "
            f"tolerance for price {price!r} (S={s}, K={k}, T={t}, kind={kind!r})."
        )
    return root


def _brenner_subrahmanyam_scalar(price: float, s: float, t: float) -> float:
    """Scalar Brenner-Subrahmanyam at-the-money implied-vol seed.

    Closed-form ATM approximation ``sigma_0 ~ sqrt(2 pi / T) * price / S``.

    Parameters
    ----------
    price:
        Observed option price.
    s, t:
        Spot and time-to-expiry (years).

    Returns
    -------
    float
        The seed volatility (clamped to be strictly positive downstream).
    """
    seed = math.sqrt(2.0 * math.pi / t) * (price / s)
    if not math.isfinite(seed) or seed <= 0.0:
        return 0.5
    return seed


def implied_volatility(
    price: FloatArray,
    s: FloatArray,
    k: FloatArray,
    t: FloatArray,
    r: FloatArray,
    q: FloatArray,
    *,
    kind: OptionKind = "call",
    tol: float = 1e-8,
    max_iter: int = 100,
) -> FloatArray:
    """Solve for the Black-Scholes implied volatility of an observed price.

    Vectorized over broadcastable inputs. Each element is solved by Newton-Raphson
    (analytic-vega Newton step from a Brenner-Subrahmanyam seed) with a bracketed
    Brent fallback, to a price tolerance of ``tol``.

    ANALYSIS OUTPUT ONLY: the returned IV must never be used as a model feature
    when the label is price (it is a one-to-one inversion of that price).

    Parameters
    ----------
    price:
        Observed option price(s) to invert (must be within the no-arbitrage
        bounds for the given parameters).
    s, k, t, r, q:
        Spot, strike, time-to-expiry (years), risk-free rate, and continuous
        dividend yield.
    kind:
        ``"call"`` (default) or ``"put"``.
    tol:
        Absolute price tolerance for convergence (default ``1e-8``).
    max_iter:
        Maximum Newton iterations before falling back to Brent (default ``100``).

    Returns
    -------
    FloatArray
        The implied volatility(ies), same broadcast shape as the inputs.

    Raises
    ------
    ValidationError
        If any of ``s, k, t`` is non-positive or ``kind`` is unknown.
    ConvergenceError
        If a price lies outside the no-arbitrage bounds (no implied vol exists)
        or neither Newton nor Brent reaches ``tol`` within the iteration budget.
    """
    if kind not in ("call", "put"):
        raise ValidationError(f"kind must be 'call' or 'put', got {kind!r}.")
    if tol <= 0.0:
        raise ValidationError(f"tol must be positive, got {tol!r}.")
    if max_iter < 1:
        raise ValidationError(f"max_iter must be >= 1, got {max_iter!r}.")

    # Broadcast all parameters to a common shape, then solve element-by-element.
    try:
        bcast = np.broadcast_arrays(
            np.asarray(price, dtype="float64"),
            np.asarray(s, dtype="float64"),
            np.asarray(k, dtype="float64"),
            np.asarray(t, dtype="float64"),
            np.asarray(r, dtype="float64"),
            np.asarray(q, dtype="float64"),
        )
    except ValueError as exc:
        raise ValidationError(f"implied_volatility inputs are not broadcastable: {exc!r}") from exc

    out_shape = bcast[0].shape
    # ``np.ravel`` on a broadcast *view* materializes a contiguous 1-D copy without
    # promoting a 0-D (all-scalar) case to shape ``(1,)`` in ``out_shape``.
    price_f = np.ravel(bcast[0])
    s_f = np.ravel(bcast[1])
    k_f = np.ravel(bcast[2])
    t_f = np.ravel(bcast[3])
    r_f = np.ravel(bcast[4])
    q_f = np.ravel(bcast[5])

    for arr, name in ((s_f, "s"), (k_f, "k"), (t_f, "t")):
        if np.any(arr <= 0.0):
            raise ValidationError(f"{name} must be strictly positive.")
    if not np.all(np.isfinite(price_f)):
        raise ValidationError("price contains non-finite values.")

    result = np.empty(price_f.shape, dtype="float64")
    for i in range(price_f.shape[0]):
        result[i] = _solve_one(
            float(price_f[i]),
            float(s_f[i]),
            float(k_f[i]),
            float(t_f[i]),
            float(r_f[i]),
            float(q_f[i]),
            kind=kind,
            tol=tol,
            max_iter=max_iter,
        )

    return np.asarray(result.reshape(out_shape), dtype="float64")


def _brenner_subrahmanyam_seed(
    price: FloatArray,
    s: FloatArray,
    t: FloatArray,
) -> FloatArray:
    """Return the Brenner-Subrahmanyam at-the-money implied-vol seed.

    The closed-form ATM approximation ``sigma_0 ~ sqrt(2 pi / T) * price / S``
    gives a fast, stable starting point for the Newton iteration. Vectorized
    counterpart of :func:`_brenner_subrahmanyam_scalar`.

    Parameters
    ----------
    price:
        Observed option price(s).
    s, t:
        Spot and time-to-expiry (years).

    Returns
    -------
    FloatArray
        The seed volatility(ies), strictly positive.
    """
    price_a = _as_1d(price, name="price")
    s_a = _as_1d(s, name="s")
    t_a = _as_1d(t, name="t")
    price_b, s_b, t_b = np.broadcast_arrays(price_a, s_a, t_a)
    if np.any(s_b <= 0.0):
        raise ValidationError("s must be strictly positive.")
    if np.any(t_b <= 0.0):
        raise ValidationError("t must be strictly positive.")
    seed = np.sqrt(2.0 * np.pi / t_b) * (price_b / s_b)
    seed = np.where(np.isfinite(seed) & (seed > 0.0), seed, 0.5)
    return np.asarray(seed, dtype="float64")
