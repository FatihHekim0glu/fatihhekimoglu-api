"""Hand-rolled Black-Scholes-Merton European option pricing and Greeks.

A from-scratch implementation of the Black-Scholes (1973) closed-form price for
European calls and puts with a continuous dividend yield ``q`` (the Merton 1973
extension), plus the analytic first/second-order Greeks (delta, gamma, vega,
theta, rho). These are the ground-truth pricer the whole project compares the
neural net against, and they are validated to ``1e-8`` against ``py_vollib`` (or
a hand-derived reference) in the parity tests.

The implementation depends only on ``numpy`` and ``scipy.stats.norm`` (via the
``[data]`` extra) and is fully vectorized: every function accepts scalars or
broadcastable arrays of ``(S, K, T, r, q, sigma)``. Put-call parity
``C - P = S e^{-qT} - K e^{-rT}`` holds by construction and is property-tested.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, get_args

import numpy as np
from scipy.stats import norm

from nnvsbs._exceptions import ValidationError
from nnvsbs._typing import FloatArray

#: Option right: ``"call"`` or ``"put"``.
OptionKind = Literal["call", "put"]

#: The accepted option-right strings, derived from :data:`OptionKind`.
_VALID_KINDS: frozenset[str] = frozenset(get_args(OptionKind))


def _check_kind(kind: str) -> None:
    """Raise :class:`ValidationError` if ``kind`` is not a known option right."""
    if kind not in _VALID_KINDS:
        valid = ", ".join(sorted(_VALID_KINDS))
        raise ValidationError(f"kind must be one of {{{valid}}}, got {kind!r}.")


def _positive(value: FloatArray, *, name: str) -> FloatArray:
    """Validate that every element of ``value`` is strictly positive.

    Parameters
    ----------
    value:
        A float64 array of option parameters.
    name:
        Human-readable label used in the error message.

    Returns
    -------
    FloatArray
        ``value`` unchanged (returned for convenient chaining).

    Raises
    ------
    ValidationError
        If ``value`` contains a NaN or a non-positive entry.
    """
    if bool(np.isnan(value).any()):
        raise ValidationError(f"{name} contains NaN values.")
    if bool((value <= 0.0).any()):
        raise ValidationError(f"{name} must be strictly positive.")
    return value


def _d1_d2(
    s: FloatArray,
    k: FloatArray,
    t: FloatArray,
    r: FloatArray,
    q: FloatArray,
    sigma: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Return the Black-Scholes ``(d1, d2)`` terms for broadcast inputs.

    Parameters
    ----------
    s, k, t, r, q, sigma:
        Spot, strike, time-to-expiry (years), risk-free rate, continuous
        dividend yield, and volatility — scalars or broadcastable float arrays,
        with ``s, k, t, sigma`` strictly positive.

    Returns
    -------
    tuple[FloatArray, FloatArray]
        The ``d1`` and ``d2`` arrays.
    """
    vol_sqrt_t = sigma * np.sqrt(t)
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return np.asarray(d1, dtype=np.float64), np.asarray(d2, dtype=np.float64)


def bs_price(
    s: FloatArray,
    k: FloatArray,
    t: FloatArray,
    r: FloatArray,
    q: FloatArray,
    sigma: FloatArray,
    *,
    kind: OptionKind = "call",
) -> FloatArray:
    r"""Black-Scholes-Merton price of a European option.

    For a call, ``C = S e^{-qT} N(d1) - K e^{-rT} N(d2)``; the put follows from
    put-call parity. Fully vectorized over broadcastable inputs.

    Parameters
    ----------
    s, k, t, r, q, sigma:
        Spot, strike, time-to-expiry (years), risk-free rate, continuous
        dividend yield, and volatility. ``s, k, t, sigma`` must be positive.
    kind:
        ``"call"`` (default) or ``"put"``.

    Returns
    -------
    FloatArray
        The option price(s), same broadcast shape as the inputs.

    Raises
    ------
    ValidationError
        If any of ``s, k, t, sigma`` is non-positive or ``kind`` is unknown.
    """
    _check_kind(kind)
    s, k, t, sigma = _validate_core(s, k, t, sigma)
    r = _as_float_array(r, name="r")
    q = _as_float_array(q, name="q")

    d1, d2 = _d1_d2(s, k, t, r, q, sigma)
    disc_s = s * np.exp(-q * t)
    disc_k = k * np.exp(-r * t)
    if kind == "call":
        price = disc_s * norm.cdf(d1) - disc_k * norm.cdf(d2)
    else:
        price = disc_k * norm.cdf(-d2) - disc_s * norm.cdf(-d1)
    return np.asarray(price, dtype=np.float64)


def bs_delta(
    s: FloatArray,
    k: FloatArray,
    t: FloatArray,
    r: FloatArray,
    q: FloatArray,
    sigma: FloatArray,
    *,
    kind: OptionKind = "call",
) -> FloatArray:
    r"""Black-Scholes delta ``\partial V / \partial S`` (dividend-adjusted).

    Parameters
    ----------
    s, k, t, r, q, sigma:
        Standard Black-Scholes parameters (see :func:`bs_price`).
    kind:
        ``"call"`` (default) or ``"put"``.

    Returns
    -------
    FloatArray
        The delta(s). Call delta is ``e^{-qT} N(d1)``; put delta is
        ``e^{-qT} (N(d1) - 1)``.

    Raises
    ------
    ValidationError
        If any of ``s, k, t, sigma`` is non-positive or ``kind`` is unknown.
    """
    _check_kind(kind)
    s, k, t, sigma = _validate_core(s, k, t, sigma)
    r = _as_float_array(r, name="r")
    q = _as_float_array(q, name="q")

    d1, _ = _d1_d2(s, k, t, r, q, sigma)
    disc = np.exp(-q * t)
    delta = disc * norm.cdf(d1) if kind == "call" else disc * (norm.cdf(d1) - 1.0)
    return np.asarray(delta, dtype=np.float64)


def bs_gamma(
    s: FloatArray,
    k: FloatArray,
    t: FloatArray,
    r: FloatArray,
    q: FloatArray,
    sigma: FloatArray,
) -> FloatArray:
    r"""Black-Scholes gamma ``\partial^2 V / \partial S^2`` (right-independent).

    Parameters
    ----------
    s, k, t, r, q, sigma:
        Standard Black-Scholes parameters (see :func:`bs_price`).

    Returns
    -------
    FloatArray
        The gamma(s), identical for calls and puts.

    Raises
    ------
    ValidationError
        If any of ``s, k, t, sigma`` is non-positive.
    """
    s, k, t, sigma = _validate_core(s, k, t, sigma)
    r = _as_float_array(r, name="r")
    q = _as_float_array(q, name="q")

    d1, _ = _d1_d2(s, k, t, r, q, sigma)
    gamma = np.exp(-q * t) * norm.pdf(d1) / (s * sigma * np.sqrt(t))
    return np.asarray(gamma, dtype=np.float64)


def bs_vega(
    s: FloatArray,
    k: FloatArray,
    t: FloatArray,
    r: FloatArray,
    q: FloatArray,
    sigma: FloatArray,
) -> FloatArray:
    r"""Black-Scholes vega ``\partial V / \partial \sigma`` (right-independent).

    Parameters
    ----------
    s, k, t, r, q, sigma:
        Standard Black-Scholes parameters (see :func:`bs_price`).

    Returns
    -------
    FloatArray
        The vega(s), per unit (1.00) of volatility, identical for calls and puts.

    Raises
    ------
    ValidationError
        If any of ``s, k, t, sigma`` is non-positive.
    """
    s, k, t, sigma = _validate_core(s, k, t, sigma)
    r = _as_float_array(r, name="r")
    q = _as_float_array(q, name="q")

    d1, _ = _d1_d2(s, k, t, r, q, sigma)
    vega = s * np.exp(-q * t) * norm.pdf(d1) * np.sqrt(t)
    return np.asarray(vega, dtype=np.float64)


def bs_theta(
    s: FloatArray,
    k: FloatArray,
    t: FloatArray,
    r: FloatArray,
    q: FloatArray,
    sigma: FloatArray,
    *,
    kind: OptionKind = "call",
) -> FloatArray:
    r"""Black-Scholes theta ``\partial V / \partial t`` (per year).

    Parameters
    ----------
    s, k, t, r, q, sigma:
        Standard Black-Scholes parameters (see :func:`bs_price`).
    kind:
        ``"call"`` (default) or ``"put"``.

    Returns
    -------
    FloatArray
        The theta(s), in price per year (divide by 365 for per-calendar-day).

    Raises
    ------
    ValidationError
        If any of ``s, k, t, sigma`` is non-positive or ``kind`` is unknown.
    """
    _check_kind(kind)
    s, k, t, sigma = _validate_core(s, k, t, sigma)
    r = _as_float_array(r, name="r")
    q = _as_float_array(q, name="q")

    d1, d2 = _d1_d2(s, k, t, r, q, sigma)
    disc_s = s * np.exp(-q * t)
    disc_k = k * np.exp(-r * t)
    term_gamma = -disc_s * norm.pdf(d1) * sigma / (2.0 * np.sqrt(t))
    if kind == "call":
        theta = term_gamma - r * disc_k * norm.cdf(d2) + q * disc_s * norm.cdf(d1)
    else:
        theta = term_gamma + r * disc_k * norm.cdf(-d2) - q * disc_s * norm.cdf(-d1)
    return np.asarray(theta, dtype=np.float64)


def bs_rho(
    s: FloatArray,
    k: FloatArray,
    t: FloatArray,
    r: FloatArray,
    q: FloatArray,
    sigma: FloatArray,
    *,
    kind: OptionKind = "call",
) -> FloatArray:
    r"""Black-Scholes rho ``\partial V / \partial r`` (per unit rate).

    Parameters
    ----------
    s, k, t, r, q, sigma:
        Standard Black-Scholes parameters (see :func:`bs_price`).
    kind:
        ``"call"`` (default) or ``"put"``.

    Returns
    -------
    FloatArray
        The rho(s), per unit (1.00) of the risk-free rate.

    Raises
    ------
    ValidationError
        If any of ``s, k, t, sigma`` is non-positive or ``kind`` is unknown.
    """
    _check_kind(kind)
    s, k, t, sigma = _validate_core(s, k, t, sigma)
    r = _as_float_array(r, name="r")
    q = _as_float_array(q, name="q")

    _, d2 = _d1_d2(s, k, t, r, q, sigma)
    disc_k = k * t * np.exp(-r * t)
    rho = disc_k * norm.cdf(d2) if kind == "call" else -disc_k * norm.cdf(-d2)
    return np.asarray(rho, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class BlackScholesGreeks:
    """The full first/second-order Greek set for a single option (or vector).

    Attributes
    ----------
    price:
        The Black-Scholes price.
    delta, gamma, vega, theta, rho:
        The analytic Greeks (see the corresponding ``bs_*`` functions).
    kind:
        The option right these Greeks were computed for.
    """

    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    kind: OptionKind = "call"

    @classmethod
    def compute(
        cls,
        s: float,
        k: float,
        t: float,
        r: float,
        q: float,
        sigma: float,
        *,
        kind: OptionKind = "call",
    ) -> BlackScholesGreeks:
        """Compute price and all Greeks for a single scalar contract.

        Parameters
        ----------
        s, k, t, r, q, sigma:
            Standard Black-Scholes parameters (see :func:`bs_price`).
        kind:
            ``"call"`` (default) or ``"put"``.

        Returns
        -------
        BlackScholesGreeks
            The frozen Greek bundle.

        Raises
        ------
        ValidationError
            If any of ``s, k, t, sigma`` is non-positive or ``kind`` is unknown.
        """
        _check_kind(kind)
        # Promote the scalar inputs to 0-D float64 arrays so the (array-typed)
        # vectorized kernels receive their declared ``FloatArray`` argument type.
        sa = _as_float_array(s, name="s")
        ka = _as_float_array(k, name="k")
        ta = _as_float_array(t, name="t")
        ra = _as_float_array(r, name="r")
        qa = _as_float_array(q, name="q")
        va = _as_float_array(sigma, name="sigma")
        return cls(
            price=float(bs_price(sa, ka, ta, ra, qa, va, kind=kind)),
            delta=float(bs_delta(sa, ka, ta, ra, qa, va, kind=kind)),
            gamma=float(bs_gamma(sa, ka, ta, ra, qa, va)),
            vega=float(bs_vega(sa, ka, ta, ra, qa, va)),
            theta=float(bs_theta(sa, ka, ta, ra, qa, va, kind=kind)),
            rho=float(bs_rho(sa, ka, ta, ra, qa, va, kind=kind)),
            kind=kind,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the Greek bundle."""
        d = asdict(self)
        d["kind"] = str(self.kind)
        for key in ("price", "delta", "gamma", "vega", "theta", "rho"):
            d[key] = float(d[key])
        return d


def _as_float_array(value: object, *, name: str) -> FloatArray:
    """Coerce ``value`` to a float64 array (helper for the vectorized kernels).

    Parameters
    ----------
    value:
        A scalar or array-like of option parameters.
    name:
        Human-readable label used in error messages.

    Returns
    -------
    FloatArray
        ``value`` as a float64 numpy array (at least 0-D).

    Raises
    ------
    ValidationError
        If ``value`` cannot be interpreted as float64 numeric data.
    """
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be numeric (float-coercible), got {value!r}.") from exc
    return arr


def _validate_core(
    s: FloatArray,
    k: FloatArray,
    t: FloatArray,
    sigma: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Coerce and domain-check the four strictly-positive Black-Scholes inputs.

    Parameters
    ----------
    s, k, t, sigma:
        Spot, strike, time-to-expiry (years), and volatility. Each must coerce to
        finite float64 data with every entry strictly positive.

    Returns
    -------
    tuple[FloatArray, FloatArray, FloatArray, FloatArray]
        The validated ``(s, k, t, sigma)`` arrays.

    Raises
    ------
    ValidationError
        If any input is non-numeric, contains NaN, or has a non-positive entry.
    """
    s_arr = _positive(_as_float_array(s, name="s"), name="s")
    k_arr = _positive(_as_float_array(k, name="k"), name="k")
    t_arr = _positive(_as_float_array(t, name="t"), name="t")
    sigma_arr = _positive(_as_float_array(sigma, name="sigma"), name="sigma")
    return s_arr, k_arr, t_arr, sigma_arr
