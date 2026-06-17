"""Live option-chain loader (yfinance) with risk-free / dividend alignment.

Fetches the current listed option chain for a ticker via ``yfinance`` (the
``[data]`` extra), normalizes it into the project's canonical panel
(:data:`nnvsbs.data.synthetic.CHAIN_COLUMNS`), computes a mid price from
bid/ask, and aligns the risk-free rate and dividend yield. The ``quote_date``
column is stamped so the leakage-free, group-by-quote-date split has well-defined
groups.

KNOWN BLIND SPOT (documented in the README Limitations): yfinance exposes only the
*current* chain — there is no historical option-quote archive — so the real-chain
smile study is a point-in-time snapshot, not a panel over quote dates. A
point-in-time underlying (PIT) hook is provided so a paid source (e.g. Polygon.io
Options) can supply as-of underlyings later; with yfinance only the live spot is
available.

``yfinance``/``pandas`` are imported LAZILY inside the loader so importing this
module pulls in no network or data dependency. Importing this module has no side
effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nnvsbs._exceptions import ValidationError

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

#: Documented flat fallbacks used when the caller supplies no rate / dividend yield.
#: They are deliberately simple constants (not a live curve) — the README Limitations
#: call this out, and a paid source can supply an aligned curve via the ``r``/``q`` args.
_DEFAULT_R: float = 0.03
_DEFAULT_Q: float = 0.0

#: Calendar days per year, used to convert an expiry-date delta to year fractions.
_DAYS_PER_YEAR: float = 365.0


def load_option_chain(
    ticker: str = "SPY",
    *,
    r: float | None = None,
    q: float | None = None,
    max_expiries: int = 6,
    pit_spot: float | None = None,
) -> tuple[pd.DataFrame, str]:
    """Load and normalize the current listed option chain for ``ticker``.

    Pulls every listed expiry (up to ``max_expiries``) via ``yfinance``, computes
    a bid/ask mid price, derives time-to-expiry from the quote date, and emits the
    canonical panel with a stamped ``quote_date``. The risk-free rate and dividend
    yield default to flat fallbacks when not supplied.

    POINT-IN-TIME (PIT) HOOK: pass ``pit_spot`` to override the live underlying with
    an as-of spot (for a paid historical source); with ``None`` the live spot from
    yfinance is used (the yfinance current-chain blind spot).

    Parameters
    ----------
    ticker:
        Underlying symbol (default ``"SPY"``).
    r:
        Annualized risk-free rate; ``None`` => a documented flat fallback.
    q:
        Continuous dividend yield; ``None`` => derived from yfinance / flat fallback.
    max_expiries:
        Cap on the number of listed expiries to pull (default ``6``).
    pit_spot:
        Optional as-of underlying spot overriding the live quote (PIT hook).

    OFFLINE DEGRADATION: when ``yfinance`` is not installed or the live fetch yields
    no usable rows (no network / no API access — the reality of this environment),
    the loader falls back to a seeded synthetic Black-Scholes chain and returns the
    ``"synthetic"`` data-source tag, so callers always receive the canonical panel.

    Parameters
    ----------
    ticker:
        Underlying symbol (default ``"SPY"``).
    r:
        Annualized risk-free rate; ``None`` => a documented flat fallback.
    q:
        Continuous dividend yield; ``None`` => a documented flat fallback.
    max_expiries:
        Cap on the number of listed expiries to pull (default ``6``).
    pit_spot:
        Optional as-of underlying spot overriding the live quote (PIT hook).

    Returns
    -------
    tuple[pandas.DataFrame, str]
        The canonical option panel and its ``data_source`` tag (``"yfinance"`` when
        a live chain was loaded, ``"synthetic"`` when degraded offline).

    Raises
    ------
    ValidationError
        If ``ticker`` is empty/blank or ``max_expiries`` is non-positive.
    """
    if not ticker or not ticker.strip():
        raise ValidationError("ticker must be a non-empty symbol.")
    if max_expiries <= 0:
        raise ValidationError(f"max_expiries must be positive, got {max_expiries}.")

    rate = _DEFAULT_R if r is None else float(r)
    div = _DEFAULT_Q if q is None else float(q)

    try:
        frame = _fetch_live_chain(
            ticker.strip(),
            r=rate,
            q=div,
            max_expiries=max_expiries,
            pit_spot=pit_spot,
        )
    except Exception:  # any live-fetch failure (offline/network) degrades to synthetic
        frame = None

    if frame is None or frame.empty:
        return _synthetic_fallback(r=rate, q=div), "synthetic"
    return frame, "yfinance"


def _fetch_live_chain(
    ticker: str,
    *,
    r: float,
    q: float,
    max_expiries: int,
    pit_spot: float | None,
) -> pd.DataFrame | None:
    """Fetch and normalize the live option chain into the canonical panel.

    Lazily imports ``yfinance`` and ``pandas`` so importing this module pulls in no
    network/data dependency. Returns ``None`` (signalling an offline degradation)
    when the ticker has no listed expiries; otherwise concatenates each expiry's
    calls and puts into the canonical :data:`CHAIN_COLUMNS` panel.

    Parameters
    ----------
    ticker:
        Underlying symbol (already stripped, non-empty).
    r, q:
        Aligned risk-free rate and dividend yield to stamp on every row.
    max_expiries:
        Cap on the number of listed expiries to pull.
    pit_spot:
        Optional as-of underlying spot overriding the live quote (PIT hook).

    Returns
    -------
    pandas.DataFrame | None
        The canonical panel, or ``None`` when no usable chain is available.
    """
    import pandas as pd
    import yfinance as yf

    from nnvsbs.data.synthetic import CHAIN_COLUMNS

    tk = yf.Ticker(ticker)
    expiries = list(getattr(tk, "options", ()) or ())
    if not expiries:
        return None

    spot = _resolve_spot(tk, pit_spot=pit_spot)
    quote_date = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)

    blocks: list[pd.DataFrame] = []
    for expiry in expiries[:max_expiries]:
        try:
            chain = tk.option_chain(expiry)
        except Exception:  # skip a single unfetchable expiry, keep the rest
            continue
        expiry_ts = pd.Timestamp(expiry)
        t_years = max((expiry_ts - quote_date).days, 1) / _DAYS_PER_YEAR
        for kind, leg in (("call", chain.calls), ("put", chain.puts)):
            block = _normalize_leg(
                leg,
                spot=spot,
                t_years=t_years,
                r=r,
                q=q,
                kind=kind,
                quote_date=quote_date,
            )
            if block is not None and not block.empty:
                blocks.append(block)

    if not blocks:
        return None

    frame = pd.concat(blocks, ignore_index=True)[list(CHAIN_COLUMNS)]
    return frame.reset_index(drop=True)


def _normalize_leg(
    leg: pd.DataFrame,
    *,
    spot: float,
    t_years: float,
    r: float,
    q: float,
    kind: str,
    quote_date: pd.Timestamp,
) -> pd.DataFrame | None:
    """Normalize one expiry's call/put leg into a canonical-panel block.

    Computes a robust mid price, carries the strike, stamps the shared spot /
    expiry / rate / dividend / quote-date, and tags the option right. Rows with a
    non-positive or non-finite mid (dead/one-sided quotes) are dropped. ``sigma`` is
    filled with the leg's reported implied vol when present purely as carried
    metadata — it is NEVER a model feature (the leakage guard lives downstream).

    Parameters
    ----------
    leg:
        The raw calls or puts frame from ``yfinance``.
    spot, t_years, r, q:
        Shared contract context for this expiry.
    kind:
        ``"call"`` or ``"put"``.
    quote_date:
        The (tz-naive) quote-date stamp shared by every contract in this snapshot.

    Returns
    -------
    pandas.DataFrame | None
        The normalized block, or ``None`` if the leg has no usable rows.
    """
    import numpy as np
    import pandas as pd

    if leg is None or len(leg) == 0 or "strike" not in leg:
        return None

    bid = leg.get("bid", pd.Series(dtype="float64"))
    ask = leg.get("ask", pd.Series(dtype="float64"))
    last = leg.get("lastPrice", pd.Series(dtype="float64"))
    mid = _mid_price(bid, ask, last)

    sigma = leg.get("impliedVolatility", pd.Series(index=leg.index, dtype="float64"))

    block = pd.DataFrame(
        {
            "quote_date": quote_date,
            "S": float(spot),
            "K": pd.to_numeric(leg["strike"], errors="coerce").astype("float64"),
            "T": float(t_years),
            "r": float(r),
            "q": float(q),
            "sigma": pd.to_numeric(sigma, errors="coerce").astype("float64"),
            "kind": kind,
            "price": pd.to_numeric(mid, errors="coerce").astype("float64"),
        }
    )

    block = block[np.isfinite(block["price"]) & (block["price"] > 0.0)]
    block = block[np.isfinite(block["K"]) & (block["K"] > 0.0)]
    cleaned: pd.DataFrame = block.reset_index(drop=True)
    return cleaned


def _resolve_spot(tk: object, *, pit_spot: float | None) -> float:
    """Return the underlying spot, honoring the point-in-time (PIT) override.

    When ``pit_spot`` is supplied it wins (a paid as-of source); otherwise the live
    spot is read from yfinance's ``fast_info`` / ``info``. A non-positive or missing
    live spot raises so the caller degrades to the synthetic fallback.

    Parameters
    ----------
    tk:
        A ``yfinance.Ticker`` (typed loosely to keep this module yfinance-free at
        import time).
    pit_spot:
        Optional as-of underlying spot overriding the live quote.

    Returns
    -------
    float
        The resolved positive spot price.

    Raises
    ------
    ValidationError
        If no positive live spot is available and ``pit_spot`` is ``None``.
    """
    if pit_spot is not None:
        if pit_spot <= 0.0:
            raise ValidationError(f"pit_spot must be positive, got {pit_spot}.")
        return float(pit_spot)

    fast = getattr(tk, "fast_info", None)
    candidate: object = None
    if fast is not None:
        candidate = getattr(fast, "last_price", None)
        if candidate is None:
            try:
                candidate = fast["lastPrice"]
            except Exception:  # mapping-style access may be absent on fast_info
                candidate = None
    if candidate is None:
        info = getattr(tk, "info", {}) or {}
        candidate = info.get("regularMarketPrice") or info.get("previousClose")

    spot = _coerce_positive_spot(candidate)
    if spot is None:
        raise ValidationError("could not resolve a positive live spot for the ticker.")
    return spot


def _coerce_positive_spot(value: object) -> float | None:
    """Coerce a loosely-typed spot candidate to a positive float, or ``None``.

    Parameters
    ----------
    value:
        A spot candidate pulled from ``fast_info``/``info`` (any object).

    Returns
    -------
    float | None
        The positive float spot, or ``None`` when missing/non-numeric/non-positive.
    """
    if value is None:
        return None
    try:
        spot = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return spot if spot > 0.0 else None


def _synthetic_fallback(*, r: float, q: float) -> pd.DataFrame:
    """Build a seeded synthetic canonical panel for the offline degradation path.

    Delegates to :func:`nnvsbs.data.synthetic.generate_synthetic_chain` (so the
    fallback shares the project's single Black-Scholes labeler and seeding) and
    returns just the panel, carrying the requested ``r``/``q`` so the fallback's
    rate/dividend match what the caller asked for.

    Parameters
    ----------
    r, q:
        Risk-free rate and dividend yield to thread into the synthetic chain.

    Returns
    -------
    pandas.DataFrame
        The canonical synthetic option panel.
    """
    from nnvsbs.data.synthetic import generate_synthetic_chain

    return generate_synthetic_chain(r=r, q=q).frame


def _mid_price(bid: pd.Series, ask: pd.Series, last: pd.Series) -> pd.Series:
    """Return a robust mid price from bid/ask, falling back to last trade.

    Uses ``(bid + ask) / 2`` where both sides are positive, else the last traded
    price — so a one-sided or stale quote does not produce a zero/NaN mid.

    Parameters
    ----------
    bid, ask, last:
        Bid, ask, and last-trade price columns from the raw chain.

    Returns
    -------
    pandas.Series
        The per-contract mid price.
    """
    import numpy as np
    import pandas as pd

    bid_f = pd.to_numeric(bid, errors="coerce").astype("float64")
    ask_f = pd.to_numeric(ask, errors="coerce").astype("float64")
    last_f = pd.to_numeric(last, errors="coerce").astype("float64")

    two_sided = (bid_f > 0.0) & (ask_f > 0.0)
    mid = (bid_f + ask_f) / 2.0
    # Where the two-sided mid is unavailable, fall back to the last traded price so a
    # one-sided or stale quote does not silently produce a zero/NaN price.
    return pd.Series(np.where(two_sided, mid, last_f), index=bid_f.index, dtype="float64")
