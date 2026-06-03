"""Tests for src.data — yfinance + Stooq fallback + cache key.

Network-hitting tests must be marked @pytest.mark.slow so the fast suite stays
offline. Pure-Python helpers (trading_day_stamp, ticker mapping) are tested
without network.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest

from src import data

# Resolve DataFetchError defensively so tests are still collectable if the
# module's exception class is renamed or temporarily missing during refactors.
DataFetchError = getattr(data, "DataFetchError", Exception)


# ---------------------------------------------------------------------------
# trading_day_stamp — boundary cases around the 21:00 UTC flip
# ---------------------------------------------------------------------------


def _make_fake_datetime(fixed_now: datetime):
    """Build a datetime stand-in whose .now(tz) returns `fixed_now`.

    Only .now is patched; date/timedelta/timezone are re-exported from the real
    datetime module so the rest of src.data keeps working.
    """

    class _FakeDateTime(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            if tz is None:
                return fixed_now.replace(tzinfo=None)
            return fixed_now.astimezone(tz)

    return _FakeDateTime


@pytest.mark.parametrize(
    "fixed_now, expected_offset_days",
    [
        # June 10 2025 is EDT (UTC-4). NYSE closes 16:00 ET = 20:00 UTC.
        # 19:59 UTC → ET 15:59, BEFORE close → yesterday's stamp.
        (datetime(2025, 6, 10, 19, 59, 0, tzinfo=UTC), -1),
        # 20:00 UTC EDT → ET 16:00 exactly, flips to today.
        (datetime(2025, 6, 10, 20, 0, 0, tzinfo=UTC), 0),
        # 21:00 UTC EDT → ET 17:00, comfortably after close → today.
        (datetime(2025, 6, 10, 21, 0, 0, tzinfo=UTC), 0),
        # 00:00 UTC on Jun 10 → ET 20:00 on Jun 9, after Jun 9's close →
        # the previous calendar day's stamp (matches "fixed_now.date() - 1").
        (datetime(2025, 6, 10, 0, 0, 0, tzinfo=UTC), -1),
        # December (EST, UTC-5): NYSE close 16:00 ET = 21:00 UTC.
        # 20:59 UTC → ET 15:59, BEFORE close → yesterday.
        (datetime(2025, 12, 10, 20, 59, 0, tzinfo=UTC), -1),
        # 21:00 UTC EST → ET 16:00 exactly, flips to today.
        (datetime(2025, 12, 10, 21, 0, 0, tzinfo=UTC), 0),
    ],
)
def test_trading_day_stamp_boundaries(monkeypatch, fixed_now, expected_offset_days):
    monkeypatch.setattr(data, "datetime", _make_fake_datetime(fixed_now))

    expected = (fixed_now.date() + timedelta(days=expected_offset_days)).isoformat()
    assert data.trading_day_stamp() == expected


# ---------------------------------------------------------------------------
# Ticker mapping — yfinance ".L" → Stooq ".UK"
# ---------------------------------------------------------------------------


def test_to_stooq_ticker_maps_london_suffix():
    mapper = getattr(data, "_to_stooq_ticker", None)
    if mapper is None:
        # NOTE: if the mapping helper is renamed/inlined, update this test to
        # cover whatever public surface replaces it.
        pytest.skip("ticker mapping not exposed as helper")

    assert mapper("VOD.L") == "VOD.UK"
    assert mapper("BP.L") == "BP.UK"
    # US tickers pass through unchanged.
    assert mapper("SPY") == "SPY"
    assert mapper("AAPL") == "AAPL"


# ---------------------------------------------------------------------------
# Slow integration tests (hit network)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_fetch_spy_recent():
    end = date.today()
    start = end - timedelta(days=30)

    df = data.fetch_ohlcv("SPY", start, end)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    for col in ("Open", "High", "Low", "Close", "Volume"):
        assert col in df.columns, f"missing column: {col}"
        assert not df[col].isna().all(), f"column {col} is entirely NaN"

    # Index should be monotonic ascending and tz-naive (matches yfinance contract).
    assert df.index.is_monotonic_increasing
    assert getattr(df.index, "tz", None) is None


@pytest.mark.slow
def test_fetch_invalid_ticker_raises_clean():
    # TODO: once src.data guarantees DataFetchError for all failure modes,
    # tighten `Exception` below to `DataFetchError`.
    expected = DataFetchError if DataFetchError is not Exception else Exception
    with pytest.raises(expected):
        data.fetch_ohlcv(
            "ZZZZ_NOT_A_REAL_TICKER_998877",
            date.today() - timedelta(days=30),
            date.today(),
        )


# ---------------------------------------------------------------------------
# Stooq CSV fallback — unit tests (no network) via monkeypatched urlopen
# ---------------------------------------------------------------------------


class _MockResp:
    """Minimal stand-in for the context-manager returned by urllib.urlopen."""

    def __init__(self, body: bytes):
        self._body = body

    def read(self, n: int | None = None) -> bytes:
        if n is None or n >= len(self._body):
            return self._body
        return self._body[:n]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patch_urlopen(monkeypatch, body: bytes) -> None:
    """Replace data.urllib.request.urlopen with a fake that yields `body`."""
    monkeypatch.setattr(
        data.urllib.request,
        "urlopen",
        lambda req, timeout=30: _MockResp(body),
    )


class TestStooqFallback:
    """Unit tests for _fetch_stooq_csv covering the failure-mode branches."""

    _START = date(2024, 1, 1)
    _END = date(2024, 2, 1)

    def test_html_response_triggers_data_fetch_error(self, monkeypatch):
        _patch_urlopen(monkeypatch, b"<html>rate limited</html>")
        with pytest.raises(DataFetchError) as exc_info:
            data._fetch_stooq_csv("spy", self._START, self._END)
        assert "HTML" in str(exc_info.value)

    def test_empty_body_triggers_data_fetch_error(self, monkeypatch):
        _patch_urlopen(monkeypatch, b"")
        with pytest.raises(DataFetchError):
            data._fetch_stooq_csv("spy", self._START, self._END)

    def test_header_only_csv_triggers_data_fetch_error(self, monkeypatch):
        _patch_urlopen(monkeypatch, b"Date,Open,High,Low,Close,Volume\n")
        with pytest.raises(DataFetchError) as exc_info:
            data._fetch_stooq_csv("spy", self._START, self._END)
        msg = str(exc_info.value).lower()
        assert "empty" in msg or "missing" in msg

    def test_malformed_csv_without_date_column(self, monkeypatch):
        _patch_urlopen(monkeypatch, b"foo,bar\n1,2\n")
        with pytest.raises(DataFetchError):
            data._fetch_stooq_csv("spy", self._START, self._END)

    def test_response_exceeding_size_cap(self, monkeypatch):
        # Build a body just over the 10 MB cap. Content is arbitrary —
        # the size-check fires before any CSV parsing happens.
        oversized = b"x" * (data._STOOQ_MAX_BYTES + 100)
        _patch_urlopen(monkeypatch, oversized)
        with pytest.raises(DataFetchError) as exc_info:
            data._fetch_stooq_csv("spy", self._START, self._END)
        assert "exceeded" in str(exc_info.value)

    def test_happy_path_returns_sorted_ohlcv_frame(self, monkeypatch):
        body = (
            b"Date,Open,High,Low,Close,Volume\n"
            b"2024-01-02,100,102,99,101,1000000\n"
            b"2024-01-03,101,103,100,102,1100000\n"
        )
        _patch_urlopen(monkeypatch, body)
        df = data._fetch_stooq_csv("spy", self._START, self._END)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert df.index.is_monotonic_increasing
        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert col in df.columns, f"missing column: {col}"


@pytest.mark.slow
def test_fallback_to_stooq_on_yfinance_failure(monkeypatch, caplog, fresh_cache):
    import yfinance as yf

    # Force the yfinance path to fail with a rate-limit error so the Stooq
    # fallback branch runs. YFRateLimitError lives under yf.exceptions in
    # recent versions; fall back to a generic Exception subclass otherwise.
    rate_limit_exc = getattr(
        getattr(yf, "exceptions", object()),
        "YFRateLimitError",
        RuntimeError,
    )

    class _BoomTicker:
        def __init__(self, *args, **kwargs):
            pass

        def history(self, *args, **kwargs):
            raise rate_limit_exc("simulated rate limit")

    monkeypatch.setattr(yf, "Ticker", _BoomTicker)

    # Bypass the retry backoff so the test stays fast.
    monkeypatch.setattr(data.time, "sleep", lambda *_a, **_kw: None)

    # Use a fresh cache key by picking a window unlikely to be cached.
    end = date.today()
    start = end - timedelta(days=30)

    with caplog.at_level(logging.INFO, logger="src.data"):
        df = data.fetch_ohlcv("SPY", start, end)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert any("falling back to Stooq" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# fetch_ohlcv unit tests — yfinance retry orchestration with isolated cache
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_cache(monkeypatch, tmp_path):
    """Swap _get_cache() for a per-test diskcache rooted in tmp_path.

    Prevents the real ``.yfcache`` from leaking into / being polluted by tests
    and gives each test a clean slate.
    """
    import diskcache

    fake = diskcache.Cache(str(tmp_path / ".yfcache"))
    monkeypatch.setattr(data, "_get_cache", lambda: fake)
    return fake


def _make_ohlcv_df(tz_aware: bool = False) -> pd.DataFrame:
    """Build a minimal valid OHLCV frame matching yfinance's shape."""
    idx = pd.date_range("2024-01-02", periods=3, freq="B")
    if tz_aware:
        idx = idx.tz_localize("America/New_York")
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [102.0, 103.0, 104.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [101.0, 102.0, 103.0],
            "Volume": [1_000_000, 1_100_000, 1_200_000],
        },
        index=idx,
    )


class _FakeTickerFactory:
    """Configurable yf.Ticker replacement; .history() runs `history_fn`."""

    def __init__(self, history_fn):
        self._history_fn = history_fn
        self.call_count = 0

    def __call__(self, *args, **kwargs):
        outer = self

        class _Ticker:
            def history(self_inner, *a, **kw):
                outer.call_count += 1
                return outer._history_fn(outer.call_count, *a, **kw)

        return _Ticker()


class TestFetchOhlcvCachePath:
    """Cache-hit short-circuit must skip yfinance entirely."""

    _START = date(2024, 1, 1)
    _END = date(2024, 1, 31)

    def test_cache_hit_returns_without_touching_yfinance(self, monkeypatch, fresh_cache):
        expected = _make_ohlcv_df()
        cache_key = f"{data._CACHE_VERSION}|TEST|{self._START.isoformat()}|{self._END.isoformat()}|{data.trading_day_stamp()}"
        fresh_cache.set(cache_key, expected)

        # If yfinance is touched, the test fails loudly — proves cache short-circuits.
        def _boom(*args, **kwargs):
            raise AssertionError("yf.Ticker should not be called on cache hit")

        monkeypatch.setattr(data.yf, "Ticker", _boom)

        got = data.fetch_ohlcv("TEST", self._START, self._END)
        pd.testing.assert_frame_equal(got, expected)

    def test_trading_day_stamp_included_in_cache_key(self, monkeypatch, fresh_cache):
        """Second call within the same trading day stamp hits the cache."""
        df_ok = _make_ohlcv_df(tz_aware=True)
        factory = _FakeTickerFactory(lambda n, *a, **kw: df_ok)
        monkeypatch.setattr(data.yf, "Ticker", factory)
        monkeypatch.setattr(data.time, "sleep", lambda *_a, **_kw: None)

        first = data.fetch_ohlcv("TEST", self._START, self._END)
        # Second call must NOT invoke yf.Ticker again — proves key reuse.
        monkeypatch.setattr(
            data.yf,
            "Ticker",
            lambda *a, **kw: (_ for _ in ()).throw(
                AssertionError("yf.Ticker called twice; cache key drifted")
            ),
        )
        second = data.fetch_ohlcv("TEST", self._START, self._END)
        pd.testing.assert_frame_equal(first, second)
        assert factory.call_count == 1


class TestFetchOhlcvYfinancePaths:
    """Exercise the retry/backoff orchestration and normalize round-trip."""

    _START = date(2024, 1, 1)
    _END = date(2024, 1, 31)

    def test_yfinance_success_first_try_strips_tz(self, monkeypatch, fresh_cache):
        """Happy path: tz-aware yfinance frame normalized to tz-naive."""
        df_ok = _make_ohlcv_df(tz_aware=True)
        assert df_ok.index.tz is not None  # sanity

        factory = _FakeTickerFactory(lambda n, *a, **kw: df_ok)
        monkeypatch.setattr(data.yf, "Ticker", factory)

        got = data.fetch_ohlcv("TEST", self._START, self._END)
        assert isinstance(got, pd.DataFrame)
        assert not got.empty
        assert got.index.tz is None  # _normalize_ohlcv stripped tz
        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert col in got.columns
        assert factory.call_count == 1  # no retries needed

    def test_empty_frame_retries_then_falls_back_to_stooq(self, monkeypatch, fresh_cache, caplog):
        """yf returns empty → exhaust 4 attempts → Stooq fallback succeeds."""
        empty = pd.DataFrame(columns=list(data._OHLCV_COLUMNS))
        factory = _FakeTickerFactory(lambda n, *a, **kw: empty)
        monkeypatch.setattr(data.yf, "Ticker", factory)

        # No-op sleep — exercises the backoff branch without blocking.
        sleep_calls: list[float] = []
        monkeypatch.setattr(data.time, "sleep", lambda s: sleep_calls.append(s))

        # Stooq returns a valid CSV.
        body = (
            b"Date,Open,High,Low,Close,Volume\n"
            b"2024-01-02,100,102,99,101,1000000\n"
            b"2024-01-03,101,103,100,102,1100000\n"
        )
        _patch_urlopen(monkeypatch, body)

        with caplog.at_level(logging.INFO, logger="src.data"):
            got = data.fetch_ohlcv("TEST", self._START, self._END)
        assert isinstance(got, pd.DataFrame)
        assert not got.empty
        assert factory.call_count == data._MAX_RETRIES
        # Backoff fires on every failed attempt EXCEPT the last (4 attempts → 3 sleeps).
        assert len(sleep_calls) == data._MAX_RETRIES - 1
        assert any("falling back to Stooq" in r.message for r in caplog.records)

    def test_all_retries_raise_then_stooq_html_raises_data_fetch_error(
        self, monkeypatch, fresh_cache
    ):
        """Every yf attempt raises AND Stooq returns HTML → DataFetchError."""

        def _raise(n, *a, **kw):
            raise RuntimeError(f"boom #{n}")

        factory = _FakeTickerFactory(_raise)
        monkeypatch.setattr(data.yf, "Ticker", factory)
        monkeypatch.setattr(data.time, "sleep", lambda *_a, **_kw: None)
        _patch_urlopen(monkeypatch, b"<html>rate limited</html>")

        with pytest.raises(DataFetchError) as exc_info:
            data.fetch_ohlcv("TEST", self._START, self._END)

        # fetch_ohlcv now collapses all fallback failure modes (including
        # Stooq-raised DataFetchError) into a single polished user-facing
        # message. The Stooq-source diagnostic detail goes to the server log
        # instead of leaking through the exception.
        msg = str(exc_info.value).lower()
        assert "could not fetch" in msg
        assert factory.call_count == data._MAX_RETRIES
