"""AQR public dataset loader for Betting-Against-Beta and Quality-Minus-Junk factors.

The AQR CDN periodically renames files and shuffles paths between the
``www.aqr.com`` and ``images.aqr.com`` hosts. If both ``BAB_URLS`` and
``QMJ_URLS`` candidates 404 the loader returns an empty DataFrame and emits
a ``UserWarning`` so the rest of the pipeline can degrade gracefully.

Users may monkey-patch ``BAB_URLS`` / ``QMJ_URLS`` at runtime if AQR moves
the assets again.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from platformdirs import user_cache_dir

BAB_URLS: tuple[str, ...] = (
    "https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Betting-Against-Beta-Equity-Factors-Monthly.xlsx",
    "https://images.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Betting-Against-Beta-Equity-Factors-Monthly.xlsx",
)
QMJ_URLS: tuple[str, ...] = (
    "https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Quality-Minus-Junk-Factors-Monthly.xlsx",
    "https://images.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Quality-Minus-Junk-Factors-Monthly.xlsx",
)

_HEADERS = {"User-Agent": "Mozilla/5.0"}
_TIMEOUT = 30
_CACHE_TTL = timedelta(days=25)
_BAB_SHEET = "BAB Factors"
_QMJ_SHEET = "QMJ Factors"
_SKIPROWS = 18
_US_COL = "USA"


def _cache_dir() -> Path:
    """Return the on-disk cache directory, creating it if missing."""
    p = Path(user_cache_dir("factorlab")) / "aqr"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cache_paths() -> tuple[Path, Path]:
    """Return the parquet and meta-json cache paths."""
    d = _cache_dir()
    return d / "bab_qmj.parquet", d / "bab_qmj.meta.json"


def _cache_fresh(meta_path: Path) -> bool:
    """Return True when meta exists and is younger than the TTL."""
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
        ts = datetime.fromisoformat(meta["fetched_at"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return False
    return datetime.now(timezone.utc) - ts < _CACHE_TTL


def _looks_like_excel(resp: requests.Response) -> bool:
    """Return True when the response Content-Type is an Excel spreadsheet."""
    ct = resp.headers.get("content-type", "").lower()
    return "spreadsheet" in ct or "excel" in ct or "officedocument" in ct


def _download(urls: tuple[str, ...]) -> bytes:
    """Try each URL in order and return the bytes of the first Excel response."""
    last_error: Exception | None = None
    for url in urls:
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        except requests.RequestException as exc:
            last_error = exc
            continue
        if resp.status_code == 200 and _looks_like_excel(resp):
            return resp.content
        last_error = RuntimeError(f"{url} returned {resp.status_code}")
    raise RuntimeError(f"all candidate URLs failed: {last_error}")


def _parse_sheet(content: bytes, sheet_name: str, factor: str) -> pd.Series:
    """Parse a single AQR workbook into a decimal monthly Series for the US column."""
    df = pd.read_excel(BytesIO(content), sheet_name=sheet_name, skiprows=_SKIPROWS)
    date_col = next((c for c in df.columns if str(c).strip().lower() in {"date", "dates"}), None)
    if date_col is None:
        raise ValueError(f"{factor}: no DATE column found in {sheet_name}")
    if _US_COL not in df.columns:
        raise ValueError(f"{factor}: USA column missing from {sheet_name}")
    df = df[[date_col, _US_COL]].dropna(subset=[date_col])
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col)
    series = pd.to_numeric(df[_US_COL], errors="coerce").dropna()
    series.index = series.index.to_period("M")
    series.name = factor
    median = series.abs().median()
    if pd.notna(median) and median > 0.5:
        raise ValueError(
            f"{factor}: median |value| {median:.3f} > 0.5 suggests percent units, expected decimal"
        )
    return series


def _empty_frame() -> pd.DataFrame:
    """Return the canonical empty result frame with the documented columns."""
    return pd.DataFrame(columns=["BAB", "QMJ"])


def load_aqr_factors(refresh: bool = False) -> pd.DataFrame:
    """Return monthly PeriodIndex DataFrame with US BAB and QMJ in decimal."""
    parquet_path, meta_path = _cache_paths()
    if not refresh and parquet_path.exists() and _cache_fresh(meta_path):
        try:
            cached = pd.read_parquet(parquet_path)
            cached.index = pd.PeriodIndex(cached.index, freq="M")
            return cached[["BAB", "QMJ"]]
        except (OSError, ValueError, KeyError):
            pass

    try:
        bab_bytes = _download(BAB_URLS)
        qmj_bytes = _download(QMJ_URLS)
        bab = _parse_sheet(bab_bytes, _BAB_SHEET, "BAB")
        qmj = _parse_sheet(qmj_bytes, _QMJ_SHEET, "QMJ")
        df = pd.concat([bab, qmj], axis=1, join="inner").dropna()
        df = df[["BAB", "QMJ"]]
        try:
            out = df.copy()
            out.index = out.index.astype(str)
            out.to_parquet(parquet_path)
            meta_path.write_text(
                json.dumps({"fetched_at": datetime.now(timezone.utc).isoformat()})
            )
        except (OSError, ValueError):
            pass
        return df
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"AQR fetch failed: {exc}; returning empty DataFrame", UserWarning, stacklevel=2
        )
        return _empty_frame()
