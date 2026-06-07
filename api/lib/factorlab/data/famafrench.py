"""Fama-French factor loader with on-disk cache and Dartmouth CSV fetcher."""

from __future__ import annotations

import io
import json
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from platformdirs import user_cache_dir

_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"

_FACTOR_URLS: dict[tuple[str, str], str] = {
    ("FF3", "M"): f"{_BASE}/F-F_Research_Data_Factors_CSV.zip",
    ("FF3", "D"): f"{_BASE}/F-F_Research_Data_Factors_daily_CSV.zip",
    ("FF5", "M"): f"{_BASE}/F-F_Research_Data_5_Factors_2x3_CSV.zip",
    ("FF5", "D"): f"{_BASE}/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
}

_MOM_URLS: dict[str, str] = {
    "M": f"{_BASE}/F-F_Momentum_Factor_CSV.zip",
    "D": f"{_BASE}/F-F_Momentum_Factor_daily_CSV.zip",
}

_ZIP_MAGIC = b"PK\x03\x04"
_NA_TOKENS = ["-99.99", "-999", "-99.99 ", " -99.99", "-999.00"]
_CACHE_TTL_DAYS = 25
_TIMEOUT = 30


def _cache_root() -> Path:
    """Return the platform-specific cache directory for factorlab/famafrench."""
    root = Path(user_cache_dir("factorlab")) / "famafrench"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _download_zip(url: str) -> bytes:
    """Fetch a zip from Dartmouth with one retry; verify magic bytes."""
    last_err: Exception | None = None
    for attempt in range(2):
        try:
            resp = requests.get(url, timeout=_TIMEOUT)
            resp.raise_for_status()
            payload = resp.content
            if not payload.startswith(_ZIP_MAGIC):
                raise ValueError(f"Not a zip archive: {url}")
            return payload
        except Exception as err:
            last_err = err
            if attempt == 0:
                time.sleep(2)
    assert last_err is not None
    raise last_err


def _unzip_single_csv(payload: bytes) -> str:
    """Return the text contents of the single CSV inside the zip."""
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError("Zip contains no CSV file")
        with zf.open(names[0]) as fh:
            return fh.read().decode("latin-1")


def _truncate_at_blank(text: str) -> str:
    """Keep only the first table; Dartmouth monthly files append an annual table."""
    lines = text.splitlines()
    # Skip 3 header lines, then collect rows until a blank or non-numeric first token.
    out = lines[:3]
    in_data = False
    for line in lines[3:]:
        stripped = line.strip()
        first = stripped.split(",", 1)[0].strip()
        is_numeric = first.isdigit()
        if not in_data:
            # We accept the header line (non-numeric, non-blank) as start of table.
            if stripped and not is_numeric:
                out.append(line)
                in_data = True
                continue
            if is_numeric:
                out.append(line)
                in_data = True
                continue
            out.append(line)
            continue
        if not stripped or not is_numeric:
            break
        out.append(line)
    return "\n".join(out)


def _parse_csv(text: str, freq: str) -> pd.DataFrame:
    """Parse the trimmed Dartmouth CSV text into a DataFrame indexed by period or date."""
    trimmed = _truncate_at_blank(text)
    buf = io.StringIO(trimmed)
    df = pd.read_csv(
        buf,
        skiprows=3,
        index_col=0,
        na_values=_NA_TOKENS,
        skipinitialspace=True,
    )
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(how="all")
    df.index = df.index.astype(str).str.strip()
    df = df[df.index.str.isdigit()]
    if freq == "M":
        df.index = pd.PeriodIndex(df.index, freq="M")
    else:
        df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df / 100.0
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Assert canonical Fama-French column names and standardize MOM if present."""
    if "Mkt-RF" not in df.columns:
        raise ValueError(f"Missing 'Mkt-RF' column; got {list(df.columns)}")
    if "RF" not in df.columns:
        raise ValueError(f"Missing 'RF' column; got {list(df.columns)}")
    rename: dict[str, str] = {}
    for c in df.columns:
        cl = c.strip()
        if cl.lower() in {"mom", "mom   ", "wml"}:
            rename[c] = "MOM"
    if rename:
        df = df.rename(columns=rename)
    return df


def _fetch_factors(model: str, freq: str) -> pd.DataFrame:
    """Download and parse the factor file for (model, freq)."""
    url = _FACTOR_URLS[(model, freq)]
    text = _unzip_single_csv(_download_zip(url))
    return _parse_csv(text, freq)


def _fetch_momentum(freq: str) -> pd.DataFrame:
    """Download and parse the momentum file for freq."""
    url = _MOM_URLS[freq]
    text = _unzip_single_csv(_download_zip(url))
    df = _parse_csv(text, freq)
    df.columns = ["MOM" for _ in df.columns]
    return df[["MOM"]]


def _cache_is_fresh(meta_path: Path) -> bool:
    """Return True when the meta file exists and is within TTL."""
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        fetched = datetime.strptime(meta["fetched_at"], "%Y-%m-%d").date()
    except (ValueError, KeyError, json.JSONDecodeError):
        return False
    return (date.today() - fetched) <= timedelta(days=_CACHE_TTL_DAYS)


def _write_cache(df: pd.DataFrame, parquet_path: Path, meta_path: Path) -> None:
    """Persist DataFrame and freshness sentinel."""
    df.to_parquet(parquet_path)
    meta_path.write_text(
        json.dumps({"fetched_at": date.today().strftime("%Y-%m-%d")}),
        encoding="utf-8",
    )


def _read_cache(parquet_path: Path, freq: str) -> pd.DataFrame:
    """Read the cached parquet and restore the period index for monthly data."""
    df = pd.read_parquet(parquet_path)
    if freq == "M" and not isinstance(df.index, pd.PeriodIndex):
        df.index = pd.PeriodIndex(df.index.astype(str), freq="M")
    return df


def _fallback_datareader(model: str, freq: str) -> pd.DataFrame:
    """Last-resort fetch via pandas_datareader, adapted to canonical schema."""
    from pandas_datareader import data as pdr  # local import keeps optional dep lazy

    if model == "FF3":
        ds = "F-F_Research_Data_Factors" if freq == "M" else "F-F_Research_Data_Factors_daily"
    else:
        ds = (
            "F-F_Research_Data_5_Factors_2x3"
            if freq == "M"
            else "F-F_Research_Data_5_Factors_2x3_daily"
        )
    raw = pdr.DataReader(ds, "famafrench")[0]
    raw = raw / 100.0
    mom_ds = "F-F_Momentum_Factor" if freq == "M" else "F-F_Momentum_Factor_daily"
    mom = pdr.DataReader(mom_ds, "famafrench")[0] / 100.0
    mom.columns = ["MOM"]
    merged = raw.join(mom, how="inner")
    if freq == "M" and not isinstance(merged.index, pd.PeriodIndex):
        merged.index = pd.PeriodIndex(merged.index.astype(str), freq="M")
    return merged


def load_ff_factors(
    model: str = "FF5",
    freq: str = "M",
    refresh: bool = False,
) -> pd.DataFrame:
    """Load Fama-French factors with momentum appended, returned in decimal form.

    model in {'FF3', 'FF5'} selects which CSV; momentum is appended automatically.
    freq in {'M','D'}.
    """
    if model not in {"FF3", "FF5"}:
        raise ValueError(f"model must be 'FF3' or 'FF5', got {model!r}")
    if freq not in {"M", "D"}:
        raise ValueError(f"freq must be 'M' or 'D', got {freq!r}")

    cache_dir = _cache_root()
    parquet_path = cache_dir / f"{model}_{freq}.parquet"
    meta_path = cache_dir / f"{model}_{freq}_meta.json"

    if not refresh and parquet_path.exists() and _cache_is_fresh(meta_path):
        return _read_cache(parquet_path, freq)

    try:
        factors = _fetch_factors(model, freq)
        mom = _fetch_momentum(freq)
        merged = factors.join(mom, how="inner")
    except Exception:
        merged = _fallback_datareader(model, freq)

    merged = _normalize_columns(merged)

    expected = ["Mkt-RF", "SMB", "HML"]
    if model == "FF5":
        expected += ["RMW", "CMA"]
    expected += ["MOM", "RF"]
    missing = [c for c in expected if c not in merged.columns]
    if missing:
        raise ValueError(f"Missing expected columns after merge: {missing}")
    merged = merged[expected]

    # Catches the percent-not-divided bug: real factor medians are << 5% in decimal.
    if merged.drop(columns="RF").abs().median().max() >= 0.05:
        raise AssertionError(
            "Factor magnitudes look like percent, not decimal; conversion likely skipped"
        )

    _write_cache(merged, parquet_path, meta_path)
    return merged
