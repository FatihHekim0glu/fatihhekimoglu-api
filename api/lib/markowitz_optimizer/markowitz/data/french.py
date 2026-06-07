"""Loader for Ken French factor datasets.

The Ken French data library distributes ZIP files containing CSV factor
returns quoted in *percent*. We download via :mod:`urllib`, parse the
CSV (skipping any header text and copyright footers), and divide by
100 so the returned frame is in decimal returns.
"""

from __future__ import annotations

import csv
import io
import os
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Final

import pandas as pd

from .exceptions import DataIntegrityError, EmptyDataError, ProviderUnavailableError

_BASE_URL: Final = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"

_FREQUENCY_SUFFIX: Final[dict[str, str]] = {
    "monthly": "_CSV.zip",
    "daily": "_daily_CSV.zip",
}


def _build_url(dataset: str, frequency: str) -> str:
    try:
        suffix = _FREQUENCY_SUFFIX[frequency]
    except KeyError as exc:
        raise ValueError(
            f"unknown frequency {frequency!r}; expected one of {sorted(_FREQUENCY_SUFFIX)}"
        ) from exc
    return f"{_BASE_URL}/{dataset}{suffix}"


def _parse_french_csv(text: str, *, frequency: str) -> pd.DataFrame:
    """Parse the body of a Ken French CSV into a numeric DataFrame.

    The files contain one or more *copyright* / *header* blocks. We walk
    rows, identifying the header line as the first one whose first cell
    is empty (the date column has no name in French's CSVs).
    """
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)

    header_idx: int | None = None
    for i, row in enumerate(rows):
        if row and row[0].strip() == "" and len(row) > 1 and any(c.strip() for c in row[1:]):
            header_idx = i
            break

    if header_idx is None:
        raise DataIntegrityError("could not locate header row in French CSV")

    header = rows[header_idx]
    columns = [c.strip() for c in header[1:]]

    records: list[tuple[str, list[str]]] = []
    for row in rows[header_idx + 1 :]:
        if not row or not row[0].strip():
            # blank row signals end of first table (annual data follows)
            break
        first = row[0].strip()
        if not first.replace("-", "").isdigit():
            break
        records.append((first, [cell.strip() for cell in row[1 : 1 + len(columns)]]))

    if not records:
        raise EmptyDataError("French CSV contained no data rows")

    dates: list[pd.Timestamp]
    if frequency == "daily":
        dates = [pd.Timestamp(s) for s in (rec[0] for rec in records)]
    else:
        # monthly: YYYYMM
        dates = [pd.Timestamp(f"{s[:4]}-{s[4:6]}-01") for s in (rec[0] for rec in records)]

    data: dict[str, list[float]] = {col: [] for col in columns}
    for _, values in records:
        for col, cell in zip(columns, values, strict=False):
            data[col].append(float(cell))

    frame = pd.DataFrame(data, index=pd.DatetimeIndex(dates))
    frame = frame / 100.0
    frame.index.name = "date"
    assert isinstance(frame, pd.DataFrame)
    return frame


def load_french_factors(
    dataset: str,
    *,
    frequency: str = "monthly",
    cache_root: str | os.PathLike[str] | None = None,
) -> pd.DataFrame:
    """Download (or read from cache) a Ken French factor table.

    Parameters
    ----------
    dataset:
        Dataset identifier, e.g. ``"F-F_Research_Data_Factors"``.
    frequency:
        ``"monthly"`` (default) or ``"daily"``.
    cache_root:
        Optional directory used to cache the raw ZIP payload.
    """
    url = _build_url(dataset, frequency)
    cache_dir = Path(cache_root) if cache_root is not None else None
    cache_file: Path | None = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{dataset}_{frequency}.zip"

    payload: bytes
    if cache_file is not None and cache_file.exists():
        payload = cache_file.read_bytes()
    else:
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                payload = resp.read()
        except urllib.error.URLError as exc:
            raise ProviderUnavailableError(
                f"Kenneth French data library unreachable ({url}): {exc.reason}"
            ) from exc
        except Exception as exc:
            raise ProviderUnavailableError(
                f"Failed to download French factors from {url}: {exc}"
            ) from exc
        if cache_file is not None:
            cache_file.write_bytes(payload)

    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                raise DataIntegrityError(f"no CSV in French archive {dataset}")
            text = zf.read(csv_names[0]).decode("latin-1")
    except zipfile.BadZipFile as exc:
        raise DataIntegrityError(f"French archive {dataset} is corrupt") from exc

    return _parse_french_csv(text, frequency=frequency)
