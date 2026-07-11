"""Download optional public macro series from FRED as CSV files.

The core demo and tests use the CO2 series bundled with statsmodels and do
not need these files. This script fetches monthly CPI (CPIAUCSL) and the
10-year breakeven inflation rate (T10YIE) from the public FRED CSV endpoint
into ``data/``. If FRED is unreachable, it prints a message and exits
gracefully.

Usage:
    python scripts/download_data.py [--out-dir data]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
DEFAULT_SERIES = ("CPIAUCSL", "T10YIE")


def download_series(series_id: str, out_dir: Path, timeout: float = 30.0) -> Path | None:
    """Download one FRED series as CSV.

    Args:
        series_id: FRED series identifier, e.g. "CPIAUCSL".
        out_dir: Directory to write ``<series_id>.csv`` into.
        timeout: Request timeout in seconds.

    Returns:
        Path to the written file, or None if the download failed.
    """
    url = FRED_CSV_URL.format(series_id=series_id)
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"  {series_id}: download failed ({exc}). Skipping.")
        return None
    text = response.text
    if not text.lstrip().lower().startswith(("date", "observation_date")):
        print(f"  {series_id}: unexpected response format. Skipping.")
        return None
    out_path = out_dir / f"{series_id}.csv"
    out_path.write_text(text, encoding="utf-8")
    n_rows = max(text.count("\n") - 1, 0)
    print(f"  {series_id}: wrote {out_path} ({n_rows} observations)")
    return out_path


def main() -> int:
    """Download all default FRED series.

    Returns:
        Process exit code, 0 even when downloads fail since they are optional.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data"), help="output directory")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading optional FRED series (the core demo does not need these):")
    results = [download_series(sid, args.out_dir) for sid in DEFAULT_SERIES]
    if not any(results):
        print(
            "No series downloaded. FRED may be unreachable from this network. "
            "The bundled CO2 demo still works offline."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
