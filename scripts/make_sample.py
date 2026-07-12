"""Regenerate the committed CO2 sample CSV from the statsmodels bundled dataset.

The Mauna Loa atmospheric CO2 record is public NOAA data and ships with
statsmodels. This script resamples it to a regular monthly series and writes
``data/co2_sample.csv`` so the demo and tests run fully offline without loading
the statsmodels dataset. Run it only when you want to refresh the committed
sample; the CSV is already in the repository.

Usage:
    python scripts/make_sample.py [--out data/co2_sample.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tsforecast.data import load_co2_monthly


def main() -> None:
    """Write the monthly CO2 series to a two-column date,co2 CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/co2_sample.csv"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    series = load_co2_monthly()
    frame = series.rename("co2").to_frame()
    frame.index.name = "date"
    frame.to_csv(args.out, date_format="%Y-%m-%d", float_format="%.4f")
    print(
        f"wrote {args.out}: {len(series)} monthly observations, "
        f"{series.index[0]:%Y-%m} to {series.index[-1]:%Y-%m}"
    )


if __name__ == "__main__":
    main()
