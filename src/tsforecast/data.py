"""Data loading utilities.

The primary demo series is the Mauna Loa atmospheric CO2 record, which ships
with statsmodels and requires no network access. Additional macro series can
be downloaded from FRED with ``scripts/download_data.py`` and loaded here via
:func:`load_fred_csv`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_SAMPLE_PATH = Path(__file__).resolve().parents[2] / "data" / "co2_sample.csv"


def load_co2_sample(path: str | Path | None = None) -> pd.Series:
    """Load the committed monthly CO2 sample CSV, no network or statsmodels data needed.

    This is the offline default for the CLI and the demo. The file ships in the
    repository at ``data/co2_sample.csv`` and holds the full monthly Mauna Loa
    series carved from the public statsmodels ``co2`` dataset (public NOAA data).

    Args:
        path: Optional override path to a two-column ``date,co2`` CSV. Defaults
            to the bundled ``data/co2_sample.csv``.

    Returns:
        Monthly CO2 concentration (ppm) as a pandas Series with a monthly
        ``DatetimeIndex`` and name ``"co2"``.

    Raises:
        FileNotFoundError: If the sample CSV cannot be found.
    """
    csv_path = Path(path) if path is not None else _SAMPLE_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. It ships with the repo; "
            "regenerate it with scripts/make_sample.py."
        )
    frame = pd.read_csv(csv_path, parse_dates=["date"])
    series = pd.Series(frame["co2"].to_numpy(dtype=float), index=frame["date"])
    series.index = pd.DatetimeIndex(series.index, freq="MS")
    series.name = "co2"
    return series


def load_co2_monthly() -> pd.Series:
    """Load the Mauna Loa CO2 series bundled with statsmodels, as monthly means.

    The raw dataset is weekly with occasional gaps. It is resampled to monthly
    means and any remaining missing months are linearly interpolated so the
    result is a regular, complete monthly series suitable for SARIMAX.

    Returns:
        Monthly CO2 concentration (ppm) as a pandas Series with a monthly
        ``DatetimeIndex`` and name ``"co2"``.
    """
    import statsmodels.api as sm

    raw = sm.datasets.co2.load_pandas().data["co2"]
    monthly = raw.resample("MS").mean().interpolate(method="linear")
    monthly.name = "co2"
    return monthly


def load_fred_csv(path: str | Path, column: str | None = None) -> pd.Series:
    """Load a FRED CSV file downloaded by ``scripts/download_data.py``.

    FRED CSV exports have a date column (``DATE`` or ``observation_date``)
    followed by one value column. Missing observations are exported as ``"."``
    and are parsed as NaN, then dropped.

    Args:
        path: Path to the CSV file.
        column: Name of the value column to use. Defaults to the first
            non-date column.

    Returns:
        The series as a pandas Series indexed by date, NaNs dropped.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If no date column or requested value column is found.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/download_data.py to fetch FRED series."
        )
    frame = pd.read_csv(path, na_values=["."])
    date_col = next(
        (c for c in frame.columns if c.strip().lower() in {"date", "observation_date"}), None
    )
    if date_col is None:
        raise ValueError(f"no date column found in {path}; columns: {list(frame.columns)}")
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.set_index(date_col)
    if column is None:
        value_cols = [c for c in frame.columns]
        if not value_cols:
            raise ValueError(f"no value column found in {path}")
        column = value_cols[0]
    elif column not in frame.columns:
        raise ValueError(f"column {column!r} not in {path}; columns: {list(frame.columns)}")
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    series.name = column
    return series
