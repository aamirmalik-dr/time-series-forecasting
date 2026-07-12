# Data

This directory is gitignored except for this file and the committed sample
`co2_sample.csv`.

## Committed sample (no download needed)

`co2_sample.csv` is the full monthly Mauna Loa atmospheric CO2 record, 526
observations from 1958-03 to 2001-12, as a two-column `date,co2` CSV. It is a
carved subset of the public NOAA record that ships with statsmodels
(`statsmodels.datasets.co2`), resampled to a regular monthly grid. It is a
carved public subset, not synthetic, and it is license clean.

The CLI (`scripts/forecast.py`), the tests, and the demo notebook all load this
file and run fully offline. Regenerate it from statsmodels with:

```
python scripts/make_sample.py
```

## Optional FRED series

To experiment with macro series, run from the repo root:

```
python scripts/download_data.py
```

This fetches two public series from the FRED CSV endpoint into this directory:

- `CPIAUCSL.csv`, Consumer Price Index for All Urban Consumers (monthly)
- `T10YIE.csv`, 10-Year Breakeven Inflation Rate (daily)

Load them with `tsforecast.load_fred_csv("data/CPIAUCSL.csv")`. If FRED is
unreachable the script prints a message and exits without failing.
