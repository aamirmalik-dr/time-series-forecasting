# Data

This directory is gitignored except for this file.

## Primary demo series (no download needed)

The headline demo uses the Mauna Loa atmospheric CO2 record that ships with
statsmodels (`statsmodels.datasets.co2`). It loads offline, so the tests,
`scripts/forecast.py`, and the demo notebook require no files here.

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
