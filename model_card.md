# Model card: CO2 monthly SARIMAX

## Overview

A seasonal ARIMA (SARIMAX) model for the Mauna Loa atmospheric CO2 monthly
series. It produces point forecasts with 95 percent prediction intervals and is
evaluated with a walk-forward (expanding-window) backtest that refits at every
fold. The model, the committed sample, and every figure in this repository run
fully offline.

## Model specification

| Field | Value |
| --- | --- |
| Family | SARIMAX (statsmodels state-space) |
| Non-seasonal order (p, d, q) | (1, 1, 1) |
| Seasonal order (P, D, Q, s) | (1, 1, 1, 12) |
| Trend term | none |
| Fit options | enforce_stationarity=False, enforce_invertibility=False |
| Interval | 95 percent prediction interval from the state-space filter |

The order is fixed, not searched. There is no auto-ARIMA. The (1, 1, 1) x
(1, 1, 1, 12) specification is a standard airline-style seasonal model that
suits a smoothly trending series with a strong yearly cycle.

## Data

- Series: monthly mean atmospheric CO2 concentration in ppm.
- Source: the public Mauna Loa record (NOAA) that ships with statsmodels
  (`statsmodels.datasets.co2`). Public data, license clean.
- Committed sample: `data/co2_sample.csv`, the full monthly series carved to a
  regular monthly grid (526 observations, 1958-03 to 2001-12). It is a carved
  public subset, not synthetic. Regenerate it with `python scripts/make_sample.py`.
- Preprocessing: the raw weekly record is resampled to monthly means and any
  missing months are linearly interpolated, giving a regular complete series.

## Backtest protocol

Expanding-window walk-forward over the last 60 months, refit at every fold.

| Field | Value |
| --- | --- |
| Initial train size | 466 months |
| Held-out months | 60 |
| Horizon per fold | 12 months |
| Step between folds | 12 months |
| Folds | 5 |
| Scored points | 60 |

## Measured results (this session)

Backtest, SARIMAX(1,1,1)x(1,1,1,12), 5 folds over 60 held-out months:

| Metric | Value |
| --- | --- |
| RMSE | 0.656 ppm |
| MAE | 0.507 ppm |
| MAPE | 0.138 percent |

Single held-out window used for the hero figure (last 24 months, one fit):

| Metric | Value |
| --- | --- |
| Mean absolute error | 0.351 ppm |
| Max absolute error | 0.896 ppm |
| 95 percent interval coverage | 24 of 24 points (100 percent) |
| Mean interval width | 2.62 ppm |

Full-series fit diagnostics:

| Metric | Value |
| --- | --- |
| AIC | 214.3 |
| STL seasonal strength | 0.984 |
| Residual mean | 0.016 ppm |
| Residual std | 0.297 ppm |
| Ljung-Box(12) statistic | 4.07 |
| Ljung-Box(12) p-value | 0.982 |

The Ljung-Box p-value of 0.982 is well above 0.05, so the residuals are
indistinguishable from white noise at 12 lags. The interval reaches 100 percent
coverage on this window, which for 24 points is consistent with a nominal 95
percent interval and reflects how regular the CO2 series is.

## Intended use and limits

- Intended as a worked example of honest out-of-sample forecasting on a highly
  regular seasonal series. The point is the methodology (refit-every-fold
  backtesting and calibrated intervals), not a hard forecasting benchmark.
- CO2 is unusually predictable. These error numbers should not be read as
  typical for noisier operational series.
- Univariate only. No exogenous regressors, no automatic order selection, and
  no regime-change or structural-break handling.

## Reproduce

```
python scripts/forecast.py backtest --horizon 12 --holdout 60   # metrics.json
python scripts/forecast.py forecast --horizon 24                # hero figure + CSV
python scripts/forecast.py diagnostics --lags 12                # residual test
```
