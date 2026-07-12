"""tsforecast: classical time-series forecasting with ARIMA/SARIMAX and walk-forward backtesting."""

from tsforecast.backtest import BacktestResult, expanding_window_splits, walk_forward_backtest
from tsforecast.data import load_co2_monthly, load_co2_sample, load_fred_csv
from tsforecast.decompose import DecompositionResult, decompose
from tsforecast.forecast import ForecastResult, ResidualDiagnostics, SarimaxForecaster
from tsforecast.metrics import mae, mape, rmse

__all__ = [
    "BacktestResult",
    "DecompositionResult",
    "ForecastResult",
    "ResidualDiagnostics",
    "SarimaxForecaster",
    "decompose",
    "expanding_window_splits",
    "load_co2_monthly",
    "load_co2_sample",
    "load_fred_csv",
    "mae",
    "mape",
    "rmse",
    "walk_forward_backtest",
]

__version__ = "0.1.0"
