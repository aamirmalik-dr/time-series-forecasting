"""Walk-forward (expanding window) backtesting for time-series forecasters."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tsforecast.forecast import SarimaxForecaster
from tsforecast.metrics import mae, mape, rmse


@dataclass(frozen=True)
class BacktestResult:
    """Outcome of a walk-forward backtest.

    Attributes:
        predictions: Out-of-sample point forecasts, indexed like the input.
        actuals: The held-out observed values aligned with ``predictions``.
        rmse: Root mean squared error over all held-out points.
        mae: Mean absolute error over all held-out points.
        mape: Mean absolute percentage error (percent) over all held-out points.
        n_folds: Number of refits performed.
    """

    predictions: pd.Series
    actuals: pd.Series
    rmse: float
    mae: float
    mape: float
    n_folds: int

    def summary(self) -> str:
        """One-line human-readable summary.

        Returns:
            Formatted string with fold count and error metrics.
        """
        return (
            f"{self.n_folds} folds, {len(self.actuals)} held-out points | "
            f"RMSE={self.rmse:.4f} MAE={self.mae:.4f} MAPE={self.mape:.3f}%"
        )


def expanding_window_splits(
    n_obs: int,
    initial_train_size: int,
    horizon: int = 1,
    step: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Generate expanding-window train/test index splits.

    The first fold trains on ``[0, initial_train_size)`` and tests on the next
    ``horizon`` points. Each subsequent fold extends the training window by
    ``step`` observations. Folds never overlap in their test points when
    ``step == horizon`` (the default).

    Args:
        n_obs: Total number of observations.
        initial_train_size: Size of the first training window. Must be at
            least 1 and leave room for at least one test point.
        horizon: Number of consecutive points in each test window. The final
            fold may be shorter if fewer points remain.
        step: How far the training window grows between folds. Defaults to
            ``horizon`` so test windows tile the holdout without overlap.

    Yields:
        Tuples of (train_indices, test_indices) as integer numpy arrays.

    Raises:
        ValueError: If sizes are inconsistent.
    """
    if initial_train_size < 1:
        raise ValueError(f"initial_train_size must be >= 1, got {initial_train_size}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if initial_train_size >= n_obs:
        raise ValueError(
            f"initial_train_size ({initial_train_size}) must be < n_obs ({n_obs}) "
            "to leave at least one test point"
        )
    if step is None:
        step = horizon
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    train_end = initial_train_size
    while train_end < n_obs:
        test_end = min(train_end + horizon, n_obs)
        yield np.arange(0, train_end), np.arange(train_end, test_end)
        train_end += step


def walk_forward_backtest(
    series: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    initial_train_size: int | None = None,
    horizon: int = 1,
    step: int | None = None,
    trend: str | None = None,
) -> BacktestResult:
    """Backtest a SARIMAX specification with an expanding training window.

    At each fold the model is refit on all data up to the fold boundary and
    asked to forecast the next ``horizon`` points. Forecasts are compared to
    the actual held-out values and errors are aggregated over all folds.

    Args:
        series: Regularly spaced series with no missing values.
        order: Non-seasonal (p, d, q) order.
        seasonal_order: Seasonal (P, D, Q, s) order.
        initial_train_size: Size of the first training window. Defaults to
            75 percent of the series.
        horizon: Forecast horizon per fold.
        step: Training-window growth between folds. Defaults to ``horizon``.
        trend: Deterministic trend passed to SARIMAX.

    Returns:
        A :class:`BacktestResult` with per-point predictions and aggregate
        RMSE, MAE, and MAPE.

    Raises:
        ValueError: If the series is too short for the requested split.
    """
    series = series.dropna()
    n_obs = len(series)
    if initial_train_size is None:
        initial_train_size = int(n_obs * 0.75)
    predictions: list[pd.Series] = []
    actuals: list[pd.Series] = []
    n_folds = 0
    for train_idx, test_idx in expanding_window_splits(n_obs, initial_train_size, horizon, step):
        train = series.iloc[train_idx]
        test = series.iloc[test_idx]
        model = SarimaxForecaster(order=order, seasonal_order=seasonal_order, trend=trend)
        model.fit(train)
        forecast = model.forecast(steps=len(test))
        predicted = pd.Series(np.asarray(forecast.mean), index=test.index)
        predictions.append(predicted)
        actuals.append(test)
        n_folds += 1
    pred = pd.concat(predictions)
    act = pd.concat(actuals)
    return BacktestResult(
        predictions=pred,
        actuals=act,
        rmse=rmse(act, pred),
        mae=mae(act, pred),
        mape=mape(act, pred),
        n_folds=n_folds,
    )
