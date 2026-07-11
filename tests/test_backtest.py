"""Tests for the expanding-window splitter and walk-forward backtest."""

import numpy as np
import pandas as pd
import pytest

from tsforecast.backtest import expanding_window_splits, walk_forward_backtest


class TestExpandingWindowSplits:
    def test_first_fold_uses_initial_train_size(self) -> None:
        splits = list(expanding_window_splits(n_obs=10, initial_train_size=6, horizon=2))
        train, test = splits[0]
        assert train.tolist() == [0, 1, 2, 3, 4, 5]
        assert test.tolist() == [6, 7]

    def test_test_windows_tile_holdout_without_overlap(self) -> None:
        splits = list(expanding_window_splits(n_obs=10, initial_train_size=6, horizon=2))
        all_test = np.concatenate([test for _, test in splits])
        assert all_test.tolist() == [6, 7, 8, 9]

    def test_final_fold_may_be_short(self) -> None:
        splits = list(expanding_window_splits(n_obs=10, initial_train_size=6, horizon=3))
        assert splits[-1][1].tolist() == [9]

    def test_training_window_expands(self) -> None:
        splits = list(expanding_window_splits(n_obs=12, initial_train_size=6, horizon=2))
        train_sizes = [len(train) for train, _ in splits]
        assert train_sizes == [6, 8, 10]

    def test_train_always_precedes_test(self) -> None:
        for train, test in expanding_window_splits(n_obs=20, initial_train_size=5, horizon=3):
            assert train.max() < test.min()
            assert train.min() == 0

    def test_step_smaller_than_horizon_gives_overlapping_tests(self) -> None:
        splits = list(expanding_window_splits(n_obs=10, initial_train_size=6, horizon=3, step=1))
        assert splits[0][1].tolist() == [6, 7, 8]
        assert splits[1][1].tolist() == [7, 8, 9]

    def test_invalid_sizes_raise(self) -> None:
        with pytest.raises(ValueError):
            list(expanding_window_splits(n_obs=10, initial_train_size=10))
        with pytest.raises(ValueError):
            list(expanding_window_splits(n_obs=10, initial_train_size=0))
        with pytest.raises(ValueError):
            list(expanding_window_splits(n_obs=10, initial_train_size=5, horizon=0))


@pytest.fixture(scope="module")
def seasonal_series() -> pd.Series:
    rng = np.random.default_rng(42)
    index = pd.date_range("2000-01-01", periods=96, freq="MS")
    t = np.arange(96)
    values = 100 + 0.5 * t + 5 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.5, 96)
    return pd.Series(values, index=index)


class TestWalkForwardBacktest:
    def test_predictions_cover_all_holdout_points(self, seasonal_series: pd.Series) -> None:
        result = walk_forward_backtest(
            seasonal_series,
            order=(1, 1, 0),
            seasonal_order=(0, 1, 0, 12),
            initial_train_size=72,
            horizon=12,
        )
        assert len(result.predictions) == 24
        assert result.n_folds == 2
        assert result.predictions.index.equals(seasonal_series.index[72:])
        assert result.actuals.equals(seasonal_series.iloc[72:])

    def test_metrics_are_finite_and_reasonable(self, seasonal_series: pd.Series) -> None:
        result = walk_forward_backtest(
            seasonal_series,
            order=(1, 1, 0),
            seasonal_order=(0, 1, 0, 12),
            initial_train_size=72,
            horizon=12,
        )
        assert np.isfinite(result.rmse)
        assert np.isfinite(result.mape)
        assert result.rmse > 0
        # the series is highly predictable, errors should be small vs its level
        assert result.mape < 10.0
        assert result.mae <= result.rmse

    def test_summary_mentions_folds(self, seasonal_series: pd.Series) -> None:
        result = walk_forward_backtest(
            seasonal_series,
            order=(1, 1, 0),
            seasonal_order=(0, 1, 0, 12),
            initial_train_size=84,
            horizon=12,
        )
        assert "1 folds" in result.summary()
        assert "RMSE" in result.summary()
