"""Tests for tsforecast.metrics."""

import numpy as np
import pytest

from tsforecast.metrics import mae, mape, rmse


def test_rmse_zero_for_perfect_prediction() -> None:
    y = [1.0, 2.0, 3.0]
    assert rmse(y, y) == 0.0


def test_rmse_known_value() -> None:
    # errors are (1, -1, 1) -> mean square 1 -> rmse 1
    assert rmse([2.0, 2.0, 2.0], [1.0, 3.0, 1.0]) == pytest.approx(1.0)


def test_mae_known_value() -> None:
    assert mae([1.0, 2.0], [2.0, 4.0]) == pytest.approx(1.5)


def test_mape_known_value() -> None:
    # errors 10% and 20% -> mean 15%
    assert mape([100.0, 100.0], [90.0, 120.0]) == pytest.approx(15.0)


def test_mape_rejects_zero_actuals() -> None:
    with pytest.raises(ValueError, match="zero"):
        mape([0.0, 1.0], [1.0, 1.0])


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="mismatch"):
        rmse([1.0, 2.0], [1.0])


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        mae([], [])


def test_metrics_accept_numpy_arrays() -> None:
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.5, 2.5, 3.5])
    assert rmse(a, b) == pytest.approx(0.5)
