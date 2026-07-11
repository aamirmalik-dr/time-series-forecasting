"""Forecast error metrics.

All metrics accept array-like inputs of equal length and return a float.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _to_arrays(y_true: object, y_pred: object) -> tuple[np.ndarray, np.ndarray]:
    """Coerce inputs to aligned 1-D float arrays and validate their shapes.

    Args:
        y_true: Observed values.
        y_pred: Predicted values.

    Returns:
        Tuple of (y_true, y_pred) as 1-D numpy float arrays.

    Raises:
        ValueError: If the inputs are empty or have mismatched lengths.
    """
    a = np.asarray(pd.Series(y_true), dtype=float).ravel()
    b = np.asarray(pd.Series(y_pred), dtype=float).ravel()
    if a.size == 0 or b.size == 0:
        raise ValueError("metric inputs must be non-empty")
    if a.shape != b.shape:
        raise ValueError(f"length mismatch: y_true has {a.size}, y_pred has {b.size}")
    return a, b


def rmse(y_true: object, y_pred: object) -> float:
    """Root mean squared error.

    Args:
        y_true: Observed values.
        y_pred: Predicted values.

    Returns:
        RMSE as a float, in the units of the series.
    """
    a, b = _to_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(y_true: object, y_pred: object) -> float:
    """Mean absolute error.

    Args:
        y_true: Observed values.
        y_pred: Predicted values.

    Returns:
        MAE as a float, in the units of the series.
    """
    a, b = _to_arrays(y_true, y_pred)
    return float(np.mean(np.abs(a - b)))


def mape(y_true: object, y_pred: object) -> float:
    """Mean absolute percentage error, in percent.

    Args:
        y_true: Observed values. Must not contain zeros.
        y_pred: Predicted values.

    Returns:
        MAPE as a float percentage (e.g. 2.5 means 2.5 percent).

    Raises:
        ValueError: If any observed value is zero, which makes MAPE undefined.
    """
    a, b = _to_arrays(y_true, y_pred)
    if np.any(a == 0):
        raise ValueError("MAPE is undefined when y_true contains zeros")
    return float(np.mean(np.abs((a - b) / a)) * 100.0)
