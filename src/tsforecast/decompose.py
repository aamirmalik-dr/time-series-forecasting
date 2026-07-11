"""Seasonal and trend decomposition wrappers around statsmodels."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from statsmodels.tsa.seasonal import STL, seasonal_decompose


@dataclass(frozen=True)
class DecompositionResult:
    """Container for the components of a decomposed series.

    Attributes:
        observed: The input series.
        trend: Estimated trend component.
        seasonal: Estimated seasonal component.
        resid: Remainder after removing trend and seasonality.
        method: Name of the method used ("stl" or "classical").
    """

    observed: pd.Series
    trend: pd.Series
    seasonal: pd.Series
    resid: pd.Series
    method: str

    def seasonal_strength(self) -> float:
        """Strength of seasonality on a 0 to 1 scale.

        Uses the Hyndman measure: ``max(0, 1 - Var(resid) / Var(seasonal + resid))``.
        Values near 1 indicate strong seasonality.

        Returns:
            Seasonal strength as a float in [0, 1].
        """
        detrended = (self.seasonal + self.resid).dropna()
        resid = self.resid.dropna()
        var_detrended = float(detrended.var())
        if var_detrended == 0.0:
            return 0.0
        return max(0.0, 1.0 - float(resid.var()) / var_detrended)


def decompose(
    series: pd.Series,
    period: int,
    method: str = "stl",
    model: str = "additive",
) -> DecompositionResult:
    """Decompose a series into trend, seasonal, and residual components.

    Args:
        series: Regularly spaced series with no missing values.
        period: Number of observations per seasonal cycle (12 for monthly data
            with yearly seasonality).
        method: "stl" for STL (LOESS-based, robust) or "classical" for moving
            average decomposition.
        model: Only for the classical method, "additive" or "multiplicative".

    Returns:
        A :class:`DecompositionResult` with the estimated components.

    Raises:
        ValueError: If ``method`` is not recognised or the series is too short.
    """
    series = series.dropna()
    if len(series) < 2 * period:
        raise ValueError(f"need at least {2 * period} observations, got {len(series)}")
    if method == "stl":
        fitted = STL(series, period=period, robust=True).fit()
        return DecompositionResult(
            observed=series,
            trend=fitted.trend,
            seasonal=fitted.seasonal,
            resid=fitted.resid,
            method="stl",
        )
    if method == "classical":
        fitted = seasonal_decompose(series, model=model, period=period)
        return DecompositionResult(
            observed=series,
            trend=fitted.trend,
            seasonal=fitted.seasonal,
            resid=fitted.resid,
            method="classical",
        )
    raise ValueError(f"unknown method {method!r}; use 'stl' or 'classical'")
