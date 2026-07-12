"""SARIMAX forecasting built on statsmodels."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults


@dataclass(frozen=True)
class ResidualDiagnostics:
    """Summary statistics of a fitted model's residuals.

    Attributes:
        mean: Mean of the residuals. Should be close to zero for a good fit.
        std: Standard deviation of the residuals.
        ljung_box_stat: Ljung-Box Q statistic at the tested lag.
        ljung_box_pvalue: Ljung-Box p-value. Values above 0.05 indicate the
            residuals are indistinguishable from white noise at the tested lag.
        lags: Number of lags used for the Ljung-Box test.
    """

    mean: float
    std: float
    ljung_box_stat: float
    ljung_box_pvalue: float
    lags: int

    def is_white_noise(self, alpha: float = 0.05) -> bool:
        """Whether the residuals pass the Ljung-Box white-noise test.

        Args:
            alpha: Significance level for the test.

        Returns:
            True if the Ljung-Box p-value exceeds ``alpha`` (no significant
            autocorrelation detected).
        """
        return self.ljung_box_pvalue > alpha


@dataclass(frozen=True)
class ForecastResult:
    """Point forecast with a confidence interval.

    Attributes:
        mean: Point forecasts.
        lower: Lower bound of the confidence interval.
        upper: Upper bound of the confidence interval.
        alpha: Significance level used for the interval (0.05 gives 95%).
    """

    mean: pd.Series
    lower: pd.Series
    upper: pd.Series
    alpha: float


@dataclass
class SarimaxForecaster:
    """SARIMAX model wrapper with a fit/forecast interface.

    Args:
        order: Non-seasonal (p, d, q) order.
        seasonal_order: Seasonal (P, D, Q, s) order. Use (0, 0, 0, 0) for a
            plain ARIMA model.
        trend: Deterministic trend, passed straight to statsmodels
            (e.g. "c" for a constant, None to omit).

    Example:
        >>> model = SarimaxForecaster(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        >>> model.fit(train)
        >>> result = model.forecast(steps=12)
    """

    order: tuple[int, int, int] = (1, 1, 1)
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0)
    trend: str | None = None
    _results: SARIMAXResults | None = field(default=None, init=False, repr=False)

    def fit(self, series: pd.Series) -> SarimaxForecaster:
        """Fit the model on a training series.

        Args:
            series: Regularly spaced series with a DatetimeIndex or
                PeriodIndex and no missing values.

        Returns:
            self, so calls can be chained.
        """
        model = SARIMAX(
            series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._results = model.fit(disp=False)
        return self

    @property
    def results(self) -> SARIMAXResults:
        """Fitted statsmodels results object.

        Returns:
            The underlying ``SARIMAXResults``.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if self._results is None:
            raise RuntimeError("call fit() before accessing results")
        return self._results

    def forecast(self, steps: int, alpha: float = 0.05) -> ForecastResult:
        """Forecast a number of steps beyond the training sample.

        Args:
            steps: Number of periods to forecast. Must be positive.
            alpha: Significance level for the confidence interval.

        Returns:
            A :class:`ForecastResult` with point forecasts and interval bounds.

        Raises:
            ValueError: If ``steps`` is not positive.
            RuntimeError: If the model has not been fitted yet.
        """
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        prediction = self.results.get_forecast(steps=steps)
        interval = prediction.conf_int(alpha=alpha)
        return ForecastResult(
            mean=prediction.predicted_mean,
            lower=interval.iloc[:, 0],
            upper=interval.iloc[:, 1],
            alpha=alpha,
        )

    def aic(self) -> float:
        """Akaike information criterion of the fitted model.

        Returns:
            AIC as a float.
        """
        return float(self.results.aic)

    def residual_diagnostics(self, lags: int = 12) -> ResidualDiagnostics:
        """Compute white-noise diagnostics on the fitted model's residuals.

        Args:
            lags: Number of lags for the Ljung-Box test. For monthly data with
                yearly seasonality, 12 is a natural choice.

        Returns:
            A :class:`ResidualDiagnostics` with residual mean, standard
            deviation, and the Ljung-Box statistic and p-value.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        results = self.results
        resid = np.asarray(results.resid, dtype=float)
        # Drop the state-space startup residuals (burn-in plus any diffuse
        # initialization); they reflect filter initialization, not model fit.
        burn = int(getattr(results, "loglikelihood_burn", 0)) + int(
            getattr(results, "nobs_diffuse", 0)
        )
        resid = resid[burn:]
        resid = resid[np.isfinite(resid)]
        lb = acorr_ljungbox(resid, lags=[lags], return_df=True)
        return ResidualDiagnostics(
            mean=float(np.mean(resid)),
            std=float(np.std(resid)),
            ljung_box_stat=float(lb["lb_stat"].iloc[0]),
            ljung_box_pvalue=float(lb["lb_pvalue"].iloc[0]),
            lags=lags,
        )
