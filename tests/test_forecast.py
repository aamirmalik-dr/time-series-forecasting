"""Tests for the SARIMAX wrapper, decomposition, and data loading."""

import numpy as np
import pandas as pd
import pytest

from tsforecast.data import load_co2_monthly, load_co2_sample, load_fred_csv
from tsforecast.decompose import decompose
from tsforecast.forecast import ResidualDiagnostics, SarimaxForecaster


@pytest.fixture(scope="module")
def trend_series() -> pd.Series:
    rng = np.random.default_rng(7)
    index = pd.date_range("2010-01-01", periods=60, freq="MS")
    values = 50 + 0.3 * np.arange(60) + rng.normal(0, 0.3, 60)
    return pd.Series(values, index=index)


class TestSarimaxForecaster:
    def test_forecast_has_requested_length(self, trend_series: pd.Series) -> None:
        model = SarimaxForecaster(order=(1, 1, 0)).fit(trend_series)
        result = model.forecast(steps=6)
        assert len(result.mean) == 6
        assert len(result.lower) == 6
        assert len(result.upper) == 6

    def test_forecast_index_continues_the_series(self, trend_series: pd.Series) -> None:
        model = SarimaxForecaster(order=(1, 1, 0)).fit(trend_series)
        result = model.forecast(steps=3)
        expected = pd.date_range("2015-01-01", periods=3, freq="MS")
        assert list(result.mean.index) == list(expected)

    def test_interval_brackets_point_forecast(self, trend_series: pd.Series) -> None:
        model = SarimaxForecaster(order=(1, 1, 0)).fit(trend_series)
        result = model.forecast(steps=6)
        assert (result.lower.values <= result.mean.values).all()
        assert (result.mean.values <= result.upper.values).all()

    def test_forecast_tracks_trend(self, trend_series: pd.Series) -> None:
        model = SarimaxForecaster(order=(1, 1, 0), trend="c").fit(trend_series)
        result = model.forecast(steps=6)
        last = float(trend_series.iloc[-1])
        assert result.mean.iloc[-1] > last  # upward trend continues

    def test_forecast_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="fit"):
            SarimaxForecaster().forecast(steps=1)

    def test_invalid_steps_raise(self, trend_series: pd.Series) -> None:
        model = SarimaxForecaster(order=(1, 1, 0)).fit(trend_series)
        with pytest.raises(ValueError, match="steps"):
            model.forecast(steps=0)


class TestResidualDiagnostics:
    def test_returns_residual_diagnostics_instance(self, trend_series: pd.Series) -> None:
        model = SarimaxForecaster(order=(1, 1, 0)).fit(trend_series)
        diag = model.residual_diagnostics(lags=6)
        assert isinstance(diag, ResidualDiagnostics)
        assert diag.lags == 6

    def test_mean_near_zero_and_std_positive(self, trend_series: pd.Series) -> None:
        model = SarimaxForecaster(order=(1, 1, 0), trend="c").fit(trend_series)
        diag = model.residual_diagnostics()
        assert abs(diag.mean) < 1.0  # residuals of a good fit centre near zero
        assert diag.std > 0.0

    def test_white_noise_residuals_pass_the_test(self) -> None:
        # A series that is genuine white noise should not show autocorrelation.
        rng = np.random.default_rng(0)
        index = pd.date_range("2005-01-01", periods=120, freq="MS")
        noise = pd.Series(rng.normal(0, 1, 120), index=index)
        model = SarimaxForecaster(order=(0, 0, 0)).fit(noise)
        diag = model.residual_diagnostics(lags=10)
        assert diag.is_white_noise()
        assert diag.ljung_box_pvalue > 0.05

    def test_pvalue_and_stat_are_finite(self, trend_series: pd.Series) -> None:
        model = SarimaxForecaster(order=(1, 1, 0)).fit(trend_series)
        diag = model.residual_diagnostics(lags=8)
        assert np.isfinite(diag.ljung_box_stat)
        assert 0.0 <= diag.ljung_box_pvalue <= 1.0

    def test_diagnostics_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="fit"):
            SarimaxForecaster().residual_diagnostics()


@pytest.fixture(scope="module")
def seasonal_series() -> pd.Series:
    index = pd.date_range("2000-01-01", periods=72, freq="MS")
    t = np.arange(72)
    values = 10 + 0.1 * t + 3 * np.sin(2 * np.pi * t / 12)
    return pd.Series(values, index=index)


class TestDecompose:
    @pytest.mark.parametrize("method", ["stl", "classical"])
    def test_components_reconstruct_series(self, seasonal_series: pd.Series, method: str) -> None:
        result = decompose(seasonal_series, period=12, method=method)
        recon = (result.trend + result.seasonal + result.resid).dropna()
        np.testing.assert_allclose(recon.values, seasonal_series.loc[recon.index].values)

    def test_seasonal_strength_high_for_clean_cycle(self, seasonal_series: pd.Series) -> None:
        result = decompose(seasonal_series, period=12, method="stl")
        assert result.seasonal_strength() > 0.9

    def test_unknown_method_raises(self, seasonal_series: pd.Series) -> None:
        with pytest.raises(ValueError, match="unknown method"):
            decompose(seasonal_series, period=12, method="fourier")

    def test_short_series_raises(self) -> None:
        short = pd.Series(np.arange(10.0), index=pd.date_range("2000-01-01", periods=10, freq="MS"))
        with pytest.raises(ValueError, match="at least"):
            decompose(short, period=12)


class TestData:
    def test_co2_sample_loads_offline_and_is_regular(self) -> None:
        series = load_co2_sample()
        assert len(series) > 400
        assert series.name == "co2"
        assert series.isna().sum() == 0
        assert pd.infer_freq(series.index) == "MS"
        assert series.min() > 300  # ppm, sanity bounds
        assert series.max() < 400

    def test_co2_sample_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="make_sample"):
            load_co2_sample(tmp_path / "absent.csv")

    def test_co2_sample_matches_custom_csv(self, tmp_path) -> None:
        csv = tmp_path / "mini.csv"
        csv.write_text("date,co2\n2010-01-01,380.0\n2010-02-01,381.5\n2010-03-01,382.0\n")
        series = load_co2_sample(csv)
        assert len(series) == 3
        assert series.iloc[1] == 381.5
        assert pd.infer_freq(series.index) == "MS"

    def test_co2_monthly_is_regular_and_complete(self) -> None:
        series = load_co2_monthly()
        assert len(series) > 400
        assert series.isna().sum() == 0
        assert pd.infer_freq(series.index) == "MS"
        assert series.min() > 300  # ppm, sanity bounds
        assert series.max() < 400

    def test_load_fred_csv_parses_fred_format(self, tmp_path) -> None:
        csv = tmp_path / "TEST.csv"
        csv.write_text("DATE,TEST\n2020-01-01,1.5\n2020-02-01,.\n2020-03-01,2.5\n")
        series = load_fred_csv(csv)
        assert len(series) == 2  # the "." row is dropped
        assert series.iloc[0] == 1.5
        assert isinstance(series.index, pd.DatetimeIndex)

    def test_load_fred_csv_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_fred_csv(tmp_path / "NOPE.csv")
