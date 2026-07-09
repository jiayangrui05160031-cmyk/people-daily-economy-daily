"""Deterministic unit tests for the public forecasting contract."""

import numpy as np

from src.forecasting import ForecastResult, Forecaster, NaiveForecaster
from src.forecasting.naive import _classify_trend, _linear_trend, _moving_average


def test_naive_forecaster_satisfies_protocol() -> None:
    forecaster = NaiveForecaster()
    assert isinstance(forecaster, Forecaster)


def test_empty_series_returns_safe_default() -> None:
    result = NaiveForecaster().fit_predict(np.array([]))
    assert result == ForecastResult(0, 0, 0, "naive", "stable", 0.0, 0)


def test_short_series_uses_moving_average() -> None:
    result = NaiveForecaster(min_window=3).fit_predict(
        np.array([1.0, 3.0]), scale=2.0
    )
    assert result.predicted == 2.0
    assert result.lower == 1.8
    assert result.upper == 2.2
    assert result.method == "MA"
    assert result.sample_size == 2


def test_increasing_series_is_classified_as_rising() -> None:
    result = NaiveForecaster(window=5).fit_predict(
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]), horizon=2, scale=1.0
    )
    assert result.predicted > 3.0
    assert result.lower < result.predicted < result.upper
    assert result.trend == "rising"
    assert result.method == "MA+trend"


def test_forecast_helpers_cover_edge_cases() -> None:
    assert _moving_average([], window=3) == 0.0
    assert _moving_average([4.0, 2.0, 8.0], window=2) == 3.0
    assert _linear_trend([7.0]) == (0.0, 7.0)
    assert _classify_trend(0.01, scale=1.0) == "stable"
    assert _classify_trend(-0.2, scale=1.0) == "falling"
