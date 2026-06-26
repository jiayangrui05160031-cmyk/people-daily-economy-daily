"""forecasting.base - Forecaster Protocol + result dataclasses.

Unified contract for time-series forecasters. Backward-compatible
re-exports of the original dataclass names (Forecast, ForecastReport,
HorizonForecast, EnhancedForecast, BacktestCase, HorizonReport,
BacktestReport) so existing callers don't break.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Result dataclasses (shared shape across forecasters)
# ---------------------------------------------------------------------------


@dataclass
class ForecastResult:
    """One-step or one-horizon forecast."""
    predicted: float
    lower: float
    upper: float
    method: str         # "MA" | "MA+trend" | "holt-winters" | "stl+ar" | "naive"
    trend: str          # "rising" | "falling" | "stable"
    confidence: float   # 0..1
    sample_size: int = 0
    seasonal_component: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HorizonForecast:
    horizon: int
    predicted: float
    lower: float
    upper: float
    method: str
    confidence: float
    trend: str
    seasonal_component: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ForecastReport:
    base_date: str
    target_date: str
    forecasts: List[ForecastResult] = field(default_factory=list)
    headline: str = ""
    summary: str = ""

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["forecasts"] = [f.as_dict() for f in self.forecasts]
        return d


@dataclass
class EnhancedForecast:
    base_date: str
    target_metric: str
    horizons: List[HorizonForecast] = field(default_factory=list)
    in_sample_mae: float = 0.0
    in_sample_mape: float = 0.0
    seasonality_period: int = 0
    trend_slope: float = 0.0
    method_selected: str = ""
    model_params: Dict[str, float] = field(default_factory=dict)
    summary: str = ""

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["horizons"] = [h.as_dict() for h in self.horizons]
        return d


@dataclass
class BacktestCase:
    target_date: str
    predicted: float
    actual: float
    lower: float
    upper: float
    abs_error: float
    direction_correct: bool
    in_band: bool

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HorizonReport:
    horizon: int
    n: int
    mae: float
    rmse: float
    bias: float
    direction_accuracy: float
    band_coverage: float

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestReport:
    base_date: str
    horizon_reports: List[HorizonReport]
    cases: List[BacktestCase]
    summary: str

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Forecaster(Protocol):
    """Every forecaster speaks this contract."""
    name: str

    def fit_predict(self, series: np.ndarray, horizon: int = 1,
                    **kwargs) -> ForecastResult:
        """Train on `series` and predict `horizon` steps ahead."""
        ...


# Legacy aliases (back-compat: forecast.py / forecast_enhanced.py /
# forecast_backtest.py re-export these names so any `from
# src.analysis.forecast import Forecast` style import keeps working).
@dataclass
class Forecast:
    """Legacy single-metric forecast (forecast.py)."""
    metric: str
    predicted: float
    lower: float
    upper: float
    trend: str
    method: str
    confidence: float
    sample_size: int

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
