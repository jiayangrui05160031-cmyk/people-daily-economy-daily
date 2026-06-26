"""forecasting.naive - 简单 MA + 线性趋势 (原 forecast.py 算法).

零依赖 (纯 Python + numpy)。
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .base import ForecastResult, ForecastReport, Forecast, Forecaster


def _moving_average(values: Sequence[float], window: int = 3) -> float:
    if not values:
        return 0.0
    recent = list(values)[:window]
    return sum(recent) / len(recent)


def _linear_trend(values: Sequence[float]) -> Tuple[float, float]:
    n = len(values)
    if n < 2:
        return 0.0, float(values[0]) if values else 0.0
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(values) / n
    num = sum((xs[i] - mean_x) * (values[i] - mean_y) for i in range(n))
    den = sum((xs[i] - mean_x) ** 2 for i in range(n))
    slope = num / den if den > 1e-9 else 0.0
    intercept = mean_y - slope * mean_x
    return slope, intercept


def _classify_trend(slope: float, scale: float) -> str:
    norm = abs(slope) / max(scale, 1e-3)
    if norm >= 0.1:
        return "rising" if slope > 0 else "falling"
    return "stable"


class NaiveForecaster:
    """MA(window) + linear trend. Replicates forecast.py behaviour."""

    name = "naive_ma_trend"

    def __init__(self, window: int = 5, min_window: int = 3):
        self.window = window
        self.min_window = min_window

    def fit_predict(self, series: np.ndarray, horizon: int = 1,
                    scale: float = 1.0, **kwargs) -> ForecastResult:
        values = list(series) if series is not None else []
        if not values:
            return ForecastResult(0, 0, 0, "naive", "stable", 0.0, 0)
        if len(values) < self.min_window:
            pred = _moving_average(values, window=len(values))
            return ForecastResult(
                predicted=round(pred, 3),
                lower=round(pred - 0.1 * scale, 3),
                upper=round(pred + 0.1 * scale, 3),
                method="MA", trend="stable",
                confidence=0.3, sample_size=len(values),
            )
        ma = _moving_average(values, window=min(self.window, len(values)))
        slope, _ = _linear_trend(values)
        pred = ma + slope * horizon
        win = min(self.window, len(values))
        recent_std = (sum((v - ma) ** 2 for v in values[:win]) / win) ** 0.5
        band = 1.96 * max(recent_std, 0.05 * scale) * (horizon ** 0.5)
        trend = _classify_trend(slope, scale)
        confidence = max(0.2, min(0.9, len(values) / 20.0))
        return ForecastResult(
            predicted=round(pred, 3),
            lower=round(pred - band, 3),
            upper=round(pred + band, 3),
            method="MA+trend", trend=trend,
            confidence=round(confidence, 3),
            sample_size=len(values),
        )


# ---------------------------------------------------------------------------
# Back-compat: original forecast.py predict_metric() / predict_next_day()
# ---------------------------------------------------------------------------


def predict_metric(values: Sequence[float], window: int = 5, scale: float = 1.0) -> Forecast:
    """Back-compat wrapper. Returns legacy Forecast dataclass."""
    f = NaiveForecaster(window=window).fit_predict(
        np.array(list(values), dtype=np.float64), horizon=1, scale=scale,
    )
    return Forecast(
        metric="", predicted=f.predicted, lower=f.lower, upper=f.upper,
        trend=f.trend, method=f.method,
        confidence=f.confidence, sample_size=f.sample_size,
    )


_DEFAULT_METRICS = [
    ("sentiment_index", 1.0),
    ("policy_stance_score", 0.1),
    ("article_count", 5.0),
    ("attention_entropy", 0.05),
]


def predict_next_day(
    date_str: str,
    lookback_days: int = 14,
    metrics_to_predict: Optional[List[Tuple[str, float]]] = None,
    forecaster: Optional[Forecaster] = None,
):
    """Back-compat: replicate forecast.predict_next_day().

    `forecaster` defaults to NaiveForecaster; pass any Forecaster to
    swap algorithm without touching this function.
    """
    from src.storage import repository as repo
    from src.utils.date_utils import parse_date
    from datetime import timedelta

    if metrics_to_predict is None:
        metrics_to_predict = _DEFAULT_METRICS
    fc = forecaster or NaiveForecaster()

    end = (parse_date(date_str) - timedelta(days=1)).isoformat()
    start = (parse_date(date_str) - timedelta(days=lookback_days)).isoformat()
    history = repo.list_metrics(start_date=start, end_date=end)
    if not history:
        return ForecastReport(
            base_date=date_str,
            target_date=(parse_date(date_str) + timedelta(days=1)).isoformat(),
            forecasts=[],
            headline="(无历史数据, 无法预测)",
        )
    history_sorted = sorted(history, key=lambda m: m.date, reverse=True)
    forecasts: List[ForecastResult] = []
    for metric_name, scale in metrics_to_predict:
        values = [float(getattr(m, metric_name) or 0) for m in history_sorted]
        f = fc.fit_predict(np.array(values, dtype=np.float64),
                            horizon=1, scale=scale)
        # The legacy API reported `metric` on the dataclass; we keep
        # metric on Forecast but ForecastResult is metric-agnostic.
        # ForecastReport stores ForecastResult; callers that need
        # metric_name can use the index alignment with metrics_to_predict.
        forecasts.append(f)
    rising = sum(1 for f in forecasts if f.trend == "rising")
    falling = sum(1 for f in forecasts if f.trend == "falling")
    if rising > falling:
        headline = f"预测次日整体偏多 (上涨指标 {rising}/{len(forecasts)})"
    elif falling > rising:
        headline = f"预测次日整体偏空 (下跌指标 {falling}/{len(forecasts)})"
    else:
        headline = f"预测次日走势平稳 ({len(forecasts)} 指标中性)"
    return ForecastReport(
        base_date=date_str,
        target_date=(parse_date(date_str) + timedelta(days=1)).isoformat(),
        forecasts=forecasts, headline=headline,
    )
