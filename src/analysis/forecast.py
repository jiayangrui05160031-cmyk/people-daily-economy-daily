"""analysis.forecast - 时序预测 (简单移动平均 + 线性趋势)

基于历史 N 天指标, 预测次日:
- sentiment_index (移动平均 + 趋势)
- policy_stance_score (移动平均 + 趋势)
- article_count (移动平均)
- attention_entropy (移动平均)

纯统计,无 ML 依赖。输出含置信区间。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import timedelta
from typing import Dict, List, Optional

from src.storage import repository as repo
from src.utils.date_utils import parse_date
from src.utils.logger import get_logger

logger = get_logger("analysis.forecast")


@dataclass
class Forecast:
    metric: str
    predicted: float
    lower: float
    upper: float
    trend: str  # 'rising' | 'falling' | 'stable'
    method: str  # 'MA' | 'MA+trend' | 'naive'
    confidence: float  # 0~1
    sample_size: int


@dataclass
class ForecastReport:
    base_date: str
    target_date: str
    forecasts: List[Forecast]
    headline: str

    def as_dict(self):
        d = asdict(self)
        d['forecasts'] = [asdict(f) for f in self.forecasts]
        return d


def _moving_average(values, window=3):
    if not values:
        return 0.0
    recent = values[:window]
    return sum(recent) / len(recent)


def _linear_trend(values):
    """简单线性回归斜率 (OLS 一阶)。返回 (slope, intercept)。"""
    n = len(values)
    if n < 2:
        return 0.0, values[0] if values else 0.0
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(values) / n
    num = sum((xs[i] - mean_x) * (values[i] - mean_y) for i in range(n))
    den = sum((xs[i] - mean_x) ** 2 for i in range(n))
    slope = num / den if den > 1e-9 else 0.0
    intercept = mean_y - slope * mean_x
    return slope, intercept


def _classify_trend(slope, scale):
    norm = abs(slope) / max(scale, 1e-3)
    if norm >= 0.1: return "rising" if slope > 0 else "falling"
    return "stable"


def predict_metric(values, window=5, scale=1.0):
    if not values:
        return Forecast(
            metric="", predicted=0.0, lower=0.0, upper=0.0,
            trend="stable", method="naive", confidence=0.0, sample_size=0,
        )
    if len(values) < 3:
        pred = _moving_average(values, window=len(values))
        return Forecast(
            metric="", predicted=round(pred, 3),
            lower=round(pred - 0.1 * scale, 3),
            upper=round(pred + 0.1 * scale, 3),
            trend="stable", method="MA",
            confidence=0.3, sample_size=len(values),
        )
    ma = _moving_average(values, window=min(window, len(values)))
    slope, intercept = _linear_trend(values)
    pred = ma + slope  # 次日预测 = 移动平均 + 斜率
    recent_std = (sum((v - ma) ** 2 for v in values[:window]) / window) ** 0.5
    band = 1.96 * max(recent_std, 0.05 * scale)  # 95% 置信区间
    trend = _classify_trend(slope, scale)
    confidence = max(0.2, min(0.9, len(values) / 20.0))
    return Forecast(
        metric="", predicted=round(pred, 3),
        lower=round(pred - band, 3),
        upper=round(pred + band, 3),
        trend=trend, method="MA+trend",
        confidence=round(confidence, 3), sample_size=len(values),
    )


def predict_next_day(date_str, lookback_days=14, metrics_to_predict=None):
    """基于历史窗口预测次日的关键指标。"""
    if metrics_to_predict is None:
        metrics_to_predict = [
            ("sentiment_index", 1.0),
            ("policy_stance_score", 0.1),
            ("article_count", 5.0),
            ("attention_entropy", 0.05),
        ]
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
    forecasts = []
    for metric_name, scale in metrics_to_predict:
        values = [float(getattr(m, metric_name) or 0) for m in history_sorted]
        f = predict_metric(values, window=5, scale=scale)
        f.metric = metric_name
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


if __name__ == "__main__":
    from src.storage import db
    from datetime import date as _date
    db.get_conn()
    today = _date.today().isoformat()
    import random
    random.seed(42)
    for i in range(15):
        d = (parse_date(today) - timedelta(days=i)).isoformat()
        repo.upsert_metric(repo.DailyMetric(
            date=d, article_count=25 + random.randint(-5, 5),
            sentiment_index=50 + (15 - i) * 0.5 + random.uniform(-2, 2),
            policy_stance_score=(15 - i) * 0.05 + random.uniform(-0.1, 0.1),
            attention_entropy=0.85 - random.uniform(0, 0.1),
        ))
    rep = predict_next_day(today, lookback_days=14)
    import json
    print(json.dumps(rep.as_dict(), ensure_ascii=False, indent=2))
