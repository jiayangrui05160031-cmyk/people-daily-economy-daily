"""analysis.anomaly - 异常检测 (Z-score on daily metrics)

使用滚动窗口 Z-score 检测当日指标是否显著偏离历史均值:
- sentiment_index 突然看多/看空 (>2 sigma)
- article_count 异常突增/骤降
- attention_entropy 急剧聚焦 (从分散到单一主题)
- industry_breadth 突然扩大/收窄

无依赖,纯 SQL + 统计。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from src.storage import repository as repo
from src.storage.db import get_conn
from src.utils.date_utils import parse_date
from src.utils.logger import get_logger

logger = get_logger("analysis.anomaly")


@dataclass
class AnomalySignal:
    metric: str
    value: float
    mean: float
    std: float
    z_score: float
    direction: str  # 'spike_high' | 'spike_low' | 'normal'
    severity: str   # 'critical' (>3) | 'warning' (>2) | 'notice' (>1.5) | 'normal'
    description: str


@dataclass
class AnomalyReport:
    date: str
    window_size: int
    signals: List[AnomalySignal]
    overall_risk: str  # 'high' | 'medium' | 'low' | 'normal'
    summary: str

    def as_dict(self):
        d = asdict(self)
        d['signals'] = [asdict(s) for s in self.signals]
        return d


def _z_score(value, mean, std):
    if std <= 1e-9:
        return 0.0
    return (value - mean) / std


def _severity(z):
    az = abs(z)
    if az >= 3.0: return "critical"
    if az >= 2.0: return "warning"
    if az >= 1.5: return "notice"
    return "normal"


def _direction(z):
    if z >= 1.5: return "spike_high"
    if z <= -1.5: return "spike_low"
    return "normal"


def detect(date_str, window_days=30, metrics_to_check=None):
    """检测当日 vs 滚动窗口的异常。"""
    if metrics_to_check is None:
        metrics_to_check = [
            ("sentiment_index", "情绪指数", 50.0),
            ("policy_stance_score", "政策倾向", 0.0),
            ("attention_entropy", "注意力熵", 0.5),
            ("attention_top_share", "头部集中度", 0.5),
            ("article_count", "文章数", 30.0),
        ]
    today = repo.get_metric(date_str)
    if not today:
        return AnomalyReport(
            date=date_str, window_size=0, signals=[],
            overall_risk="unknown",
            summary="(无当日指标,无法做异常检测)",
        )
    from datetime import timedelta
    end_date = (parse_date(date_str) - timedelta(days=1)).isoformat()
    start_date = (parse_date(date_str) - timedelta(days=window_days)).isoformat()
    history = repo.list_metrics(start_date=start_date, end_date=end_date)
    if len(history) < 3:
        return AnomalyReport(
            date=date_str, window_size=len(history), signals=[],
            overall_risk="insufficient_data",
            summary=f"(历史样本仅 {len(history)} 天, 不足 3 天, 跳过异常检测)",
        )
    signals = []
    for metric_name, label, _default in metrics_to_check:
        hist_vals = [float(getattr(m, metric_name) or 0) for m in history]
        if not hist_vals:
            continue
        mean = sum(hist_vals) / len(hist_vals)
        var = sum((v - mean) ** 2 for v in hist_vals) / len(hist_vals)
        std = var ** 0.5
        today_val = float(getattr(today, metric_name) or 0)
        z = _z_score(today_val, mean, std)
        if abs(z) < 1.5:
            continue
        direction = _direction(z)
        severity = _severity(z)
        if direction == "spike_high":
            desc = f"{label} 突增至 {today_val:.3f} (历史均值 {mean:.3f} ± {std:.3f}, Z={z:+.2f})"
        else:
            desc = f"{label} 骤降至 {today_val:.3f} (历史均值 {mean:.3f} ± {std:.3f}, Z={z:+.2f})"
        signals.append(AnomalySignal(
            metric=metric_name, value=today_val, mean=mean, std=std,
            z_score=round(z, 3), direction=direction, severity=severity,
            description=desc,
        ))
    if not signals:
        overall = "normal"
        summary = "全部指标在历史 ±1.5 sigma 范围内,无异常。"
    else:
        max_sev = max(s.severity for s in signals)
        overall = {"critical": "high", "warning": "medium", "notice": "low"}.get(max_sev, "low")
        summary = f"检测到 {len(signals)} 项异常 (最高严重度: {max_sev}): " + "; ".join(s.description for s in signals[:3])
    return AnomalyReport(
        date=date_str, window_size=len(history),
        signals=sorted(signals, key=lambda x: abs(x.z_score), reverse=True),
        overall_risk=overall, summary=summary,
    )


if __name__ == "__main__":
    from src.storage import db
    from datetime import date as _date, timedelta
    db.get_conn()
    today = _date.today().isoformat()
    for i in range(20):
        d = (parse_date(today) - timedelta(days=i)).isoformat()
        repo.upsert_metric(repo.DailyMetric(
            date=d, article_count=20 + (i % 7),
            sentiment_index=50 + (i % 10) * 0.5,
            attention_entropy=0.85 - (i % 5) * 0.01,
            attention_top_share=0.4 + (i % 4) * 0.02,
            policy_stance_score=(i % 3 - 1) * 0.2,
        ))
    repo.upsert_metric(repo.DailyMetric(
        date=today, article_count=120,
        sentiment_index=85.0,
        attention_entropy=0.40,
        attention_top_share=0.85,
        policy_stance_score=0.95,
    ))
    rep = detect(today, window_days=20)
    import json
    print(json.dumps(rep.as_dict(), ensure_ascii=False, indent=2))
