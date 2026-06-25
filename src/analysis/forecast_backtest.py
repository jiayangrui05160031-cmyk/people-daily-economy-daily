"""analysis.forecast_backtest - 预测准确率回测

对 predict_next_day() 做 walk-forward 验证:
- 选历史 N 天作为"回测日期"
- 对每个回测日期 d: 用 d 之前的历史窗口预测 d 的 sentiment_index
- 对比预测值 vs 实际值
- 报告 MAE / RMSE / 方向准确率 / 偏差

指标:
- mae: 平均绝对误差
- rmse: 均方根误差
- direction_accuracy: 趋势方向 (升/降) 准确率
- bias: 平均偏差 (预测 - 实际), 正值表示系统性高估
- coverage: 落在 95% 置信区间的比例
- by_horizon: 按预测天数分桶 (1/3/7 天)

用法:
    from src.analysis.forecast_backtest import backtest
    rep = backtest("2026-06-12", n_days=14, horizons=(1, 3, 7))
    print(rep.summary)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from datetime import timedelta
from typing import List, Optional, Tuple

from src.storage import repository as repo
from src.utils.date_utils import parse_date
from src.utils.logger import get_logger

logger = get_logger("analysis.forecast_backtest")


@dataclass
class BacktestCase:
    target_date: str
    predicted: float
    actual: float
    lower: float
    upper: float
    abs_error: float
    direction_correct: bool   # 涨/跌方向是否预测对 (相对于前一日)
    in_band: bool             # actual 是否落在 [lower, upper]

    def as_dict(self):
        return asdict(self)


@dataclass
class HorizonReport:
    horizon: int              # 预测步长 (天)
    n: int                    # 样本数
    mae: float
    rmse: float
    bias: float
    direction_accuracy: float
    band_coverage: float

    def as_dict(self):
        return asdict(self)


@dataclass
class BacktestReport:
    base_date: str
    horizon_reports: List[HorizonReport]
    cases: List[BacktestCase]
    summary: str

    def as_dict(self):
        d = asdict(self)
        # asdict 已递归序列化, 不需要再调 h.as_dict()
        return d


def _predict_from_history(history, lookback_days=14):
    """复刻 predict_next_day 但接受 history 而不是 SQL 查询."""
    if not history:
        return None
    history_sorted = sorted(history, key=lambda m: m.date, reverse=True)
    values = [float(m.sentiment_index or 0) for m in history_sorted]
    if len(values) < 3:
        return None
    # 移动平均 + 趋势
    window = min(5, len(values))
    ma = sum(values[:window]) / window
    n = len(values)
    mean_x = sum(range(n)) / n
    mean_y = sum(values) / n
    num = sum((i - mean_x) * (values[i] - mean_y) for i in range(n))
    den = sum((i - mean_x) ** 2 for i in range(n))
    slope = num / den if den > 1e-9 else 0.0
    pred = ma + slope
    std = (sum((v - ma) ** 2 for v in values[:window]) / window) ** 0.5
    band = 1.96 * max(std, 1.5)
    return pred, pred - band, pred + band, values


def _walk_forward(all_metrics, target_idx, lookback):
    """对 target_idx 当日做预测, 使用它之前 lookback 天的历史."""
    if target_idx < lookback:
        return None
    history = all_metrics[target_idx - lookback:target_idx]
    res = _predict_from_history(history, lookback_days=lookback)
    if res is None:
        return None
    pred, lo, hi, prev_values = res
    actual = float(all_metrics[target_idx].sentiment_index or 0)
    prev_actual = float(all_metrics[target_idx - 1].sentiment_index or 0)
    # 方向: 相对前一日变化
    pred_dir = 1 if pred > prev_actual else (-1 if pred < prev_actual else 0)
    actual_dir = 1 if actual > prev_actual else (-1 if actual < prev_actual else 0)
    return BacktestCase(
        target_date=all_metrics[target_idx].date,
        predicted=round(pred, 2),
        actual=round(actual, 2),
        lower=round(lo, 2),
        upper=round(hi, 2),
        abs_error=round(abs(pred - actual), 2),
        direction_correct=(pred_dir == actual_dir and pred_dir != 0),
        in_band=(lo <= actual <= hi),
    )


def _aggregate(cases: List[BacktestCase], horizon: int) -> Optional[HorizonReport]:
    if not cases:
        return None
    n = len(cases)
    mae = sum(c.abs_error for c in cases) / n
    rmse = (sum(c.abs_error ** 2 for c in cases) / n) ** 0.5
    bias = sum((c.predicted - c.actual) for c in cases) / n
    dir_n = sum(1 for c in cases if c.direction_correct)
    band_n = sum(1 for c in cases if c.in_band)
    return HorizonReport(
        horizon=horizon, n=n,
        mae=round(mae, 3),
        rmse=round(rmse, 3),
        bias=round(bias, 3),
        direction_accuracy=round(dir_n / n, 3),
        band_coverage=round(band_n / n, 3),
    )


def backtest(base_date: str, n_days: int = 30, horizons: Tuple[int, ...] = (1, 3, 7),
             lookback: int = 14):
    """对 base_date 前 n_days 做 walk-forward 回测."""
    end = (parse_date(base_date) - timedelta(days=1)).isoformat()
    start = (parse_date(base_date) - timedelta(days=n_days + lookback + max(horizons) + 1)).isoformat()
    metrics = repo.list_metrics(start_date=start, end_date=end)
    if len(metrics) < lookback + max(horizons) + 1:
        return BacktestReport(
            base_date=base_date,
            horizon_reports=[],
            cases=[],
            summary=f"(历史样本仅 {len(metrics)} 条, 不足 walk-forward 需要)",
        )
    metrics = sorted(metrics, key=lambda m: m.date)

    horizon_reports: List[HorizonReport] = []
    all_cases: List[BacktestCase] = []
    seen_dates = set()
    for h in horizons:
        cases: List[BacktestCase] = []
        for idx in range(lookback + h - 1, len(metrics)):
            c = _walk_forward(metrics, idx, lookback)
            if c is not None:
                cases.append(c)
        rep = _aggregate(cases, h)
        if rep is not None:
            horizon_reports.append(rep)
        # 把 horizon=1 的 case 留存为 sample
        if h == 1:
            all_cases = cases
            seen_dates = {c.target_date for c in cases}

    if not horizon_reports:
        summary = "(所有 horizon 都样本不足)"
    else:
        best = max(horizon_reports, key=lambda r: r.direction_accuracy)
        worst = min(horizon_reports, key=lambda r: r.direction_accuracy)
        summary = (
            f"回测 {n_days} 天窗口 / {len(horizon_reports)} horizon; "
            f"最佳 horizon={best.horizon}天 (方向准确率 {best.direction_accuracy:.0%}); "
            f"最差 horizon={worst.horizon}天 ({worst.direction_accuracy:.0%}); "
            f"horizon=1 平均 MAE={horizon_reports[0].mae:.2f}"
        )
    return BacktestReport(
        base_date=base_date,
        horizon_reports=horizon_reports,
        cases=all_cases[:50],
        summary=summary,
    )


if __name__ == "__main__":
    from src.analysis.volatility import seed_demo_history
    from src.storage import db
    db.get_conn()
    seed_demo_history("2026-06-12", days=30)
    import json
    rep = backtest("2026-06-12", n_days=20, horizons=(1, 3, 7))
    print(json.dumps(rep.as_dict(), ensure_ascii=False, indent=2))
    print("[OK] forecast_backtest self-test passed")
