"""forecasting.backtest - walk-forward 回测, 接受任意 Forecaster.

原 forecast_backtest.py 自己实现了一遍 MA+趋势预测, 与 forecast.py
重复。本模块用注入的 Forecaster(Protocol) 替代, 任何预测算法
都可以被回测。默认 forecaster=NaiveForecaster (行为兼容)。
"""
from __future__ import annotations

from datetime import timedelta
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.utils.date_utils import parse_date
from src.utils.logger import get_logger
from .base import (
    BacktestCase, HorizonReport, BacktestReport, Forecaster,
)
from .naive import NaiveForecaster

logger = get_logger("forecasting.backtest")


class BacktestEngine:
    """walk-forward 回测, 对任意 Forecaster 报告 MAE/RMSE/bias/coverage。"""

    def __init__(self, forecaster: Optional[Forecaster] = None,
                 lookback: int = 14, value_attr: str = "sentiment_index"):
        self.forecaster = forecaster or NaiveForecaster()
        self.lookback = lookback
        self.value_attr = value_attr  # attr name on metric rows

    def _walk_forward(self, all_metrics: Sequence, target_idx: int
                      ) -> Optional[BacktestCase]:
        if target_idx < self.lookback:
            return None
        history_rows = all_metrics[target_idx - self.lookback: target_idx]
        values = np.array(
            [float(getattr(m, self.value_attr) or 0) for m in history_rows],
            dtype=np.float64,
        )
        try:
            f = self.forecaster.fit_predict(values, horizon=1)
        except Exception as e:
            logger.debug(f"forecaster 失败: {e}")
            return None
        actual = float(getattr(all_metrics[target_idx], self.value_attr) or 0)
        prev_actual = float(getattr(all_metrics[target_idx - 1], self.value_attr) or 0)
        pred_dir = 1 if f.predicted > prev_actual else (-1 if f.predicted < prev_actual else 0)
        actual_dir = 1 if actual > prev_actual else (-1 if actual < prev_actual else 0)
        return BacktestCase(
            target_date=all_metrics[target_idx].date,
            predicted=round(f.predicted, 2),
            actual=round(actual, 2),
            lower=round(f.lower, 2),
            upper=round(f.upper, 2),
            abs_error=round(abs(f.predicted - actual), 2),
            direction_correct=(pred_dir == actual_dir and pred_dir != 0),
            in_band=(f.lower <= actual <= f.upper),
        )

    @staticmethod
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
            horizon=horizon, n=n, mae=round(mae, 3),
            rmse=round(rmse, 3), bias=round(bias, 3),
            direction_accuracy=round(dir_n / n, 3),
            band_coverage=round(band_n / n, 3),
        )

    def run(self, base_date: str, n_days: int = 30,
            horizons: Tuple[int, ...] = (1, 3, 7)
            ) -> BacktestReport:
        """对 base_date 前 n_days 做 walk-forward 回测."""
        from src.storage import repository as repo
        end = (parse_date(base_date) - timedelta(days=1)).isoformat()
        start = (parse_date(base_date) - timedelta(
            days=n_days + self.lookback + max(horizons) + 1
        )).isoformat()
        metrics = repo.list_metrics(start_date=start, end_date=end)
        if len(metrics) < self.lookback + max(horizons) + 1:
            return BacktestReport(
                base_date=base_date,
                horizon_reports=[], cases=[],
                summary=f"(历史样本仅 {len(metrics)} 条, 不足 walk-forward 需要)",
            )
        metrics = sorted(metrics, key=lambda m: m.date)
        horizon_reports: List[HorizonReport] = []
        all_cases: List[BacktestCase] = []
        for h in horizons:
            cases: List[BacktestCase] = []
            for idx in range(self.lookback + h - 1, len(metrics)):
                c = self._walk_forward(metrics, idx)
                if c is not None:
                    cases.append(c)
            rep = self._aggregate(cases, h)
            if rep is not None:
                horizon_reports.append(rep)
            if h == 1:
                all_cases = cases
        if not horizon_reports:
            summary = "(所有 horizon 都样本不足)"
        else:
            best = max(horizon_reports, key=lambda r: r.direction_accuracy)
            worst = min(horizon_reports, key=lambda r: r.direction_accuracy)
            summary = (
                f"回测 {n_days} 天窗口 / {len(horizon_reports)} horizon; "
                f"最佳 horizon={best.horizon}天 (方向准确率 {best.direction_accuracy:.0%}); "
                f"最差 horizon={worst.horizon}天 ({worst.direction_accuracy:.0%}); "
                f"horizon=1 平均 MAE={horizon_reports[0].mae:.2f}; "
                f"forecaster={getattr(self.forecaster, 'name', '?')}"
            )
        return BacktestReport(
            base_date=base_date,
            horizon_reports=horizon_reports,
            cases=all_cases[:50],
            summary=summary,
        )


# ---------------------------------------------------------------------------
# Back-compat: 原 forecast_backtest.backtest() 接口
# ---------------------------------------------------------------------------


def backtest(base_date: str, n_days: int = 30,
             horizons: Tuple[int, ...] = (1, 3, 7),
             lookback: int = 14,
             forecaster: Optional[Forecaster] = None) -> BacktestReport:
    """Back-compat: replicate forecast_backtest.backtest() with any Forecaster."""
    return BacktestEngine(
        forecaster=forecaster or NaiveForecaster(), lookback=lookback,
    ).run(base_date, n_days=n_days, horizons=horizons)
