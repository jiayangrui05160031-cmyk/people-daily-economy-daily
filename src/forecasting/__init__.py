"""forecasting/ - 统一时序预测 (v9 重构)

取代之前 3 个分散的预测文件:
  - forecast.py           (简单 MA+趋势, 4 个 metric 同时跑)
  - forecast_enhanced.py  (STL + Holt-Winters + 蒙特卡洛 + 周期检测)
  - forecast_backtest.py  (walk-forward 回测, 自己又写一遍预测)

设计:
  Forecaster(Protocol)
    .fit_predict(series: np.ndarray, horizon: int) -> ForecastResult

  两个具体实现:
    - NaiveForecaster       (原 forecast.py 的 MA+趋势, 0 依赖)
    - StlHoltWinters        (原 forecast_enhanced.py)

  BacktestEngine
    对任意 Forecaster 做 walk-forward; 报告 MAE/RMSE/bias/coverage。

收益:
  - backtest 不再自带预测逻辑, 改用注入的 Forecaster
  - 新算法实现 Forecaster protocol 即可
  - 三文件合成一个子包 (660 -> ~450 行)
"""
from .base import (
    Forecaster, ForecastResult, HorizonForecast, BacktestCase,
    HorizonReport, BacktestReport, ForecastReport,
)
from .naive import NaiveForecaster
from .stl_holt_winters import StlHoltWinters
from .backtest import BacktestEngine

__all__ = [
    "Forecaster", "ForecastResult", "HorizonForecast", "BacktestCase",
    "HorizonReport", "BacktestReport", "ForecastReport",
    "NaiveForecaster", "StlHoltWinters", "BacktestEngine",
]
