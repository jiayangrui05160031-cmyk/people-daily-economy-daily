"""analysis.forecast_backtest - v9 重构: 薄 re-export

实际算法在 src.forecasting.backtest (BacktestEngine + backtest())。
backtest() 现在接受可选的 forecaster 参数 (任何 Forecaster protocol
的实现都可以), 默认 NaiveForecaster (行为与原版一致)。
"""
from src.forecasting.backtest import (
    backtest, BacktestEngine,
)
from src.forecasting.base import (
    BacktestCase, HorizonReport, BacktestReport,
)

__all__ = [
    "backtest", "BacktestEngine",
    "BacktestCase", "HorizonReport", "BacktestReport",
]


if __name__ == "__main__":
    from src.analysis.volatility import seed_demo_history
    from src.storage import db
    db.get_conn()
    seed_demo_history("2026-06-12", days=30)
    import json
    rep = backtest("2026-06-12", n_days=20, horizons=(1, 3, 7))
    print(json.dumps(rep.as_dict(), ensure_ascii=False, indent=2))
    print("[OK] forecast_backtest self-test passed")
