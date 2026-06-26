"""analysis.forecast - v9 重构: 薄 re-export

实际算法在 src.forecasting.naive (NaiveForecaster, predict_metric,
predict_next_day)。本文件保留 import 路径, 让所有旧 caller 零改动。
"""
from src.forecasting.naive import (
    predict_metric, predict_next_day, NaiveForecaster,
    _moving_average, _linear_trend, _classify_trend,
)
from src.forecasting.base import Forecast, ForecastResult, ForecastReport

__all__ = [
    "predict_metric", "predict_next_day", "NaiveForecaster",
    "Forecast", "ForecastResult", "ForecastReport",
    "_moving_average", "_linear_trend", "_classify_trend",
]


if __name__ == "__main__":
    from src.storage import db
    from datetime import date as _date
    db.get_conn()
    today = _date.today().isoformat()
    import random
    random.seed(42)
    for i in range(15):
        from src.utils.date_utils import parse_date
        from src.storage import repository as repo
        from datetime import timedelta
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
