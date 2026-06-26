"""analysis.forecast_enhanced - v9 重构: 薄 re-export

实际算法在 src.forecasting.stl_holt_winters (StlHoltWinters, predict)。
"""
from src.forecasting.stl_holt_winters import (
    predict, StlHoltWinters,
    _stl_decompose, _holt_winters, _holt_winters_forecast,
    _best_holt_winters, _detect_period, _mc_confidence,
)
from src.forecasting.base import HorizonForecast, EnhancedForecast

__all__ = [
    "predict", "StlHoltWinters",
    "HorizonForecast", "EnhancedForecast",
    "_stl_decompose", "_holt_winters", "_holt_winters_forecast",
    "_best_holt_winters", "_detect_period", "_mc_confidence",
]


if __name__ == "__main__":
    rep = predict("2026-06-12")
    if rep:
        print(rep.summary)
        for h in rep.horizons:
            print(f"  h={h.horizon}: {h.predicted:+.2f} [{h.lower:+.2f}, {h.upper:+.2f}] "
                  f"trend={h.trend} conf={h.confidence:.0%}")
