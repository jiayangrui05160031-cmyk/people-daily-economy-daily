"""forecasting.stl_holt_winters - STL 分解 + Holt-Winters + 蒙特卡洛置信区间

(原 forecast_enhanced.py 算法, 0 依赖: numpy + Python stdlib)
"""
from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .base import (
    ForecastResult, HorizonForecast, EnhancedForecast, Forecaster,
)


# ============================================================
# STL 分解 (加法)
# ============================================================


def _stl_decompose(series: np.ndarray, period: int = 7) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(series)
    if n < period * 2:
        return series.copy(), np.zeros(n), np.zeros(n)
    half = period // 2
    trend = np.full(n, np.nan)
    for i in range(half, n - half):
        trend[i] = np.mean(series[i - half: i + half + 1])
    if half > 0:
        for i in range(half):
            trend[i] = trend[half]
            trend[n - 1 - i] = trend[n - 1 - half]
    detrended = series - trend
    seasonal = np.zeros(n)
    for s in range(period):
        idx = np.arange(s, n, period)
        if len(idx) > 0:
            val = np.mean(detrended[idx])
            for i in idx:
                seasonal[i] = val
    seasonal -= np.mean(seasonal)
    residual = series - trend - seasonal
    return trend, seasonal, residual


# ============================================================
# Holt-Winters 加法
# ============================================================


def _holt_winters(series: np.ndarray, season_len: int = 7,
                  alpha: float = 0.3, beta: float = 0.1, gamma: float = 0.2
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(series)
    if n < season_len * 2:
        s = np.zeros(n); t = np.zeros(n); l = series.copy()
        return l, t, s
    level = np.zeros(n); trend_arr = np.zeros(n); seasonal = np.zeros(n)
    seasonals_init = series[:season_len] - np.mean(series[:season_len])
    for i in range(season_len):
        seasonal[i] = seasonals_init[i]
    level[season_len - 1] = np.mean(series[:season_len])
    trend_arr[season_len - 1] = (
        np.mean(series[season_len:2 * season_len]) - np.mean(series[:season_len])
    ) / season_len
    for i in range(season_len, n):
        lvl_prev = level[i - 1]
        trd_prev = trend_arr[i - 1]
        seas_prev = seasonal[i - season_len]
        level[i] = alpha * (series[i] - seas_prev) + (1 - alpha) * (lvl_prev + trd_prev)
        trend_arr[i] = beta * (level[i] - lvl_prev) + (1 - beta) * trd_prev
        seasonal[i] = gamma * (series[i] - level[i]) + (1 - gamma) * seas_prev
    return level, trend_arr, seasonal


def _holt_winters_forecast(level: float, trend: float, seasonals: np.ndarray,
                            h: int, season_len: int) -> float:
    season_idx = (h - 1) % season_len
    return level + h * trend + seasonals[season_idx]


def _best_holt_winters(series: np.ndarray, season_len: int
                       ) -> Tuple[float, float, float, float]:
    best = (0.3, 0.1, 0.2, float("inf"))
    for a in (0.1, 0.3, 0.5, 0.7):
        for b in (0.05, 0.1, 0.3):
            for g in (0.1, 0.3, 0.5):
                lvl, trd, seas = _holt_winters(series, season_len, a, b, g)
                n = len(series)
                if n < season_len + 1:
                    continue
                errs = []
                for i in range(season_len, n):
                    pred = _holt_winters_forecast(lvl[i - 1], trd[i - 1], seas, 1, season_len)
                    errs.append(abs(pred - series[i]))
                mae = float(np.mean(errs)) if errs else float("inf")
                if mae < best[3]:
                    best = (a, b, g, mae)
    return best[0], best[1], best[2], best[3]


def _detect_period(series: np.ndarray) -> int:
    if len(series) < 30:
        return 7
    n = len(series)
    detrended = series - np.mean(series)
    candidates = (5, 7, 14, 30)
    best_p = 7
    best_ac = 0.0
    for p in candidates:
        if p >= n:
            continue
        ac = np.corrcoef(detrended[:-p], detrended[p:])[0, 1] if n - p > 1 else 0
        if not np.isnan(ac) and abs(ac) > abs(best_ac):
            best_ac = ac
            best_p = p
    return best_p


def _mc_confidence(series: np.ndarray, level: float, trend: float,
                    seasonals: np.ndarray, h: int, season_len: int,
                    n_sims: int = 200, seed: int = 42) -> Tuple[float, float]:
    rng = random.Random(seed)
    n = len(series)
    if n < 14:
        sd = float(np.std(series[-14:])) if n >= 2 else 1.0
        band = 1.96 * sd * (h ** 0.5)
        return -band, band
    _, _, seas = _holt_winters(series, season_len)
    fitted = level + np.arange(n) * trend + np.array(
        [seas[i % season_len] for i in range(n)]
    )
    resid = series - fitted
    sd = float(np.std(resid)) or 1.0
    sims = []
    for _ in range(n_sims):
        noise = sum(rng.gauss(0, sd) for _ in range(h))
        sims.append(_holt_winters_forecast(level, trend, seasonals, h, season_len) + noise)
    sims.sort()
    lo = sims[int(0.025 * n_sims)]
    hi = sims[int(0.975 * n_sims) - 1]
    return float(lo), float(hi)


# ============================================================
# Forecaster 实现
# ============================================================


class StlHoltWinters:
    """STL 分解 + Holt-Winters + 蒙特卡洛置信区间."""

    name = "stl_holt_winters"

    def __init__(self, season_len: Optional[int] = None):
        self.season_len = season_len
        self._alpha = self._beta = self._gamma = 0.0
        self._period = 0
        self._level = self._trend = 0.0
        self._seasonals: np.ndarray = np.zeros(0)
        self._fitted = False

    def _ensure_fitted(self, series: np.ndarray) -> None:
        if self._fitted:
            return
        self._period = self.season_len or _detect_period(series)
        self._alpha, self._beta, self._gamma, _ = _best_holt_winters(series, self._period)
        lvl, trd, seas = _holt_winters(
            series, self._period, self._alpha, self._beta, self._gamma,
        )
        self._level = float(lvl[-1])
        self._trend = float(trd[-1])
        self._seasonals = seas[-self._period:].copy()
        self._fitted = True

    def fit_predict(self, series: np.ndarray, horizon: int = 1,
                    **kwargs) -> ForecastResult:
        if series is None or len(series) < 14:
            return ForecastResult(0, 0, 0, "naive", "stable", 0.0, sample_size=len(series) if series is not None else 0)
        self._ensure_fitted(series)
        pred = _holt_winters_forecast(
            self._level, self._trend, self._seasonals, horizon, self._period,
        )
        lo, hi = _mc_confidence(
            series, self._level, self._trend, self._seasonals,
            horizon, self._period,
        )
        if self._trend > 0.1:
            trend = "rising"
        elif self._trend < -0.1:
            trend = "falling"
        else:
            trend = "stable"
        width = hi - lo
        rel = width / max(abs(pred), 1.0)
        conf = max(0.0, min(1.0, 1.0 - rel * 0.5))
        return ForecastResult(
            predicted=round(float(pred), 3),
            lower=round(float(lo), 3),
            upper=round(float(hi), 3),
            method="holt-winters", trend=trend,
            confidence=round(conf, 3),
            sample_size=len(series),
            seasonal_component=round(
                float(self._seasonals[(horizon - 1) % self._period]), 3,
            ),
        )


# ============================================================
# Back-compat: 原 forecast_enhanced.predict() 接口
# ============================================================


def predict(target_date: str, metric: str = "sentiment_index",
            lookback: int = 90,
            horizons: Tuple[int, ...] = (1, 3, 7, 14)
            ) -> Optional[EnhancedForecast]:
    """Back-compat: replicate forecast_enhanced.predict()."""
    from src.storage import db as db_mod
    from src.utils.date_utils import parse_date
    from datetime import timedelta

    end = parse_date(target_date)
    start = end - timedelta(days=lookback)
    try:
        with db_mod.get_conn() as conn:
            rows = conn.execute(
                f"SELECT date, {metric} FROM daily_metric "
                f"WHERE date BETWEEN ? AND ? ORDER BY date",
                (start.isoformat(), end.isoformat()),
            ).fetchall()
        if len(rows) < 14:
            return None
        series = np.array([float(r[metric] or 50.0) for r in rows], dtype=np.float64)
    except Exception:
        return None
    period = _detect_period(series)
    alpha, beta, gamma, mae = _best_holt_winters(series, period)
    lvl, trd, seas = _holt_winters(series, period, alpha, beta, gamma)
    cur_level = float(lvl[-1])
    cur_trend = float(trd[-1])
    cur_seas = seas[-period:].copy()
    in_sample_errs = []
    for i in range(period, len(series)):
        pred = _holt_winters_forecast(lvl[i - 1], trd[i - 1], seas, 1, period)
        in_sample_errs.append(abs(pred - series[i]))
    in_sample_mae = float(np.mean(in_sample_errs)) if in_sample_errs else 0.0
    in_sample_mape = float(np.mean(
        [abs(e / s) for e, s in zip(in_sample_errs, series[period:]) if s != 0]
    )) * 100 if in_sample_errs else 0.0
    if cur_trend > 0.1:
        trend_dir = "rising"
    elif cur_trend < -0.1:
        trend_dir = "falling"
    else:
        trend_dir = "stable"
    horizon_fcs: List[HorizonForecast] = []
    for h in horizons:
        pred = _holt_winters_forecast(cur_level, cur_trend, cur_seas, h, period)
        lo, hi = _mc_confidence(series, cur_level, cur_trend, cur_seas, h, period)
        future_lvl = cur_level + h * cur_trend
        if future_lvl > cur_level + 1:
            h_trend = "rising"
        elif future_lvl < cur_level - 1:
            h_trend = "falling"
        else:
            h_trend = "stable"
        width = hi - lo
        rel = width / max(abs(pred), 1.0)
        conf = max(0.0, min(1.0, 1.0 - rel * 0.5))
        horizon_fcs.append(HorizonForecast(
            horizon=h, predicted=round(float(pred), 3),
            lower=round(float(lo), 3), upper=round(float(hi), 3),
            method="holt-winters", confidence=round(conf, 3),
            trend=h_trend,
            seasonal_component=round(float(cur_seas[(h - 1) % period]), 3),
        ))
    summary = (
        f"{metric} Holt-Winters({alpha:.1f}/{beta:.2f}/{gamma:.1f}) 周期={period}; "
        f"in-sample MAE={in_sample_mae:.2f} MAPE={in_sample_mape:.1f}%; "
        f"趋势 {trend_dir} (slope={cur_trend:+.3f}); "
        + " | ".join(
            f"h={h.horizon}: {h.predicted:+.1f} [{h.lower:+.1f}, {h.upper:+.1f}]"
            for h in horizon_fcs
        )
    )
    return EnhancedForecast(
        base_date=target_date,
        target_metric=metric,
        horizons=horizon_fcs,
        in_sample_mae=round(in_sample_mae, 3),
        in_sample_mape=round(in_sample_mape, 3),
        seasonality_period=period,
        trend_slope=round(cur_trend, 4),
        method_selected="holt-winters",
        model_params={"alpha": alpha, "beta": beta, "gamma": gamma, "period": period},
        summary=summary,
    )
