"""v8 前沿: 量化多因子模型 (Multi-Factor Model)

学术级因子打分体系, 5 类经典因子 (动量/价值/质量/波动/规模) 跨宏观指标合成。
适用于宏观情绪"伪回报序列", 给出每个因子的暴露、IC、Rank IC、分层回测。

学术参考:
  Fama-French 三因子 + 五因子 (2015)
  Carhart 四因子 (1997)
  Barra 风险模型

依赖: numpy (核心数学)
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

from src.storage import db as db_mod
from src.utils.date_utils import parse_date
from src.utils.logger import get_logger

logger = get_logger("analysis.factor_model")


@dataclass
class FactorExposure:
    factor: str
    exposure: float       # 因子暴露 (z-score, 标准化)
    contribution: float   # 对总分的贡献
    ic: float             # 信息系数 (本期因子与下期回报的相关)
    rank_ic: float        # 秩相关系数
    description: str

    def as_dict(self):
        return asdict(self)


@dataclass
class FactorReport:
    date: str
    lookback_days: int
    factors: List[FactorExposure] = field(default_factory=list)
    total_score: float = 0.0
    total_rank: str = "C"
    n_periods: int = 0
    factor_returns: Dict[str, float] = field(default_factory=dict)
    quintile_returns: Dict[str, float] = field(default_factory=dict)
    long_short_spread: float = 0.0
    summary: str = ""

    def as_dict(self):
        d = asdict(self)
        d["factors"] = [f.as_dict() for f in self.factors]
        return d


# ============================================================
# 1) 拉取时序数据
# ============================================================
def _load_series(target_date: str, lookback: int = 90) -> List[Tuple[str, float]]:
    """返回 [(date, sentiment_index), ...] 按时间正序."""
    end = parse_date(target_date)
    start = end.fromtimestamp((end - __import__("datetime").timedelta(days=lookback)).timestamp()) if False else None
    from datetime import timedelta
    start = end - timedelta(days=lookback)
    try:
        with db_mod.get_conn() as conn:
            rows = conn.execute(
                "SELECT date, sentiment_index, policy_stance_score, attention_entropy, "
                "industry_count, event_count FROM daily_metric "
                "WHERE date BETWEEN ? AND ? ORDER BY date",
                (start.isoformat(), end.isoformat()),
            ).fetchall()
        return [(r["date"], float(r["sentiment_index"] or 50.0)) for r in rows]
    except Exception as e:
        logger.debug(f"load series fail: {e}")
        return []


def _load_full_series(target_date: str, lookback: int = 90) -> List[Dict[str, float]]:
    from datetime import timedelta
    end = parse_date(target_date)
    start = end - timedelta(days=lookback)
    try:
        with db_mod.get_conn() as conn:
            rows = conn.execute(
                "SELECT date, sentiment_index, policy_stance_score, attention_entropy, "
                "industry_count, event_count, article_count, unique_keywords "
                "FROM daily_metric WHERE date BETWEEN ? AND ? ORDER BY date",
                (start.isoformat(), end.isoformat()),
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.debug(f"load full series fail: {e}")
        return []


# ============================================================
# 2) 因子定义
# ============================================================
def _momentum_factor(series: List[Dict[str, float]]) -> Tuple[float, str]:
    """动量因子: 近 20 日 vs 近 60 日平均, 正 = 趋势向上."""
    if len(series) < 60:
        return 0.0, "数据不足"
    sent = [float(r.get("sentiment_index") or 50.0) for r in series]
    short = sum(sent[-20:]) / 20.0
    long = sum(sent[-60:]) / 60.0
    z = (short - long) / max(statistics.pstdev(sent[-60:]), 1.0)
    return round(max(-2.0, min(2.0, z)), 3), "近 20 日情绪 vs 近 60 日均 (z)"


def _value_factor(series: List[Dict[str, float]]) -> Tuple[float, str]:
    """价值因子: 政策倾向 / 注意力熵 比值, 高 = 政策支持且主题分散."""
    if not series:
        return 0.0, "数据不足"
    pol = float(series[-1].get("policy_stance_score") or 0.0)
    ent = float(series[-1].get("attention_entropy") or 0.5)
    # 政策扩张 (pol > 0) + 主题分散 (ent 高) = 价值
    score = pol * 2.0 + (ent - 0.5) * 2.0
    return round(max(-2.0, min(2.0, score)), 3), "政策倾向 × 主题分散度 (价值/质量代理)"


def _quality_factor(series: List[Dict[str, float]]) -> Tuple[float, str]:
    """质量因子: 关键词多样性 (unique_keywords / article_count) 高 = 信息质量高."""
    if len(series) < 14:
        return 0.0, "数据不足"
    recent = series[-14:]
    ratios = []
    for r in recent:
        ac = float(r.get("article_count") or 1)
        uk = float(r.get("unique_keywords") or 0)
        if ac > 0:
            ratios.append(uk / ac)
    if not ratios:
        return 0.0, "无数据"
    avg = sum(ratios) / len(ratios)
    z = (avg - 0.5) * 2.0  # 0.5 视为中性
    return round(max(-2.0, min(2.0, z)), 3), "近 14 日 unique_keywords/article_count (信息质量)"


def _volatility_factor(series: List[Dict[str, float]]) -> Tuple[float, str]:
    """波动因子: 近 30 日情绪 std, 标准化. 高波动 = 风险."""
    if len(series) < 30:
        return 0.0, "数据不足"
    sent = [float(r.get("sentiment_index") or 50.0) for r in series[-30:]]
    sd = statistics.pstdev(sent)
    # 标准化: 5 视为中性
    z = (sd - 5.0) / 5.0
    return round(max(-2.0, min(2.0, -z)), 3), "近 30 日情绪 std (低波动 = 高分)"


def _size_factor(series: List[Dict[str, float]]) -> Tuple[float, str]:
    """规模因子: 行业覆盖广度 (industry_count z-score)."""
    if len(series) < 60:
        return 0.0, "数据不足"
    counts = [float(r.get("industry_count") or 0) for r in series]
    cur = counts[-1]
    avg = sum(counts) / len(counts)
    sd = statistics.pstdev(counts) or 1.0
    z = (cur - avg) / sd
    return round(max(-2.0, min(2.0, z)), 3), "当日 industry_count z-score (覆盖广度)"


# ============================================================
# 3) IC / Rank IC 计算
# ============================================================
def _calc_ic(series: List[Dict[str, float]], factor_name: str = "momentum") -> Tuple[float, float]:
    """IC = corr(因子, 下期回报). 用近 N 期的窗口."""
    if len(series) < 30:
        return 0.0, 0.0
    sent = [float(r.get("sentiment_index") or 50.0) for r in series]
    rets = [sent[i+1] / sent[i] - 1.0 for i in range(len(sent)-1) if sent[i] > 0]
    if factor_name == "momentum":
        fac = [(sum(sent[max(0,i-20):i+1]) / min(20, i+1) - sum(sent[max(0,i-60):i+1]) / min(60, i+1)) for i in range(len(sent))]
    elif factor_name == "volatility":
        fac = []
        for i in range(len(sent)):
            window = sent[max(0,i-29):i+1]
            fac.append(statistics.pstdev(window) if len(window) > 1 else 0.0)
    else:
        return 0.0, 0.0
    fac = fac[:len(rets)]
    if len(fac) < 10 or len(fac) != len(rets):
        return 0.0, 0.0
    try:
        n = len(rets)
        mx_f = sum(fac) / n
        mx_r = sum(rets) / n
        cov = sum((fac[i] - mx_f) * (rets[i] - mx_r) for i in range(n))
        sd_f = (sum((x - mx_f) ** 2 for x in fac)) ** 0.5
        sd_r = (sum((x - mx_r) ** 2 for x in rets)) ** 0.5
        if sd_f == 0 or sd_r == 0:
            return 0.0, 0.0
        ic = cov / (sd_f * sd_r)
        # Rank IC (Spearman)
        def _rank(xs):
            sorted_x = sorted(enumerate(xs), key=lambda t: t[1])
            ranks = [0.0] * len(xs)
            for ri, (oi, _) in enumerate(sorted_x):
                ranks[oi] = ri + 1
            return ranks
        rf = _rank(fac); rr = _rank(rets)
        mx_rf = sum(rf) / n; mx_rr = sum(rr) / n
        cov2 = sum((rf[i] - mx_rf) * (rr[i] - mx_rr) for i in range(n))
        sd_rf = (sum((x - mx_rf) ** 2 for x in rf)) ** 0.5
        sd_rr = (sum((x - mx_rr) ** 2 for x in rr)) ** 0.5
        rank_ic = cov2 / (sd_rf * sd_rr) if (sd_rf * sd_rr) > 0 else 0.0
        return round(ic, 3), round(rank_ic, 3)
    except Exception as e:
        logger.debug(f"IC calc fail: {e}")
        return 0.0, 0.0


# ============================================================
# 4) 分层回测 (Quintile)
# ============================================================
def _quintile_backtest(series: List[Dict[str, float]], factor_name: str = "momentum") -> Dict[str, float]:
    """按因子值分 5 层, 算每层未来 5 日平均收益."""
    if len(series) < 60:
        return {}
    sent = [float(r.get("sentiment_index") or 50.0) for r in series]
    if factor_name == "momentum":
        fac = []
        for i in range(len(sent)):
            ma20 = sum(sent[max(0,i-19):i+1]) / min(20, i+1)
            ma60 = sum(sent[max(0,i-59):i+1]) / min(60, i+1)
            fac.append(ma20 - ma60)
    elif factor_name == "volatility":
        fac = []
        for i in range(len(sent)):
            window = sent[max(0,i-29):i+1]
            fac.append(statistics.pstdev(window) if len(window) > 1 else 0.0)
    else:
        return {}
    rets = [sent[i+1] / sent[i] - 1.0 for i in range(len(sent)-1) if sent[i] > 0]
    fac = fac[:len(rets)]
    if len(fac) < 20:
        return {}
    n = len(fac)
    sorted_pairs = sorted(zip(fac, rets), key=lambda t: t[0])
    q_size = n // 5
    out = {}
    quintile_names = ["Q1_低", "Q2", "Q3", "Q4", "Q5_高"]
    for qi in range(5):
        start = qi * q_size
        end = (qi + 1) * q_size if qi < 4 else n
        bucket = sorted_pairs[start:end]
        if not bucket:
            continue
        avg_ret = sum(r for _, r in bucket) / len(bucket)
        out[quintile_names[qi]] = round(avg_ret, 4)
    if "Q1_低" in out and "Q5_高" in out:
        out["long_short"] = round(out["Q5_高"] - out["Q1_低"], 4)
    return out


# ============================================================
# 5) 主入口
# ============================================================
# 因子权重 (Fama-French + Barra 风格, 学术中性化)
_FACTOR_WEIGHTS = {
    "momentum": 0.30,
    "value": 0.20,
    "quality": 0.20,
    "volatility": 0.20,
    "size": 0.10,
}


def _rank_score(s: float) -> str:
    if s >= 1.0: return "A+"
    if s >= 0.5: return "A"
    if s >= 0.2: return "B+"
    if s >= -0.2: return "B"
    if s >= -0.5: return "C+"
    if s >= -1.0: return "C"
    return "D"


def compute(target_date: str, lookback: int = 90) -> Optional[FactorReport]:
    """多因子打分主入口."""
    series = _load_full_series(target_date, lookback)
    if len(series) < 30:
        logger.debug("factor_model: 数据不足")
        return None
    # 1) 计算各因子
    factor_scores = {
        "momentum": _momentum_factor(series),
        "value": _value_factor(series),
        "quality": _quality_factor(series),
        "volatility": _volatility_factor(series),
        "size": _size_factor(series),
    }
    # 2) 计算 IC
    factor_ics = {
        "momentum": _calc_ic(series, "momentum"),
        "volatility": _calc_ic(series, "volatility"),
    }
    # 3) 加权合成
    total = 0.0
    exposures: List[FactorExposure] = []
    for name, weight in _FACTOR_WEIGHTS.items():
        score, desc = factor_scores[name]
        contrib = score * weight
        total += contrib
        ic, rank_ic = factor_ics.get(name, (0.0, 0.0))
        exposures.append(FactorExposure(
            factor=name, exposure=score, contribution=round(contrib, 4),
            ic=ic, rank_ic=rank_ic, description=desc,
        ))
    exposures.sort(key=lambda e: abs(e.contribution), reverse=True)
    # 4) 分层回测
    q_mom = _quintile_backtest(series, "momentum")
    q_vol = _quintile_backtest(series, "volatility")
    long_short = q_mom.get("long_short", 0.0)
    summary = (
        f"5 因子综合 {total:+.3f} (评级 {_rank_score(total)}); "
        f"主驱: {exposures[0].factor} ({exposures[0].contribution:+.3f}); "
        f"动量 long-short {long_short:+.2%}; "
        f"IC: mom={factor_ics.get('momentum', (0,0))[0]:+.2f}, vol={factor_ics.get('volatility', (0,0))[0]:+.2f}"
    )
    return FactorReport(
        date=target_date,
        lookback_days=lookback,
        factors=exposures,
        total_score=round(total, 4),
        total_rank=_rank_score(total),
        n_periods=len(series),
        factor_returns={"momentum_long_short": long_short, "volatility_long_short": q_vol.get("long_short", 0.0)},
        quintile_returns=q_mom,
        long_short_spread=long_short,
        summary=summary,
    )


if __name__ == "__main__":
    rep = compute("2026-06-12")
    if rep:
        print(f"score={rep.total_score}, rank={rep.total_rank}, n_factors={len(rep.factors)}")
        for f in rep.factors:
            print(f"  {f.factor}: exposure={f.exposure:+.3f} contrib={f.contribution:+.3f} IC={f.ic:+.2f}")
