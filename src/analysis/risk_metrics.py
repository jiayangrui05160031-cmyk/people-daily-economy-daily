"""analysis.risk_metrics - 风险指标体系 (Sharpe / Sortino / MaxDD / Calmar / VaR)

把 daily_metric 中的 sentiment_index / policy_stance_score 等量化指标
当作宏观情绪"回报序列", 计算全套风险管理指标:
  - annualized_return / annualized_vol
  - sharpe_ratio          (夏普, 风险调整收益)
  - sortino_ratio         (索提诺, 只看下行波动)
  - max_drawdown / max_dd_duration  (回撤 + 修复天数)
  - calmar_ratio          (卡玛, 年化收益 / 最大回撤)
  - var_95 / var_99       (在险价值, 历史法)
  - expected_shortfall    (期望损失, CVaR)
  - downside_capture / upside_capture (牛熊市捕获)
  - skewness / kurtosis   (偏度 / 峰度)

无外部 ML 依赖, 纯 SQL + 统计。失败时优雅返回 None。

典型用法:
    from src.analysis.risk_metrics import compute
    rep = compute("2026-06-12", lookback=120, rf_rate=0.025)
    print(rep.sharpe_ratio, rep.max_drawdown, rep.var_95)
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, asdict, field
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

from src.storage import db as db_mod
from src.utils.date_utils import parse_date
from src.utils.logger import get_logger

logger = get_logger("analysis.risk_metrics")


# ============================================================
# 数据类
# ============================================================
@dataclass
class ReturnSeries:
    metric: str
    dates: List[str] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    returns: List[float] = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


@dataclass
class RiskReport:
    date: str
    lookback_days: int
    series: List[ReturnSeries] = field(default_factory=list)
    n: int = 0
    mean_daily_return: float = 0.0
    annualized_return: float = 0.0
    annualized_vol: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_dd_duration_days: int = 0
    var_95: float = 0.0
    var_99: float = 0.0
    expected_shortfall_95: float = 0.0
    downside_capture: float = 0.0
    upside_capture: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    best_day: float = 0.0
    worst_day: float = 0.0
    positive_days: int = 0
    negative_days: int = 0
    risk_level: str = "unknown"
    summary: str = ""

    def as_dict(self):
        d = asdict(self)
        d["series"] = [s.as_dict() for s in self.series]
        return d


# ============================================================
# 工具
# ============================================================
def _to_returns(values):
    out = []
    for i in range(1, len(values)):
        prev = values[i - 1]
        cur = values[i]
        if prev == 0:
            out.append(0.0)
        else:
            out.append(cur / prev - 1.0)
    return out


def _annualized_return(daily_returns, trading_days=252):
    if not daily_returns:
        return 0.0
    cum = 1.0
    for r in daily_returns:
        cum *= (1.0 + r)
    n = len(daily_returns)
    if n == 0:
        return 0.0
    return cum ** (trading_days / n) - 1.0


def _annualized_vol(daily_returns, trading_days=252):
    if len(daily_returns) < 2:
        return 0.0
    sd = statistics.pstdev(daily_returns)
    return sd * math.sqrt(trading_days)


def _sharpe(daily_returns, rf, trading_days=252):
    if len(daily_returns) < 2:
        return 0.0
    ar = _annualized_return(daily_returns, trading_days)
    av = _annualized_vol(daily_returns, trading_days)
    if av == 0:
        return 0.0
    return (ar - rf) / av


def _sortino(daily_returns, rf, trading_days=252):
    if len(daily_returns) < 2:
        return 0.0
    ar = _annualized_return(daily_returns, trading_days)
    downside = [r for r in daily_returns if r < 0]
    if len(downside) < 2:
        return 0.0
    ds = math.sqrt(sum(r * r for r in downside) / len(downside)) * math.sqrt(trading_days)
    if ds == 0:
        return 0.0
    return (ar - rf) / ds


def _max_drawdown(returns):
    if not returns:
        return 0.0, 0
    nav = [1.0]
    for r in returns:
        nav.append(nav[-1] * (1.0 + r))
    peak = nav[0]
    max_dd = 0.0
    max_dur = 0
    cur_dur = 0
    for v in nav:
        if v >= peak:
            peak = v
            cur_dur = 0
        else:
            dd = (v - peak) / peak
            if dd < max_dd:
                max_dd = dd
            cur_dur += 1
            max_dur = max(max_dur, cur_dur)
    return max_dd, max_dur


def _calmar(ar, max_dd):
    if max_dd == 0:
        return 0.0
    return ar / abs(max_dd)


def _var(returns, q=0.05):
    if not returns:
        return 0.0
    s = sorted(returns)
    idx = max(0, int(q * len(s)) - 1)
    return s[idx]


def _expected_shortfall(returns, q=0.05):
    if not returns:
        return 0.0
    s = sorted(returns)
    cutoff = max(1, int(q * len(s)))
    tail = s[:cutoff]
    if not tail:
        return 0.0
    return sum(tail) / len(tail)


def _capture(returns, benchmark):
    up_rets = [r for r, b in zip(returns, benchmark) if b > 0]
    down_rets = [r for r, b in zip(returns, benchmark) if b < 0]
    up_cap = sum(up_rets) / len(up_rets) if up_rets else 0.0
    down_cap = sum(down_rets) / len(down_rets) if down_rets else 0.0
    return up_cap, down_cap


def _skew_kurt(values):
    if len(values) < 3:
        return 0.0, 0.0
    m = statistics.mean(values)
    sd = statistics.pstdev(values)
    if sd == 0:
        return 0.0, 0.0
    n = len(values)
    skew = sum((x - m) ** 3 for x in values) / n / (sd ** 3)
    kurt = sum((x - m) ** 4 for x in values) / n / (sd ** 4) - 3.0
    return skew, kurt


def _classify_risk(sharpe, max_dd, vol):
    if max_dd <= -0.30 or vol > 0.40 or sharpe < -0.5:
        return "extreme"
    if max_dd <= -0.15 or vol > 0.25 or sharpe < 0.0:
        return "high"
    if max_dd <= -0.08 or vol > 0.15:
        return "medium"
    return "low"


def _load_series(target_date, lookback):
    end = parse_date(target_date)
    start = end - timedelta(days=lookback)
    metrics = ("sentiment_index", "policy_stance_score", "attention_entropy",
               "industry_count", "event_count")
    out = []
    try:
        with db_mod.get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM daily_metric WHERE date BETWEEN ? AND ? ORDER BY date",
                (start.isoformat(), end.isoformat()),
            ).fetchall()
        for m in metrics:
            dates = [r["date"] for r in rows if r[m] is not None]
            values = [float(r[m]) for r in rows if r[m] is not None]
            rets = _to_returns(values)
            out.append(ReturnSeries(metric=m, dates=dates, values=values, returns=rets))
    except Exception as e:
        logger.debug(f"load series 失败: {e}")
    return out


def compute(target_date="2026-06-12", lookback=120, rf_rate=0.025):
    series_list = _load_series(target_date, lookback)
    if not series_list or not series_list[0].returns:
        return None
    main = series_list[0]
    rets = main.returns
    n = len(rets)
    mean_r = statistics.mean(rets) if rets else 0.0
    ar = _annualized_return(rets)
    av = _annualized_vol(rets)
    sharpe = _sharpe(rets, rf_rate)
    sortino = _sortino(rets, rf_rate)
    max_dd, max_dd_dur = _max_drawdown(rets)
    calmar = _calmar(ar, max_dd)
    var95 = _var(rets, 0.05)
    var99 = _var(rets, 0.01)
    es95 = _expected_shortfall(rets, 0.05)
    up_cap, down_cap = (0.0, 0.0)
    bench = None
    for s in series_list:
        if s.metric == "attention_entropy":
            bench = s.returns
            break
    if bench and len(bench) == len(rets):
        up_cap, down_cap = _capture(rets, bench)
    skew, kurt = _skew_kurt(rets)
    pos_days = sum(1 for r in rets if r > 0)
    neg_days = sum(1 for r in rets if r < 0)
    best = max(rets) if rets else 0.0
    worst = min(rets) if rets else 0.0
    level = _classify_risk(sharpe, max_dd, av)
    summary = (
        f"近 {n} 个交易日: 年化收益 {ar:+.2%} / 波动 {av:.2%} / 夏普 {sharpe:+.2f} / "
        f"索提诺 {sortino:+.2f} / 最大回撤 {max_dd:+.2%} / VaR95 {var95:+.2%} / 风险等级 {level}"
    )
    return RiskReport(
        date=target_date, lookback_days=n, series=series_list,
        n=n, mean_daily_return=round(mean_r, 5),
        annualized_return=round(ar, 4), annualized_vol=round(av, 4),
        sharpe_ratio=round(sharpe, 3), sortino_ratio=round(sortino, 3),
        calmar_ratio=round(calmar, 3),
        max_drawdown=round(max_dd, 4), max_dd_duration_days=max_dd_dur,
        var_95=round(var95, 4), var_99=round(var99, 4),
        expected_shortfall_95=round(es95, 4),
        downside_capture=round(down_cap, 4),
        upside_capture=round(up_cap, 4),
        skewness=round(skew, 3), kurtosis=round(kurt, 3),
        best_day=round(best, 4), worst_day=round(worst, 4),
        positive_days=pos_days, negative_days=neg_days,
        risk_level=level, summary=summary,
    )


if __name__ == "__main__":
    import json
    rep = compute("2026-06-12", lookback=60, rf_rate=0.025)
    if rep is None:
        print("(无时序数据, 先跑 smoke_test 注入 mock)")
    else:
        d = rep.as_dict()
        d["series"] = f"[{len(d['series'])} series omitted]"
        print(json.dumps(d, ensure_ascii=False, indent=2))
        assert rep.risk_level in ("low", "medium", "high", "extreme")
        print("[OK] risk_metrics self-test passed")
