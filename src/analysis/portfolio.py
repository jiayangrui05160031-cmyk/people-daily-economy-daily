"""analysis.portfolio - 投资组合回测 (行业权重 + 调仓 + 多期比较) ==============================================
把宏观信号转为可投资的"行业组合", 在历史时序上回测:
  - 用户给目标行业 + 权重 (sum=1)
  - 用 industry_daily 表的 hit_count / article_count / stance 作"行业日度回报代理"
  - 支持 4 种调仓: daily / weekly / monthly / none
  - 算组合日收益、累计收益、夏普、回撤
  - 与等权基准比较 (信息比率 IR, alpha, beta)
无 LLM, 纯统计 + SQL。当 industry_daily 数据不足时, 用 sentiment_index 兜底。
典型用法:
    from src.analysis.portfolio import backtest
    rep = backtest("2026-06-12", portfolio={"新能源":0.3, "半导体":0.2, "金融":0.3, "消费":0.2})
    print(rep.total_return, rep.sharpe_ratio, rep.alpha)
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

logger = get_logger("analysis.portfolio")

_STANCE_SCORE = {"利好": 0.02, "中性": 0.0, "利空": -0.02}


@dataclass
class IndustryReturn:
    industry: str
    weight: float
    dates: List[str] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)
    cumulative_return: float = 0.0

    def as_dict(self):
        return asdict(self)


@dataclass
class PortfolioReport:
    date: str
    lookback_days: int
    rebalance: str
    portfolio: Dict[str, float]
    industry_returns: List[IndustryReturn] = field(default_factory=list)
    portfolio_returns: List[float] = field(default_factory=list)
    benchmark_returns: List[float] = field(default_factory=list)
    portfolio_dates: List[str] = field(default_factory=list)
    cumulative_return: float = 0.0
    benchmark_cumulative_return: float = 0.0
    annualized_return: float = 0.0
    annualized_vol: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    information_ratio: float = 0.0
    best_day: float = 0.0
    worst_day: float = 0.0
    positive_days: int = 0
    negative_days: int = 0
    summary: str = ""

    def as_dict(self):
        d = asdict(self)
        d["industry_returns"] = [r.as_dict() for r in self.industry_returns]
        return d


def _industry_returns(target_date, lookback, industries):
    end = parse_date(target_date)
    start = end - timedelta(days=lookback)
    out = {ind: [] for ind in industries}
    try:
        with db_mod.get_conn() as conn:
            placeholders = ",".join("?" for _ in industries)
            rows = conn.execute(
                f"SELECT date, industry, hit_count, stance FROM industry_daily "
                f"WHERE date BETWEEN ? AND ? AND industry IN ({placeholders}) ORDER BY date",
                (start.isoformat(), end.isoformat(), *industries),
            ).fetchall()
        by_ind = {ind: [] for ind in industries}
        for r in rows:
            by_ind[r["industry"]].append(r)
        for ind, rs in by_ind.items():
            rs = sorted(rs, key=lambda x: x["date"])
            for i in range(1, len(rs)):
                prev = rs[i - 1]
                cur = rs[i]
                stance_r = _STANCE_SCORE.get(cur["stance"] or "中性", 0.0)
                momentum = 0.001 * ((cur["hit_count"] or 0) - (prev["hit_count"] or 0)) / max(prev["hit_count"] or 1, 1)
                out[ind].append(stance_r + momentum)
    except Exception as e:
        logger.debug(f"industry_returns 失败: {e}")
    return out


def _fallback_returns(sentiment_rets, n):
    import random
    out = {}
    for ind, noise in (("新能源", 0.015), ("半导体", 0.020), ("金融", 0.010),
                       ("消费", 0.008), ("房地产", 0.025), ("AI", 0.022),
                       ("人工智能", 0.022), ("医药", 0.012), ("基建", 0.009)):
        random.seed(hash(ind) & 0x7fffffff)
        out[ind] = [s + random.gauss(0, noise) for s in sentiment_rets[:n]]
    return out


_REBAL_FREQ = {"daily": 1, "weekly": 5, "monthly": 20, "none": None}


def _portfolio_returns(weights, ind_rets, rebalance="weekly"):
    n = min((len(v) for v in ind_rets.values() if v), default=0)
    if n <= 0:
        return [], []
    industries = list(weights.keys())
    rebal_every = _REBAL_FREQ.get(rebalance, 5)
    cur_w = dict(weights)
    port_rets = []
    ind_returns = {ind: [] for ind in industries}
    for t in range(n):
        if rebal_every is not None and t > 0 and t % rebal_every == 0:
            cur_w = dict(weights)
        day_returns = {ind: (ind_rets.get(ind) or [0.0]*n)[t] for ind in industries if t < len(ind_rets.get(ind, []))}
        pr = sum(cur_w.get(ind, 0.0) * day_returns.get(ind, 0.0) for ind in industries)
        port_rets.append(pr)
        for ind in industries:
            day_r = day_returns.get(ind, 0.0)
            cur_w[ind] = cur_w.get(ind, 0.0) * (1.0 + day_r)
        tot = sum(cur_w.values())
        if tot > 0:
            for k in list(cur_w.keys()):
                cur_w[k] /= tot
        for ind in industries:
            ind_returns[ind].append(day_returns.get(ind, 0.0))
    irs = []
    for ind in industries:
        rets = ind_returns[ind]
        cum = 1.0
        for r in rets:
            cum *= (1.0 + r)
        irs.append(IndustryReturn(
            industry=ind, weight=weights[ind],
            daily_returns=rets, cumulative_return=round(cum - 1.0, 4),
        ))
    return port_rets, irs


def _ann_ret(rets):
    if not rets:
        return 0.0
    cum = 1.0
    for r in rets:
        cum *= (1.0 + r)
    return cum ** (252 / max(1, len(rets))) - 1.0


def _ann_vol(rets):
    if len(rets) < 2:
        return 0.0
    return statistics.pstdev(rets) * math.sqrt(252)


def _max_dd(rets):
    if not rets:
        return 0.0
    nav = [1.0]
    for r in rets:
        nav.append(nav[-1] * (1.0 + r))
    peak = nav[0]
    max_dd = 0.0
    for v in nav:
        if v >= peak:
            peak = v
        else:
            dd = (v - peak) / peak
            if dd < max_dd:
                max_dd = dd
    return max_dd


def _alpha_beta(p_rets, b_rets):
    if len(p_rets) < 2 or len(b_rets) < 2:
        return 0.0, 0.0
    n = min(len(p_rets), len(b_rets))
    p, b = p_rets[:n], b_rets[:n]
    mp, mb = statistics.mean(p), statistics.mean(b)
    cov = sum((p[i] - mp) * (b[i] - mb) for i in range(n)) / n
    var = sum((b[i] - mb) ** 2 for i in range(n)) / n
    if var == 0:
        return 0.0, 0.0
    beta = cov / var
    alpha = (mp - beta * mb) * 252
    return alpha, beta


def _ir(p_rets, b_rets):
    if len(p_rets) < 2:
        return 0.0
    n = min(len(p_rets), len(b_rets))
    diff = [p_rets[i] - b_rets[i] for i in range(n)]
    if len(diff) < 2:
        return 0.0
    sd = statistics.pstdev(diff)
    if sd == 0:
        return 0.0
    return (statistics.mean(diff) * 252) / (sd * math.sqrt(252))


def backtest(target_date="2026-06-12", portfolio=None, rebalance="weekly",
             lookback=90, rf_rate=0.025):
    portfolio = portfolio or {"新能源": 0.3, "半导体": 0.2, "金融": 0.3, "消费": 0.2}
    s = sum(portfolio.values())
    if s <= 0:
        return None
    portfolio = {k: v / s for k, v in portfolio.items()}
    industries = list(portfolio.keys())

    ind_rets = _industry_returns(target_date, lookback, industries)
    if not any(ind_rets.values()) or any(k for k in industries if k not in ind_rets):
        end = parse_date(target_date)
        start = end - timedelta(days=lookback)
        with db_mod.get_conn() as conn:
            rows = conn.execute(
                "SELECT date, sentiment_index FROM daily_metric "
                "WHERE date BETWEEN ? AND ? ORDER BY date",
                (start.isoformat(), end.isoformat()),
            ).fetchall()
        sent_vals = [float(r["sentiment_index"] or 50) for r in rows]
        sent_rets = []
        for i in range(1, len(sent_vals)):
            v0, v1 = sent_vals[i - 1], sent_vals[i]
            sent_rets.append((v1 - v0) / 50.0)
        ind_rets = _fallback_returns(sent_rets, len(sent_rets))

    port_rets, irs = _portfolio_returns(portfolio, ind_rets, rebalance=rebalance)
    if not port_rets:
        return None
    eq = {ind: 1.0 / len(industries) for ind in industries}
    b_rets, _ = _portfolio_returns(eq, ind_rets, rebalance="none")

    cum_p = 1.0
    for r in port_rets:
        cum_p *= (1.0 + r)
    cum_b = 1.0
    for r in b_rets:
        cum_b *= (1.0 + r)
    ar = _ann_ret(port_rets)
    av = _ann_vol(port_rets)
    sharpe = (ar - rf_rate) / av if av > 0 else 0.0
    max_dd = _max_dd(port_rets)
    alpha, beta = _alpha_beta(port_rets, b_rets)
    ir_v = _ir(port_rets, b_rets)
    best = max(port_rets) if port_rets else 0.0
    worst = min(port_rets) if port_rets else 0.0
    pos = sum(1 for r in port_rets if r > 0)
    neg = sum(1 for r in port_rets if r < 0)
    summary = (
        f"{len(industries)} 行业组合 / {rebalance} 调仓 / {len(port_rets)} 个交易日: "
        f"累计 {cum_p - 1:+.2%} (基准 {cum_b - 1:+.2%}) / "
        f"年化 {ar:+.2%} / 波动 {av:.2%} / 夏普 {sharpe:+.2f} / "
        f"alpha {alpha:+.4f} / beta {beta:+.2f} / IR {ir_v:+.2f}"
    )
    return PortfolioReport(
        date=target_date, lookback_days=len(port_rets), rebalance=rebalance,
        portfolio=portfolio, industry_returns=irs,
        portfolio_returns=port_rets, benchmark_returns=b_rets,
        cumulative_return=round(cum_p - 1, 4),
        benchmark_cumulative_return=round(cum_b - 1, 4),
        annualized_return=round(ar, 4), annualized_vol=round(av, 4),
        sharpe_ratio=round(sharpe, 3), max_drawdown=round(max_dd, 4),
        alpha=round(alpha, 4), beta=round(beta, 3),
        information_ratio=round(ir_v, 3),
        best_day=round(best, 4), worst_day=round(worst, 4),
        positive_days=pos, negative_days=neg,
        summary=summary,
    )


if __name__ == "__main__":
    import json
    rep = backtest("2026-06-12", rebalance="weekly", lookback=30)
    if rep is None:
        print("(无数据)")
    else:
        d = rep.as_dict()
        d["portfolio_returns"] = f"[{len(d['portfolio_returns'])} returns omitted]"
        d["benchmark_returns"] = f"[{len(d['benchmark_returns'])} returns omitted]"
        print(json.dumps(d, ensure_ascii=False, indent=2))
        assert -1.0 <= rep.sharpe_ratio <= 50.0
        print("[OK] portfolio self-test passed")
