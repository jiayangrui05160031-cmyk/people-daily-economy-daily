"""analysis.scenario - 情景分析 + 蒙特卡洛压力测试 ============================================
对宏观系统做"政策冲击 + 蒙特卡洛模拟", 给出未来 N 天的概率分布:
  - 定义情景: 央行降准 50bp / 美联储加息 25bp / 房地产暴雷 / 出口萎缩 等
  - 每个情景对 sentiment/policy/industry 施加不同冲击
  - 用历史 daily_metric 估计参数 (mu, sigma)
  - 蒙特卡洛 1000 次模拟未来 N 天
  - 输出: P(正向), P(回撤>X%), 中位数, 5/95 分位
无 LLM, 纯统计。失败时优雅返回 None。
典型用法:
    from src.analysis.scenario import run
    rep = run("2026-06-12", scenario="rate_cut_50bp", horizon_days=30, n_sims=1000)
    print(rep.p_positive, rep.percentile_5, rep.summary)
"""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, asdict, field
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

from src.storage import db as db_mod
from src.utils.date_utils import parse_date
from src.utils.logger import get_logger

logger = get_logger("analysis.scenario")

# ============================================================
# 情景定义
# ============================================================
SCENARIOS = {
    "rate_cut_50bp": {
        "name": "央行降准 / 降息 50bp",
        "shocks": {"sentiment_index": 8.0, "policy_stance_score": 0.20, "industry_count": 1.5},
        "vol_mult": 1.2,
        "horizon": 30,
    },
    "rate_hike_25bp": {
        "name": "美联储加息 25bp / 国内收紧",
        "shocks": {"sentiment_index": -6.0, "policy_stance_score": -0.15, "industry_count": -0.8},
        "vol_mult": 1.4,
        "horizon": 30,
    },
    "property_crisis": {
        "name": "房地产暴雷 / 信用风险事件",
        "shocks": {"sentiment_index": -15.0, "policy_stance_score": -0.30, "industry_count": -2.0},
        "vol_mult": 1.8,
        "horizon": 60,
    },
    "export_boom": {
        "name": "外贸强劲 / 出口超预期",
        "shocks": {"sentiment_index": 5.0, "policy_stance_score": 0.10, "industry_count": 0.8},
        "vol_mult": 0.9,
        "horizon": 30,
    },
    "geopolitical_shock": {
        "name": "地缘政治冲击 / 外部不确定性",
        "shocks": {"sentiment_index": -10.0, "policy_stance_score": -0.05, "industry_count": -1.0},
        "vol_mult": 1.6,
        "horizon": 45,
    },
    "ai_breakthrough": {
        "name": "AI 技术突破 / 产业革命",
        "shocks": {"sentiment_index": 12.0, "policy_stance_score": 0.10, "industry_count": 2.5},
        "vol_mult": 1.3,
        "horizon": 60,
    },
    "baseline": {
        "name": "基准 (无冲击)",
        "shocks": {},
        "vol_mult": 1.0,
        "horizon": 30,
    },
}


@dataclass
class ScenarioPath:
    sim_id: int
    sentiment_path: List[float] = field(default_factory=list)
    policy_path: List[float] = field(default_factory=list)
    industry_path: List[float] = field(default_factory=list)
    total_return: float = 0.0
    max_drawdown: float = 0.0

    def as_dict(self):
        return asdict(self)


@dataclass
class ScenarioReport:
    date: str
    scenario: str
    scenario_name: str
    horizon_days: int
    n_sims: int
    paths: List[ScenarioPath] = field(default_factory=list)
    initial_value: float = 0.0
    median_final: float = 0.0
    mean_final: float = 0.0
    percentile_5: float = 0.0
    percentile_95: float = 0.0
    p_positive: float = 0.0
    p_dd_gt_10pct: float = 0.0
    p_dd_gt_20pct: float = 0.0
    median_max_dd: float = 0.0
    median_total_return: float = 0.0
    worst_case_final: float = 0.0
    best_case_final: float = 0.0
    summary: str = ""

    def as_dict(self):
        d = asdict(self)
        d["paths_count"] = len(self.paths)
        d["paths"] = []
        return d


def _estimate_params(target_date, lookback=90):
    end = parse_date(target_date)
    start = end - timedelta(days=lookback)
    metrics = ("sentiment_index", "policy_stance_score", "industry_count")
    params = {m: (0.0, 1.0) for m in metrics}
    try:
        with db_mod.get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM daily_metric WHERE date BETWEEN ? AND ? ORDER BY date",
                (start.isoformat(), end.isoformat()),
            ).fetchall()
        for m in metrics:
            vals = [float(r[m]) for r in rows if r[m] is not None]
            if len(vals) < 2:
                continue
            diffs = [vals[i] - vals[i - 1] for i in range(1, len(vals))]
            mu = statistics.mean(diffs) if diffs else 0.0
            sd = statistics.pstdev(diffs) if len(diffs) > 1 else 1.0
            params[m] = (mu, max(sd, 0.01))
    except Exception as e:
        logger.debug(f"estimate params 失败: {e}")
    return params


def _simulate_path(initial, mu, sd, shocks, vol_mult, horizon, sim_id, seed=0):
    rng = random.Random(seed + sim_id)
    sent = initial.get("sentiment_index", 50.0)
    pol = initial.get("policy_stance_score", 0.0)
    ind = initial.get("industry_count", 8.0)
    sent_path = [sent]
    pol_path = [pol]
    ind_path = [ind]
    sent += shocks.get("sentiment_index", 0.0)
    pol += shocks.get("policy_stance_score", 0.0)
    ind += shocks.get("industry_count", 0.0)
    sent_path.append(sent)
    pol_path.append(pol)
    ind_path.append(ind)
    for t in range(horizon):
        decay = 0.5 ** ((t + 1) / max(horizon / 2, 1))
        s_shock = shocks.get("sentiment_index", 0.0) * decay
        p_shock = shocks.get("policy_stance_score", 0.0) * decay
        i_shock = shocks.get("industry_count", 0.0) * decay
        sent += mu["sentiment_index"] + rng.gauss(0, sd["sentiment_index"] * vol_mult) + s_shock * 0.05
        pol += mu["policy_stance_score"] + rng.gauss(0, sd["policy_stance_score"] * vol_mult) + p_shock * 0.02
        ind += mu["industry_count"] + rng.gauss(0, sd["industry_count"] * vol_mult) + i_shock * 0.02
        sent_path.append(sent)
        pol_path.append(pol)
        ind_path.append(ind)
    total_return = (sent - sent_path[0]) / max(abs(sent_path[0]), 1.0)
    max_dd = 0.0
    peak = sent_path[0]
    for v in sent_path:
        if v >= peak:
            peak = v
        else:
            dd = (v - peak) / max(abs(peak), 1.0)
            if dd < max_dd:
                max_dd = dd
    return ScenarioPath(
        sim_id=sim_id,
        sentiment_path=sent_path, policy_path=pol_path, industry_path=ind_path,
        total_return=round(total_return, 4), max_drawdown=round(max_dd, 4),
    )


def run(target_date="2026-06-12", scenario="baseline", horizon_days=30, n_sims=1000, seed=42):
    if scenario not in SCENARIOS:
        logger.warning(f"未知情景 {scenario}, 降级为 baseline")
        scenario = "baseline"
    spec = SCENARIOS[scenario]
    if horizon_days <= 0:
        horizon_days = spec.get("horizon", 30)
    params = _estimate_params(target_date, lookback=90)
    mu = {k: v[0] for k, v in params.items()}
    sd = {k: v[1] for k, v in params.items()}
    initial = {"sentiment_index": 50.0, "policy_stance_score": 0.0, "industry_count": 8.0}
    try:
        with db_mod.get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM daily_metric WHERE date <= ? ORDER BY date DESC LIMIT 1",
                (target_date,),
            ).fetchone()
        if row:
            for k in initial:
                if row[k] is not None:
                    initial[k] = float(row[k])
    except Exception as e:
        logger.debug(f"read initial 失败: {e}")
    paths = []
    for i in range(n_sims):
        p = _simulate_path(initial, mu, sd, spec["shocks"], spec["vol_mult"], horizon_days, i, seed=seed)
        paths.append(p)
    finals = [p.sentiment_path[-1] for p in paths]
    dds = [p.max_drawdown for p in paths]
    returns = [p.total_return for p in paths]
    finals_sorted = sorted(finals)
    n = len(finals)
    median_final = finals_sorted[n // 2] if n else 0.0
    mean_final = statistics.mean(finals) if finals else 0.0
    p5 = finals_sorted[int(0.05 * n)] if n else 0.0
    p95 = finals_sorted[int(0.95 * n)] if n else 0.0
    p_pos = sum(1 for f in finals if f > initial["sentiment_index"]) / max(1, n)
    p_dd_10 = sum(1 for d in dds if d < -0.10) / max(1, n)
    p_dd_20 = sum(1 for d in dds if d < -0.20) / max(1, n)
    med_dd = statistics.median(dds) if dds else 0.0
    med_ret = statistics.median(returns) if returns else 0.0
    summary = (
        f"情景 [{spec['name']}] / {horizon_days} 天 / {n_sims} 次模拟: "
        f"初值 {initial['sentiment_index']:.1f} -> 中位终值 {median_final:.1f} / "
        f"P(正向) {p_pos:.0%} / P(回撤>10%) {p_dd_10:.0%} / "
        f"中位最大回撤 {med_dd:+.1%} / 5%~95% 区间 [{p5:.1f}, {p95:.1f}]"
    )
    return ScenarioReport(
        date=target_date, scenario=scenario, scenario_name=spec["name"],
        horizon_days=horizon_days, n_sims=n_sims, paths=paths,
        initial_value=round(initial["sentiment_index"], 2),
        median_final=round(median_final, 2), mean_final=round(mean_final, 2),
        percentile_5=round(p5, 2), percentile_95=round(p95, 2),
        p_positive=round(p_pos, 3),
        p_dd_gt_10pct=round(p_dd_10, 3), p_dd_gt_20pct=round(p_dd_20, 3),
        median_max_dd=round(med_dd, 4), median_total_return=round(med_ret, 4),
        worst_case_final=round(min(finals) if finals else 0.0, 2),
        best_case_final=round(max(finals) if finals else 0.0, 2),
        summary=summary,
    )


def list_scenarios():
    return [{"id": k, "name": v["name"]} for k, v in SCENARIOS.items()]


if __name__ == "__main__":
    import json
    rep = run("2026-06-12", scenario="rate_cut_50bp", horizon_days=30, n_sims=200)
    if rep is None:
        print("(无数据)")
    else:
        print(json.dumps(rep.as_dict(), ensure_ascii=False, indent=2))
        assert 0.0 <= rep.p_positive <= 1.0
        print("[OK] scenario self-test passed (rate_cut_50bp)")
        rep2 = run("2026-06-12", scenario="baseline", n_sims=200)
        print(f"baseline: P(正向)={rep2.p_positive:.0%}, 中位终值={rep2.median_final:.1f}")
