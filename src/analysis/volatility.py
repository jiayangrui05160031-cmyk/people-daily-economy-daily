"""analysis.volatility - 波动率 / 恐慌指数 (类 VIX)"

从历史 N 天的 sentiment_index / policy_stance_score / industry_count / event_count
四个维度的滚动统计合成,输出 0~100 的恐慌指数。

- 情绪波动 (sentiment_vol)  - sentiment_index 滚动 std
- 政策波动 (policy_vol)    - |policy_stance_score| 均值
- 行业活跃 (industry_vol)  - industry_count 一阶差分
- 事件密度 (event_density) - event_count / max(industry_count, 1)

指数合成: 0.35*sent_vol + 0.25*policy_vol + 0.20*industry_vol + 0.20*event_density
归一化到 0~100。

无外部依赖,纯 SQL + 统计。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import timedelta
from typing import Dict, List

from src.storage import repository as repo
from src.utils.date_utils import parse_date
from src.utils.logger import get_logger

logger = get_logger("analysis.volatility")


@dataclass
class VolatilityComponents:
    sentiment_vol: float = 0.0
    policy_vol: float = 0.0
    industry_vol: float = 0.0
    event_density: float = 0.0

    def as_dict(self):
        return asdict(self)


@dataclass
class VolatilityReport:
    date: str
    index: float
    level: str
    ma7: float
    components: VolatilityComponents
    history: List[Dict[str, float]]
    summary: str

    def as_dict(self):
        d = asdict(self)
        d["components"] = self.components.as_dict()
        return d


def _normalize(value, lo, hi):
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _classify(idx):
    if idx >= 80: return "extreme_fear"
    if idx >= 60: return "fear"
    if idx >= 40: return "neutral"
    if idx >= 20: return "calm"
    return "extreme_calm"


def compute(date_str, window_days=14):
    """计算当日类 VIX 恐慌指数。"""
    end = (parse_date(date_str) - timedelta(days=1)).isoformat()
    start = (parse_date(date_str) - timedelta(days=window_days)).isoformat()
    history = sorted(repo.list_metrics(start_date=start, end_date=end), key=lambda m: m.date)
    if len(history) < 3:
        return VolatilityReport(
            date=date_str, index=0.0, level="unknown", ma7=0.0,
            components=VolatilityComponents(), history=[],
            summary="(历史样本仅 " + str(len(history)) + " 天, 不足 3 天, 跳过波动率计算)",
        )
    sents = [float(m.sentiment_index or 0) for m in history]
    policies = [abs(float(m.policy_stance_score or 0)) for m in history]
    industry_counts = [int(m.industry_count or 0) for m in history]
    event_counts = [int(m.event_count or 0) for m in history]
    mean_s = sum(sents) / len(sents)
    sent_vol = (sum((s - mean_s) ** 2 for s in sents) / len(sents)) ** 0.5
    sent_vol_n = _normalize(sent_vol, 0, 30)
    policy_mean = sum(policies) / len(policies)
    policy_vol_n = _normalize(policy_mean, 0, 0.7)
    if len(industry_counts) >= 2:
        diffs = [abs(industry_counts[i] - industry_counts[i-1]) for i in range(1, len(industry_counts))]
        industry_vol = sum(diffs) / len(diffs)
    else:
        industry_vol = 0.0
    industry_vol_n = _normalize(industry_vol, 0, 3)
    densities = []
    for i in range(len(event_counts)):
        ind = max(int(history[i].industry_count or 1), 1)
        densities.append(event_counts[i] / ind)
    event_density = sum(densities) / len(densities) if densities else 0.0
    event_density_n = _normalize(event_density, 0, 5)
    raw = 0.35 * sent_vol_n + 0.25 * policy_vol_n + 0.20 * industry_vol_n + 0.20 * event_density_n
    index = round(raw * 100, 1)
    level = _classify(index)
    comp = VolatilityComponents(
        sentiment_vol=round(sent_vol, 3),
        policy_vol=round(policy_mean, 3),
        industry_vol=round(industry_vol, 3),
        event_density=round(event_density, 3),
    )
    ma7 = round(sum(sents[-7:]) / 7, 1) if len(sents) >= 7 else round(mean_s, 1)
    series = []
    for i, m in enumerate(history):
        if i < 3:
            series.append({"date": m.date, "index": 0.0})
            continue
        win = sents[max(0, i-3):i+1]
        m_s = sum(win) / len(win)
        local_std = (sum((v - m_s) ** 2 for v in win) / len(win)) ** 0.5
        local_idx = round(_normalize(local_std, 0, 30) * 100, 1)
        series.append({"date": m.date, "index": local_idx})
    summary = (
        "情绪 std=" + str(round(sent_vol, 2)) +
        ", 政策强度=" + str(round(policy_mean, 2)) +
        ", 行业变化=" + str(round(industry_vol, 3)) +
        ", 事件密度=" + str(round(event_density, 2)) +
        "; 综合指数 " + str(index) + " (" + level + ")"
    )
    return VolatilityReport(
        date=date_str, index=index, level=level, ma7=ma7,
        components=comp, history=series, summary=summary,
    )


def seed_demo_history(target_date, days=30, seed=42):
    """注入合成历史指标, 用于演示/测试波动率/异常/预测模块。

    模拟一个温和上行 + 偶发突发的市场, 不覆盖已有数据。
    """
    import random
    random.seed(seed)
    today = parse_date(target_date)
    written = 0
    for i in range(days, 0, -1):
        d = (today - timedelta(days=i)).isoformat()
        if repo.get_metric(d) is not None:
            continue
        # 温和上行趋势 + 偶发跳水
        base = 55 + (days - i) * 0.4 + random.uniform(-3, 3)
        if i in (5, 12, 20):
            base += random.choice([-15, 12])  # 异常
        pol = max(-1.0, min(1.0, 0.1 + random.uniform(-0.3, 0.4)))
        ent = max(0.3, min(1.0, 0.85 - i * 0.005 + random.uniform(-0.05, 0.05)))
        top = max(0.1, min(0.9, 0.4 + random.uniform(-0.1, 0.15)))
        ind = random.randint(3, 8)
        ev = random.randint(2, 10) if i < 5 else random.randint(0, 6)
        pol_n = random.randint(0, 4)
        art = random.randint(20, 60)
        m = repo.DailyMetric(
            date=d, article_count=art, total_words=art * 350,
            unique_keywords=random.randint(40, 120),
            policy_stance_score=round(pol, 3),
            sentiment_index=round(base, 2),
            attention_entropy=round(ent, 3),
            attention_top_share=round(top, 3),
            industry_count=ind,
            policy_count=pol_n, event_count=ev,
        )
        repo.upsert_metric(m)
        written += 1
    return written


if __name__ == "__main__":
    import json
    from src.storage import db
    db.get_conn()
    today = "2026-06-12"
    n = seed_demo_history(today, days=30)
    print("seeded", n, "rows")
    rep = compute(today, window_days=14)
    print(json.dumps(rep.as_dict(), ensure_ascii=False, indent=2))