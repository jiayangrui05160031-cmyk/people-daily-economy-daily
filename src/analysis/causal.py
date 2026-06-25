"""analysis.causal - 政策-市场传导链 (简单因果/关联分析)

思路:
  对当日每个 AI 抽取的政策事件:
    1) 查历史同类政策事件 (event_study._classify 同款分类)
    2) 看同期受影响的产业 (从历史 ai_report.industries.industries 找同关键词)
    3) 算同期 (T+0/+3/+7) 的 industry_breadth / sentiment_index 平均变化
    4) 若变化方向一致 + 强度 > 0.1, 算作一条潜在传导链

注: 这是规则驱动的轻量因果分析, 不等同于 Granger/DoWhy 等严格因果方法,
但作为报告补充足够给出"政策 -> 行业"的关联链路线索。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

from src.ai.schema import NewsEvent
from src.analysis.event_study import _classify as _event_classify
from src.config import INDUSTRY_KEYWORDS
from src.utils.date_utils import parse_date
from src.utils.logger import get_logger

logger = get_logger("analysis.causal")


@dataclass
class CausalChain:
    policy: str  # 政策/事件描述
    category: str
    mediator: str  # 中介: 关联到的产业
    market_response: str  # 关联到的市场反应描述
    strength: float  # 0~1
    lag_days: int  # 关联滞后期
    sample_size: int  # 历史样本数

    def as_dict(self):
        return asdict(self)


@dataclass
class CausalReport:
    date: str
    chains: List[CausalChain] = field(default_factory=list)
    confidence: float = 0.0
    summary: str = ""

    def as_dict(self):
        d = asdict(self)
        d["chains"] = [c.as_dict() for c in self.chains]
        return d


def _match_industry(text: str) -> Optional[str]:
    """在文本中找匹配的产业链。"""
    for ind, keys in INDUSTRY_KEYWORDS.items():
        for k in keys:
            if k in text:
                return ind
    return None


def _classify_event_for_causal(event: NewsEvent) -> str:
    txt = (event.subject or "") + (event.action or "") + (event.object or "")
    return _event_classify(txt)


def _scan_history(category: str, mediator: str, target_date: str,
                   lookback_days: int = 90) -> Tuple[int, float, int]:
    """扫描历史 ai_report: 同类政策 + 同产业, 计算行业 sentiment 变化均值。

    Returns: (hit_count, avg_breadth_change, lag_with_max_effect)
    """
    from src.storage.db import get_conn
    today = parse_date(target_date)
    start = (today - timedelta(days=lookback_days)).isoformat()
    end = today.isoformat()
    conn = get_conn()
    rows = conn.execute(
        "SELECT date, payload_json FROM ai_report WHERE date >= ? AND date <= ? ORDER BY date",
        (start, end),
    ).fetchall()
    if not rows:
        return 0, 0.0, 0
    # 找包含同类政策 + 同一产业关键词的事件日
    matched_dates = []
    for r in rows:
        try:
            p = json.loads(r["payload_json"] or "{}")
        except Exception:
            continue
        ind_names = [i.get("name", "") for i in (p.get("industries") or {}).get("industries", [])]
        if mediator not in ind_names and not any(mediator in n for n in ind_names):
            continue
        evs = (p.get("events") or {}).get("events", [])
        for e in evs:
            txt = (e.get("subject", "") + e.get("action", "") + e.get("object", ""))
            if _event_classify(txt) == category:
                matched_dates.append(r["date"])
                break
    if not matched_dates:
        return 0, 0.0, 0
    # 计算每个匹配日 + N 天后 industry_breadth / sentiment 的变化
    diffs: List[Tuple[int, float, float]] = []  # (lag, breadth_diff, sent_diff)
    for d in matched_dates:
        d0 = parse_date(d)
        m0 = conn.execute(
    "SELECT sentiment_index, industry_count FROM daily_metric WHERE date = ?",
            (d,),
        ).fetchone()
        if not m0:
            continue
        sent0 = float(m0["sentiment_index"] or 50)
        breadth0 = float(m0["industry_count"] or 0)
        for lag in (1, 3, 7):
            dN = (d0 + timedelta(days=lag)).isoformat()
            if dN > target_date:
                break
            mN = conn.execute(
                "SELECT sentiment_index, industry_count FROM daily_metric WHERE date = ?",
                (dN,),
            ).fetchone()
            if mN:
                sent_diff = float(mN["sentiment_index"] or 0) - sent0
                breadth_diff = float(mN["industry_count"] or 0) - breadth0
                diffs.append((lag, breadth_diff, sent_diff))
    if not diffs:
        return len(matched_dates), 0.0, 0
    # 选 lag 维度上绝对变化最大的
    by_lag: Dict[int, List[Tuple[float, float]]] = {1: [], 3: [], 7: []}
    for lag, b, s in diffs:
        by_lag[lag].append((b, s))
    best_lag = 0
    best_score = 0.0
    for lag, samples in by_lag.items():
        if not samples:
            continue
        avg_b = sum(b for b, _ in samples) / len(samples)
        avg_s = sum(s for _, s in samples) / len(samples)
        score = abs(avg_s) / 50.0  # sentiment 0-100, 归一化
        if score > best_score:
            best_score = score
            best_lag = lag
            best_avg_b = avg_b
    return len(matched_dates), best_avg_b if matched_dates else 0.0, best_lag


def analyze(events: List[NewsEvent], target_date: str = "2026-06-12",
            lookback_days: int = 90) -> CausalReport:
    """主入口: 对当日事件做政策-市场传导分析。"""
    chains: List[CausalChain] = []
    for ev in events[:8]:  # 最多取前 8 个事件避免太慢
        cat = _classify_event_for_causal(ev)
        if cat == "other":
            continue
        text = (ev.subject or "") + (ev.action or "") + (ev.object or "")
        mediator = _match_industry(text) or _match_industry(ev.impact or "")
        if not mediator:
            # 没命中具体产业, 用 category 推断一个最相关的
            cat_to_ind = {
                "monetary": "金融", "fiscal": "金融",
                "real_estate": "房地产",
                "trade": "外贸", "consumption": "消费",
                "employment": "消费",
            }
            mediator = cat_to_ind.get(cat)
        if not mediator:
            continue
        hit, breadth_diff, lag = _scan_history(cat, mediator, target_date, lookback_days)
        if hit < 1:
            continue
        strength = min(1.0, (hit / 5.0) * 0.7 + min(1.0, abs(breadth_diff) / 3.0) * 0.3)
        if strength < 0.2:
            continue
        if breadth_diff > 0.3:
            response = f"{mediator} 关注度上升 (Δ={breadth_diff:+.2f})"
        elif breadth_diff < -0.3:
            response = f"{mediator} 关注度下降 (Δ={breadth_diff:+.2f})"
        else:
            response = f"{mediator} 关注度基本持平 (Δ={breadth_diff:+.2f})"
        chains.append(CausalChain(
            policy=(ev.subject + " " + ev.action + " " + ev.object)[:50],
            category=cat,
            mediator=mediator,
            market_response=response,
            strength=round(strength, 3),
            lag_days=lag,
            sample_size=hit,
        ))
    chains.sort(key=lambda c: -c.strength)
    confidence = round(min(1.0, len(chains) / 5.0), 3) if chains else 0.0
    summary = (
        f"识别 {len(chains)} 条潜在传导链; "
        f"平均强度 {sum(c.strength for c in chains) / max(1, len(chains)):.2f}; "
        f"历史样本基线 {sum(c.sample_size for c in chains)} 条同类事件。"
    ) if chains else "未识别出明显传导链 (历史数据不足或当日事件无明确产业关联)"
    return CausalReport(date=target_date, chains=chains[:6],
                        confidence=confidence, summary=summary)


if __name__ == "__main__":
    from src.ai.schema import NewsEvent, EventType
    from src.storage import db
    db.get_conn()
    sample = [
        NewsEvent(subject="央行", action="降准", object="释放流动性",
                  event_type=EventType.MONETARY, impact="利好金融地产"),
        NewsEvent(subject="财政部", action="出台补贴", object="新能源车",
                  event_type=EventType.FISCAL, impact="新能源受益"),
        NewsEvent(subject="国务院", action="支持房地产", object="保交楼",
                  event_type=EventType.INDUSTRIAL, impact="房地产链"),
    ]
    rep = analyze(sample, target_date="2026-06-12")
    print(json.dumps(rep.as_dict(), ensure_ascii=False, indent=2))
