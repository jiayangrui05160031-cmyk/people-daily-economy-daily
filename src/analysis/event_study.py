"""analysis.event_study - 事件研究 (新闻事件 -> 政策/产业分类 + 历史同类 + 影响评估)

输入: 当日 AI 抽取的 events 列表 (subject/action/object/event_type/impact)。
处理:
1. 规则分类: 根据关键词 + event_type 划分 政策/产业/货币/财政/国际/民生 6 类。
2. 强度评分: 0~1, 综合 action 动词强度 + 行业敏感度 + 主体权威性。
3. 历史同类: 在时序库的 ai_report.payload_json 中 grep 同类事件出现频次。
4. 预期影响: 关键词驱动的模板 (积极/中性/消极 + 时间窗 1-3 天 / 1-2 周)。

无外部 ML 依赖, 纯规则 + SQL。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

from src.ai.schema import NewsEvent
from src.storage import repository as repo
from src.utils.date_utils import parse_date
from src.utils.logger import get_logger

logger = get_logger("analysis.event_study")

# 分类关键词
_CATEGORY_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    ("monetary", ("降准", "降息", "LPR", "MLF", "逆回购", "货币政策", "流动性", "央行")),
    ("fiscal", ("财政", "赤字", "国债", "专项债", "减税", "退税", "转移支付")),
    ("regulatory", ("监管", "处罚", "罚款", "整顿", "新规", "征求意见")),
    ("industry_support", ("补贴", "支持", "鼓励", "促进", "试点", "示范")),
    ("industry_clamp", ("限制", "禁止", "淘汰", "出清", "去产能")),
    ("international", ("美联储", "美元", "人民币汇率", "G20", "IMF", "WTO", "一带一路", "RCEP")),
    ("trade", ("进出口", "外贸", "关税", "跨境", "贸易")),
    ("real_estate", ("房地产", "楼市", "房企", "保交楼", "保障房", "城中村")),
    ("employment", ("就业", "失业", "稳岗", "招聘")),
    ("consumption", ("消费", "内需", "以旧换新", "消费券", "下沉")),
]

_ACTION_INTENSITY: Dict[str, float] = {
    "宣布": 0.7, "下调": 0.85, "上调": 0.85, "释放": 0.8, "发放": 0.7,
    "出台": 0.75, "发布": 0.5, "实施": 0.7, "推进": 0.4, "启动": 0.7,
    "试点": 0.6, "扩容": 0.65, "扩围": 0.7, "提速": 0.6, "突破": 0.7,
    "增长": 0.5, "下降": 0.6, "回落": 0.5, "反弹": 0.55,
    "签约": 0.4, "达成": 0.5, "批复": 0.6, "审议": 0.4,
    "约谈": 0.75, "处罚": 0.85, "罚款": 0.85, "禁止": 0.9, "限制": 0.7,
}

_SUBJECT_WEIGHT: Dict[str, float] = {
    "央行": 0.95, "中国人民银行": 0.95, "国务院": 0.95, "发改委": 0.85,
    "财政部": 0.9, "工信部": 0.8, "证监会": 0.85, "银保监会": 0.85,
    "国资委": 0.8, "国家发改委": 0.85, "央行": 0.95, "人民银行": 0.95,
}

_CATEGORY_TO_IMPACT: Dict[str, Tuple[str, str]] = {
    "monetary": ("宽松利好股债, 短期 1-3 天", "扩张"),
    "fiscal": ("财政发力, 短期 1-2 周", "扩张"),
    "regulatory": ("板块分化, 需关注细则", "中性"),
    "industry_support": ("受益板块有望提振, 中期 1-4 周", "扩张"),
    "industry_clamp": ("相关板块承压, 短期 1-2 周", "收紧"),
    "international": ("扰动汇率与外资流向, 短期 1-3 天", "中性"),
    "trade": ("外贸板块受益或承压, 中期 1-2 月", "中性"),
    "real_estate": ("地产链上下游联动, 中期 1-3 月", "中性"),
    "employment": ("民生与消费侧, 中长期 1-6 月", "扩张"),
    "consumption": ("消费板块提振, 短期 1-2 周", "扩张"),
    "other": ("需结合具体行业判断", "中性"),
}


def _classify(text: str) -> str:
    for cat, keys in _CATEGORY_RULES:
        for k in keys:
            if k in text:
                return cat
    return "other"


def _intensity(event: NewsEvent) -> float:
    action_score = 0.5
    for k, v in _ACTION_INTENSITY.items():
        if k in (event.action or ""):
            action_score = max(action_score, v)
    subj_score = 0.5
    for k, v in _SUBJECT_WEIGHT.items():
        if k in (event.subject or ""):
            subj_score = max(subj_score, v)
    type_bonus = 0.0
    if str(event.event_type) in ("Monetary", "Policy", "MONETARY"):
        type_bonus = 0.1
    if event.impact and len(event.impact) > 10:
        type_bonus += 0.05
    score = action_score * 0.5 + subj_score * 0.4 + type_bonus
    return round(min(1.0, max(0.1, score)), 3)


def _expected_impact(category: str) -> str:
    return _CATEGORY_TO_IMPACT.get(category, _CATEGORY_TO_IMPACT["other"])[0]


@dataclass
class StudiedEvent:
    subject: str
    action: str
    object: str
    event_type: str
    impact: str
    category: str
    intensity: float
    historical_count: int
    expected_impact: str

    def as_dict(self):
        return asdict(self)


@dataclass
class HistoricalEvent:
    date: str
    category: str
    title: str
    outcome: str

    def as_dict(self):
        return asdict(self)


@dataclass
class EventStudyReport:
    date: str
    events: List[StudiedEvent]
    historical: List[HistoricalEvent] = field(default_factory=list)
    history_size: int = 0
    summary: str = ""

    def as_dict(self):
        d = asdict(self)
        d["events"] = [e.as_dict() for e in self.events]
        d["historical"] = [h.as_dict() for h in self.historical]
        return d


def _grep_history(category: str, lookback_days: int = 90, ref_date: str = "2026-06-12") -> List[HistoricalEvent]:
    """在 ai_report.payload_json 中检索同类历史事件。"""
    from src.storage.db import get_conn
    today = parse_date(ref_date)
    start = (today - timedelta(days=lookback_days)).isoformat()
    end = today.isoformat()
    conn = get_conn()
    rows = conn.execute(
        "SELECT date, payload_json FROM ai_report WHERE date >= ? AND date <= ? ORDER BY date",
        (start, end),
    ).fetchall()
    out: List[HistoricalEvent] = []
    for r in rows:
        try:
            payload = json.loads(r["payload_json"] or "{}")
        except Exception:
            continue
        evs = (payload.get("events") or {}).get("events", [])
        for e in evs:
            txt = (e.get("subject", "") + e.get("action", "") + e.get("object", ""))
            c = _classify(txt)
            if c == category:
                out.append(HistoricalEvent(
                    date=r["date"],
                    category=category,
                    title=(e.get("subject", "") + " " + e.get("action", "") + " " + e.get("object", ""))[:60],
                    outcome=(e.get("impact", "") or "—")[:60],
                ))
                if len(out) >= 6:
                    return out
    return out


def study(events: List[NewsEvent], target_date: str = "2026-06-12",
          lookback_days: int = 90) -> EventStudyReport:
    """主入口: 对当日 events 做事件研究。"""
    studied: List[StudiedEvent] = []
    seen_cats: set = set()
    for ev in events:
        text = (ev.subject or "") + (ev.action or "") + (ev.object or "")
        cat = _classify(text)
        seen_cats.add(cat)
        studied.append(StudiedEvent(
            subject=ev.subject or "—",
            action=ev.action or "—",
            object=ev.object or "—",
            event_type=str(ev.event_type or "—"),
            impact=ev.impact or "—",
            category=cat,
            intensity=_intensity(ev),
            historical_count=0,  # 后补
            expected_impact=_expected_impact(cat),
        ))
    # 统计每类历史数 (扫一遍 ai_report 库)
    cat_counts: Dict[str, int] = {c: 0 for c in seen_cats}
    hist_summary: List[HistoricalEvent] = []
    for c in seen_cats:
        h = _grep_history(c, lookback_days=lookback_days)
        cat_counts[c] = len(h)
        if c in ("monetary", "fiscal", "industry_support", "industry_clamp"):
            hist_summary.extend(h[:2])
    for s in studied:
        s.historical_count = cat_counts.get(s.category, 0)
    studied.sort(key=lambda e: -e.intensity)
    summary = (
        f"共 {len(studied)} 个事件, 涉及 {len(seen_cats)} 个类别; "
        f"高强度 (>=0.7) {sum(1 for e in studied if e.intensity >= 0.7)} 个, "
        f"历史同类事件总计 {sum(cat_counts.values())} 条。"
    )
    return EventStudyReport(
        date=target_date, events=studied,
        historical=hist_summary[:6],
        history_size=sum(cat_counts.values()),
        summary=summary,
    )


if __name__ == "__main__":
    from src.ai.schema import NewsEvent, EventType
    sample = [
        NewsEvent(subject="中国人民银行", action="宣布下调存款准备金率",
                  object="释放长期资金一亿元", event_type=EventType.MONETARY,
                  impact="利好银行地产板块"),
        NewsEvent(subject="财政部", action="出台新能源汽车补贴",
                  object="延长至 2027 年", event_type=EventType.FISCAL,
                  impact="新能源车板块受益"),
        NewsEvent(subject="美联储", action="宣布加息 25 个基点",
                  object="联邦基金利率", event_type=EventType.MONETARY,
                  impact="人民币汇率承压"),
    ]
    rep = study(sample)
    print(json.dumps(rep.as_dict(), ensure_ascii=False, indent=2))