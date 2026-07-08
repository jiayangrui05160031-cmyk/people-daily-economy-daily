"""analysis.metrics - 量化指标计算

输出数值,供报告 / 仪表盘 / DB 使用:
- attention_entropy (0~1): 词频分布香农熵归一化
- attention_top_share: 头部词集中度 (top10 / 总频次)
- sentiment_index (0~100): 多空情绪聚合指数
- policy_stance_score (-1~+1): 政策倾向得分
- event_density: 事件 / 文章密度
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple

from src.ai.schema import AnalysisReport
from src.config import (
    ATTENTION_ENTROPY_HIGH,
    POLICY_STANCE_SCORE,
    SENTIMENT_STANCE_SCORE,
)
from src.utils.logger import get_logger

logger = get_logger("analysis.metrics")


@dataclass
class QuantitativeMetrics:
    attention_entropy: float = 0.0
    attention_top_share: float = 0.0
    sentiment_index: float = 50.0
    sentiment_dispersion: float = 0.0
    policy_stance_score: float = 0.0
    policy_confidence: float = 0.0
    industry_breadth: float = 0.0
    event_density: float = 0.0
    intensity_label: str = "中性"
    sentiment_label: str = "中性观望"

    def as_dict(self) -> Dict:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, float):
                d[k] = round(v, 4)
        return d


def attention_entropy(word_freq):
    freqs = [f for _, f in word_freq if f > 0]
    if not freqs or len(freqs) == 1:
        return 0.0
    total = sum(freqs)
    if total == 0:
        return 0.0
    probs = [f / total for f in freqs]
    H = -sum(p * math.log(p) for p in probs if p > 0)
    H_max = math.log(len(probs))
    return round(H / H_max, 4) if H_max > 0 else 0.0


def attention_top_share(word_freq, top_k=10):
    freqs = [f for _, f in word_freq if f > 0]
    if not freqs:
        return 0.0
    total = sum(freqs)
    head = sum(sorted(freqs, reverse=True)[:top_k])
    return round(head / total, 4) if total else 0.0


def sentiment_index(sentiment_items):
    if not sentiment_items:
        return 50.0, 0.0
    raw = []
    for it in sentiment_items:
        s = SENTIMENT_STANCE_SCORE.get(str(it.stance), 0.0)
        raw.append(s * float(it.intensity))
    if not raw:
        return 50.0, 0.0
    mean = sum(raw) / len(raw)
    var = sum((x - mean) ** 2 for x in raw) / len(raw)
    std = math.sqrt(var)
    idx = (mean + 1) * 50.0
    return round(idx, 2), round(std, 4)


def _sentiment_label(idx):
    if idx >= 75: return "极度看多"
    if idx >= 60: return "偏多"
    if idx >= 45: return "中性偏多"
    if idx >= 40: return "中性观望"
    if idx >= 25: return "中性偏空"
    if idx >= 10: return "偏空"
    return "极度看空"


def policy_stance(policy_dir):
    if not policy_dir:
        return 0.0, 0.0
    direction = str(policy_dir.direction) if hasattr(policy_dir, "direction") else str(policy_dir)
    score = POLICY_STANCE_SCORE.get(direction, 0.0)
    conf = float(policy_dir.confidence) if hasattr(policy_dir, "confidence") else 0.5
    return score, round(conf, 4)


def industry_breadth(hit_industries, total):
    if total <= 0:
        return 0.0
    return round(len(hit_industries) / total, 4)


def event_density(event_count, article_count):
    if article_count <= 0:
        return 0.0
    return round(event_count / article_count, 4)


def _intensity_label(entropy, top_share):
    if top_share >= 0.7: return "高度聚焦"
    if top_share >= 0.5: return "中等聚焦"
    if entropy >= ATTENTION_ENTROPY_HIGH: return "高度分散"
    return "常规分布"


def compute(word_freq, ai_report, hit_industries, total_industries, article_count=None):
    ent = attention_entropy(word_freq)
    top_share = attention_top_share(word_freq)
    s_idx, s_disp = sentiment_index(getattr(ai_report.sentiment, "items", []) if ai_report else [])
    p_score, p_conf = policy_stance(getattr(ai_report, "policy_direction", None) if ai_report else None)
    breadth = industry_breadth(hit_industries, total_industries)
    events = len(getattr(ai_report.events, "events", []) if ai_report else [])
    if article_count is None:
        arts = max(
            len(getattr(ai_report.industries, "industries", []) if ai_report else []),
            1,
        )
    else:
        arts = int(article_count)
    e_density = event_density(events, max(arts, 1))

    return QuantitativeMetrics(
        attention_entropy=ent,
        attention_top_share=top_share,
        sentiment_index=s_idx,
        sentiment_dispersion=s_disp,
        policy_stance_score=p_score,
        policy_confidence=p_conf,
        industry_breadth=breadth,
        event_density=e_density,
        intensity_label=_intensity_label(ent, top_share),
        sentiment_label=_sentiment_label(s_idx),
    )


if __name__ == "__main__":
    freq = [("降准", 8), ("新能源", 6), ("半导体", 5), ("房地产", 3), ("消费", 2), ("外贸", 1)]
    print("entropy:", attention_entropy(freq))
    print("top_share:", attention_top_share(freq))
    print("sentiment idx (empty):", sentiment_index([]))
    from src.ai.schema import (
        AnalysisReport, ThemeKeywordsResult, ThemeKeyword,
        PolicyDirectionResult, IndustriesResult, IndustryFocus,
        PoliciesResult, CoreInsightsResult, OutlooksResult, Outlook,
        SentimentResult, SentimentItem, EventsResult, NewsEvent,
        HeatLevel, Stance, Judgment, EventType,
    )
    ai = AnalysisReport(
        theme_keywords=ThemeKeywordsResult(keywords=[ThemeKeyword(word="降准", score=0.9, explain="央行降准释放流动性支持实体经济")]),
        policy_direction=PolicyDirectionResult(direction="扩张", confidence=0.8,
                                               keywords=["降准", "消费券"], interpretation="央行降准释放长期流动性,体现明显的政策扩张倾向"),
        industries=IndustriesResult(industries=[IndustryFocus(name="新能源", heat=HeatLevel.HIGH,
                                                              article_count=2, summary="销量持续增长", stance=Stance.POSITIVE)]),
        policies=PoliciesResult(policies=[]),
        core_insights=CoreInsightsResult(insights="昨日经济新闻聚焦货币政策宽松与产业升级,降准为市场注入流动性"),
        outlooks=OutlooksResult(outlooks=[Outlook(topic="货币宽松", judgment=Judgment.SUPPORT, rationale="降准信号明确后续仍可能下调LPR再次降息空间打开")]),
        sentiment=SentimentResult(items=[
            SentimentItem(target="新能源", stance=Stance.POSITIVE, intensity=0.8, evidence="销量增长"),
            SentimentItem(target="房地产", stance=Stance.NEGATIVE, intensity=0.6, evidence="投资下滑"),
        ]),
        events=EventsResult(events=[NewsEvent(subject="中国人民银行", action="宣布下调存款准备金率", object="释放长期资金一亿元",
                                              event_type=EventType.MONETARY, impact="利好银行地产板块释放流动性")]),
    )
    qm = compute(freq, ai, hit_industries=["新能源", "房地产"], total_industries=12)
    print("metrics:", qm.as_dict())
