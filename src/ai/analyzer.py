"""analyzer — 6 个 LLM 任务串联 ==============================================
主入口 analyze_all(articles, nlp_stats) -> AnalysisResult。

任何任务失败都用规则降级,确保报告仍能生成。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.ai.client import get_default_client, LLMClient
from src.ai.prompts import (
    CORE_INSIGHTS_PROMPT,
    FUTURE_OUTLOOK_PROMPT,
    INDUSTRY_FOCUS_PROMPT,
    POLICY_DETAIL_PROMPT,
    POLICY_DIRECTION_PROMPT,
    SYSTEM_PROMPT,
    THEME_KEYWORDS_PROMPT,
    format_articles_for_llm,
    format_industry_hint,
)
from src.utils.logger import get_logger

logger = get_logger("ai.analyzer")


@dataclass
class ThemeKeyword:
    word: str = ""
    score: float = 0.0
    explain: str = ""


@dataclass
class PolicyEvent:
    title: str = ""
    issuer: str = ""
    content: str = ""
    source_article: str = ""


@dataclass
class IndustryFocus:
    name: str = ""
    heat: str = "中"
    article_count: int = 0
    summary: str = ""


@dataclass
class Outlook:
    topic: str = ""
    judgment: str = "中性"
    rationale: str = ""


@dataclass
class AnalysisResult:
    """6 维度 AI 分析结果聚合。"""

    theme_keywords: List[ThemeKeyword] = field(default_factory=list)
    policy_direction: Dict[str, Any] = field(default_factory=dict)
    industries: List[IndustryFocus] = field(default_factory=list)
    policies: List[PolicyEvent] = field(default_factory=list)
    core_insights: str = ""
    outlooks: List[Outlook] = field(default_factory=list)

    def is_valid(self) -> bool:
        """判断是否有任何有效内容(用于决定是否跳过报告)。"""
        return bool(
            self.theme_keywords
            or self.policy_direction
            or self.industries
            or self.policies
            or self.core_insights
            or self.outlooks
        )


# ============================================================
# 单任务包装
# ============================================================
def _safe_chat_json(client: LLMClient, user_prompt: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    """调用 LLM 并安全返回 dict,失败时返回 fallback。"""
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return client.chat_json(messages, temperature=0.3, max_tokens=2000)
    except Exception as e:
        logger.warning(f"LLM 任务失败,使用降级结果: {e}")
        return fallback


# ============================================================
# 6 个任务(带降级)
# ============================================================
def _task_theme_keywords(client, articles) -> List[ThemeKeyword]:
    text = format_articles_for_llm(articles)
    prompt = THEME_KEYWORDS_PROMPT.format(articles_text=text)
    result = _safe_chat_json(client, prompt, fallback={"keywords": []})
    return [ThemeKeyword(**kw) for kw in result.get("keywords", [])][:10]


def _task_policy_direction(client, articles) -> Dict[str, Any]:
    text = format_articles_for_llm(articles)
    prompt = POLICY_DIRECTION_PROMPT.format(articles_text=text)
    return _safe_chat_json(client, prompt, fallback={
        "direction": "中性", "confidence": 0.0,
        "keywords": [], "interpretation": "(LLM 不可用,跳过政策风向分析)"
    })


def _task_industries(client, articles, nlp_stats) -> List[IndustryFocus]:
    text = format_articles_for_llm(articles)
    hint = format_industry_hint(nlp_stats)
    prompt = INDUSTRY_FOCUS_PROMPT.format(articles_text=text, industry_hint=hint)
    result = _safe_chat_json(client, prompt, fallback={"industries": []})

    industries = [IndustryFocus(**x) for x in result.get("industries", [])]
    # 降级:若 LLM 没返回,用 NLP 命中填充
    if not industries and nlp_stats and nlp_stats.industry_hits:
        for name, count in sorted(nlp_stats.industry_hits.items(), key=lambda x: x[1], reverse=True)[:5]:
            industries.append(IndustryFocus(
                name=name, heat="高" if count >= 3 else "中",
                article_count=count, summary="(基于 NLP 关键词匹配,未做 AI 归纳)"
            ))
    return industries


def _task_policies(client, articles) -> List[PolicyEvent]:
    text = format_articles_for_llm(articles)
    prompt = POLICY_DETAIL_PROMPT.format(articles_text=text)
    result = _safe_chat_json(client, prompt, fallback={"policies": []})
    return [PolicyEvent(**p) for p in result.get("policies", [])][:5]


def _task_core_insights(client, articles) -> str:
    text = format_articles_for_llm(articles)
    prompt = CORE_INSIGHTS_PROMPT.format(articles_text=text)
    result = _safe_chat_json(client, prompt, fallback={"insights": ""})
    return result.get("insights", "")


def _task_outlooks(client, articles) -> List[Outlook]:
    text = format_articles_for_llm(articles)
    prompt = FUTURE_OUTLOOK_PROMPT.format(articles_text=text)
    result = _safe_chat_json(client, prompt, fallback={"outlooks": []})
    return [Outlook(**o) for o in result.get("outlooks", [])][:5]


# ============================================================
# 主入口
# ============================================================
def analyze_all(articles, nlp_stats=None, client: Optional[LLMClient] = None) -> AnalysisResult:
    """对 Article 列表做 6 维度 AI 分析。

    Args:
        articles: List[Article]
        nlp_stats: NLPStats(用于 industries 任务的降级)
        client: 自定义 LLMClient(便于测试)

    Returns:
        AnalysisResult
    """
    if client is None:
        try:
            client = get_default_client()
        except Exception as e:
            logger.error(f"无法初始化 LLM 客户端: {e}")
            return _build_pure_nlp_fallback(articles, nlp_stats)

    if not articles:
        return AnalysisResult()

    result = AnalysisResult()
    logger.info("AI 分析开始(共 6 个任务)")

    try:
        result.theme_keywords = _task_theme_keywords(client, articles)
        logger.info(f"  [1/6] 主题词: {len(result.theme_keywords)} 条")
    except Exception as e:
        logger.warning(f"  [1/6] 主题词失败: {e}")

    try:
        result.policy_direction = _task_policy_direction(client, articles)
        logger.info(f"  [2/6] 政策风向: {result.policy_direction.get('direction', '?')}")
    except Exception as e:
        logger.warning(f"  [2/6] 政策风向失败: {e}")

    try:
        result.industries = _task_industries(client, articles, nlp_stats)
        logger.info(f"  [3/6] 重点产业: {len(result.industries)} 个")
    except Exception as e:
        logger.warning(f"  [3/6] 重点产业失败: {e}")

    try:
        result.policies = _task_policies(client, articles)
        logger.info(f"  [4/6] 重点政策: {len(result.policies)} 条")
    except Exception as e:
        logger.warning(f"  [4/6] 重点政策失败: {e}")

    try:
        result.core_insights = _task_core_insights(client, articles)
        logger.info(f"  [5/6] 核心信息: {len(result.core_insights)} 字")
    except Exception as e:
        logger.warning(f"  [5/6] 核心信息失败: {e}")

    try:
        result.outlooks = _task_outlooks(client, articles)
        logger.info(f"  [6/6] 未来判断: {len(result.outlooks)} 条")
    except Exception as e:
        logger.warning(f"  [6/6] 未来判断失败: {e}")

    return result


def _build_pure_nlp_fallback(articles, nlp_stats) -> AnalysisResult:
    """LLM 完全不可用时,用 NLP 统计结果构建降级报告。"""
    result = AnalysisResult()
    if nlp_stats:
        for word, score in nlp_stats.keywords[:10]:
            result.theme_keywords.append(ThemeKeyword(word=word, score=score, explain="(基于 NLP 关键词权重)"))
        for name, count in sorted(nlp_stats.industry_hits.items(), key=lambda x: x[1], reverse=True)[:5]:
            result.industries.append(IndustryFocus(
                name=name, heat="高" if count >= 3 else "中",
                article_count=count, summary="(纯 NLP 命中,未做 AI 归纳)"
            ))
        result.core_insights = (
            f"昨日共 {len(articles)} 篇经济新闻,"
            f"主要话题涉及 {' / '.join(w for w, _ in nlp_stats.keywords[:5])} 等。"
            f"(LLM 不可用,此为纯 NLP 摘要)"
        )
    return result


if __name__ == "__main__":
    # 演示用法(需要 .env 中配置 AI_API_KEY)
    from src.scraper.pipeline import Article

    sample = [
        Article(
            title="央行宣布降准1万亿",
            content=["央行宣布降准释放长期资金约1万亿元,支持实体经济"],
            content_text="央行宣布降准释放长期资金约1万亿元,支持实体经济",
            source="新华社", publish_time="2026-06-11 10:00", channel="金融",
        ),
        Article(
            title="新能源汽车销量创新高",
            content=["新能源汽车销量持续增长,产业进入高质量发展新阶段"],
            content_text="新能源汽车销量持续增长,产业进入高质量发展新阶段",
            source="经济日报", publish_time="2026-06-11 14:00", channel="产业",
        ),
    ]
    r = analyze_all(sample)
    print(f"主题词: {[k.word for k in r.theme_keywords]}")
    print(f"政策风向: {r.policy_direction}")
    print(f"产业: {[i.name for i in r.industries]}")
    print(f"核心信息: {r.core_insights}")