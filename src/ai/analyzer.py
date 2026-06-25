"""analyzer - 10 维 AI 分析编排器 (v2)

设计要点:
- 9 个主任务并发执行 (asyncio.Semaphore)
- 第 10 个任务 self_eval 作为第二阶段,依赖主任务结果
- 每个任务独立错误处理,失败时用 NLP 兜底
- 输出统一为 Pydantic AnalysisReport (强类型,可序列化)
- 集成 cache (router.chat_json 内部)+ trace

新维度的对应关系 (vs 旧版):
+ sentiment      (立场/情感)
+ events         (主谓宾事件)
+ cross_links    (跨文章聚类)
+ self_eval      (AI 自评 - 第二阶段)
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

from src.ai.parallel import TaskResult, run_two_stage_sync
from src.ai.prompts import (
    SELF_EVAL_PROMPT,

    SYSTEM_PROMPT,
    TASK_REGISTRY,
    format_articles_for_llm,
    format_industry_hint,
)
from src.ai.router import ModelRouter, get_default_router
from src.ai.schema import (
    AnalysisReport,
    CoreInsightsResult,
    CrossLinksResult,
    EventsResult,
    IndustriesResult,
    OutlooksResult,
    PoliciesResult,
    PolicyDirectionResult,
    SelfEval,
    SentimentResult,
    ThemeKeywordsResult,
)
from src.utils.logger import get_logger

logger = get_logger("ai.analyzer")


# 任务 -> Schema 映射
TASK_SCHEMA_MAP = {
    "theme_keywords": ThemeKeywordsResult,
    "policy_direction": PolicyDirectionResult,
    "industries": IndustriesResult,
    "policies": PoliciesResult,
    "core_insights": CoreInsightsResult,
    "outlooks": OutlooksResult,
    "sentiment": SentimentResult,
    "events": EventsResult,
    "cross_links": CrossLinksResult,
    "self_eval": SelfEval,
}


def _fallback_for(task_name, articles, nlp_stats):
    """每个任务的 NLP 兜底结果。"""
    if task_name == "theme_keywords" and nlp_stats:
        from src.ai.schema import ThemeKeyword
        return ThemeKeywordsResult(
            keywords=[
                ThemeKeyword(word=w, score=s, explain="(NLP 兜底,非 AI)")
                for w, s in (nlp_stats.keywords or [])[:10]
            ]
        )
    if task_name == "industries" and nlp_stats and nlp_stats.industry_hits:
        from src.ai.schema import IndustryFocus, HeatLevel, Stance
        return IndustriesResult(industries=[
            IndustryFocus(
                name=name, heat=HeatLevel.HIGH if cnt >= 3 else HeatLevel.MEDIUM,
                article_count=cnt, summary="(NLP 兜底,非 AI)", stance=Stance.NEUTRAL,
            )
            for name, cnt in sorted(
                nlp_stats.industry_hits.items(), key=lambda x: x[1], reverse=True
            )[:5]
        ])
    if task_name == "core_insights":
        return CoreInsightsResult(insights=(
            "昨日共 " + str(len(articles)) + " 篇经济新闻。"
            "AI 分析不可用,本报告为纯 NLP 摘要。"
        ))
    return TASK_SCHEMA_MAP[task_name]()


def _truncate_long_strings(obj, max_len=200):
    """递归截断 dict/list 里过长的字符串, 避免 LLM 越界触发 schema 失败。"""
    if isinstance(obj, dict):
        return {k: _truncate_long_strings(v, max_len) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_truncate_long_strings(x, max_len) for x in obj]
    if isinstance(obj, str):
        if len(obj) > max_len:
            return obj[:max_len].rstrip() + "..."
        return obj
    return obj


def _safe_validate(task_name, raw):
    """带截断回退的 schema 校验: 先尝试原数据, 失败时降级为更激进截断。"""
    schema_cls = TASK_SCHEMA_MAP[task_name]
    try:
        return schema_cls.model_validate(raw)
    except Exception as e1:
        # 第一次失败: 截断到 80 字符
        try:
            return schema_cls.model_validate(_truncate_long_strings(raw, max_len=80))
        except Exception as e2:
            # 第二次失败: 截断到 30 字符 (events/object 等 schema 限制)
            try:
                return schema_cls.model_validate(_truncate_long_strings(raw, max_len=30))
            except Exception as e3:
                # 第三次失败: 截断到 15 字符 (subject/title 等)
                try:
                    return schema_cls.model_validate(_truncate_long_strings(raw, max_len=15))
                except Exception as e4:
                    logger.warning(f"[{task_name}] schema 校验失败 (三次截断仍失败): {e4}")
                    raise


def _format_prior_results(prior):
    """把第一阶段 9 个任务结果格式化为 self_eval 的输入。"""
    parts = []
    for name, obj in prior.items():
        try:
            d = obj.model_dump()
            parts.append("## " + name + "\n\`\`\`json\n" + json.dumps(d, ensure_ascii=False, indent=2)[:2000] + "\n\`\`\`")
        except Exception:
            parts.append("## " + name + "\n(序列化失败)")
    return "\n\n".join(parts)


def _build_messages(task_name, template, articles, nlp_stats, date, prior_results=None):
    text = format_articles_for_llm(articles)
    hint = format_industry_hint(nlp_stats)
    if task_name == "self_eval":
        if not prior_results:
            raise ValueError("self_eval 需要第一阶段结果")
        user_content = template.format(prior_results=_format_prior_results(prior_results))
    else:
        user_content = template.format(articles_text=text, industry_hint=hint)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


async def _async_call(router, task_name, date, articles, messages, temperature, max_tokens):
    return await asyncio.to_thread(
        router.chat_json,
        messages=messages, task_name=task_name, date=date,
        articles=articles, temperature=temperature, max_tokens=max_tokens,
    )


def _make_task_builder(router, task_name, date, articles, nlp_stats, prior_results_provider, temperature, max_tokens):
    template = SELF_EVAL_PROMPT if task_name == "self_eval" else TASK_REGISTRY[task_name]

    async def _runner():
        prior = prior_results_provider() if task_name == "self_eval" else None
        msgs = _build_messages(task_name, template, articles, nlp_stats, date, prior)
        cap = max_tokens
        if task_name in ("cross_links", "events", "sentiment"):
            cap = min(3000, max_tokens * 2)
        if task_name == "self_eval":
            cap = 2000
        if task_name == "core_insights":
            cap = 1000
        raw = await _async_call(router, task_name, date, articles, msgs, temperature, cap)
        try:
            return _safe_validate(task_name, raw)
        except Exception as e:
            logger.warning("[" + task_name + "] schema 校验失败: " + str(e) + "; raw=" + str(raw)[:200])
            raise

    return _runner


def analyze_all(articles, nlp_stats=None, date="", router=None, concurrency=4, temperature=0.3, max_tokens=2000, enable_new_tasks=True):
    if router is None:
        try:
            router = get_default_router()
        except Exception as e:
            logger.error("无法初始化 router: " + str(e))
            return _build_pure_nlp_report(articles, nlp_stats, date)

    if not articles:
        return AnalysisReport(
            theme_keywords=ThemeKeywordsResult(keywords=[]),
            policy_direction=PolicyDirectionResult(direction="中性", confidence=0.0, keywords=[], interpretation="(无文章)"),
            industries=IndustriesResult(industries=[]),
            policies=PoliciesResult(policies=[]),
            core_insights=CoreInsightsResult(insights="(无文章)"),
            outlooks=OutlooksResult(outlooks=[]),
        )

    stage1_tasks = list(TASK_REGISTRY.keys())
    if not enable_new_tasks:
        stage1_tasks = [t for t in stage1_tasks if t not in ("sentiment", "events", "cross_links")]

    prior_holder = {}

    def prior_provider():
        return dict(prior_holder)

    stage1 = {
        t: _make_task_builder(router, t, date, articles, nlp_stats, prior_provider, temperature, max_tokens)
        for t in stage1_tasks
    }

    def stage2_factory(stage1_data):
        prior_holder.update(stage1_data)
        if enable_new_tasks:
            return {
                "self_eval": _make_task_builder(
                    router, "self_eval", date, articles, nlp_stats,
                    prior_provider, temperature, max_tokens,
                )
            }
        return {}

    logger.info("AI 分析开始: stage1=" + str(len(stage1)) + " 任务, 并发=" + str(concurrency) + ", 文章数=" + str(len(articles)))
    results = run_two_stage_sync(stage1, stage2_factory, concurrency)

    for name, r in results.items():
        status = "OK" if r.success else "FAIL"
        logger.info("  [" + status + "] " + name + ": " + str(r.latency_ms) + "ms" + ((" err=" + r.error) if not r.success else ""))

    def _get(name):
        r = results.get(name)
        if r and r.success and r.data:
            return r.data
        return _fallback_for(name, articles, nlp_stats)

    _se = results.get("self_eval")
    return AnalysisReport(
        theme_keywords=_get("theme_keywords"),
        policy_direction=_get("policy_direction"),
        industries=_get("industries"),
        policies=_get("policies"),
        core_insights=_get("core_insights"),
        outlooks=_get("outlooks"),
        sentiment=_get("sentiment"),
        events=_get("events"),
        cross_links=_get("cross_links"),
        self_eval=(_se.data if _se is not None and _se.success else None),
    )


def _build_pure_nlp_report(articles, nlp_stats, date):
    return AnalysisReport(
        theme_keywords=_fallback_for("theme_keywords", articles, nlp_stats),
        policy_direction=PolicyDirectionResult(direction="中性", confidence=0.0, keywords=[], interpretation="(LLM 不可用,跳过政策风向)"),
        industries=_fallback_for("industries", articles, nlp_stats),
        policies=PoliciesResult(policies=[]),
        core_insights=_fallback_for("core_insights", articles, nlp_stats),
        outlooks=OutlooksResult(outlooks=[]),
    )


def analyze_for_date(date, articles, nlp_stats=None, **kwargs):
    return analyze_all(articles=articles, nlp_stats=nlp_stats, date=date, **kwargs)


if __name__ == "__main__":
    from src.scraper.pipeline import Article
    sample = [
        Article(
            title="央行宣布降准1万亿", content=["央行宣布降准释放长期资金约1万亿元,支持实体经济"],
            content_text="央行宣布降准释放长期资金约1万亿元,支持实体经济,这是年内第二次降准。",
            source="新华社", publish_time="2026-06-12 10:00", channel="金融", url="https://example.com/1",
        ),
        Article(
            title="新能源汽车5月销量同比+38.5%", content=["中国新能源汽车5月销量同比增长38.5%"],
            content_text="中国新能源汽车5月销量同比增长38.5%,市场渗透率突破47%,产业进入高质量发展新阶段。",
            source="经济日报", publish_time="2026-06-12 14:00", channel="汽车", url="https://example.com/2",
        ),
    ]

    @dataclass
    class FakeNLP:
        keywords = field(default_factory=lambda: [("降准", 0.9), ("新能源", 0.8), ("汽车", 0.7)])
        industry_hits = field(default_factory=lambda: {"新能源": 2, "金融": 1})

    import sys
    if "--real" in sys.argv:
        report = analyze_for_date("2026-06-25", sample, FakeNLP(), enable_new_tasks=True)
        print("=== 10 维分析报告 ===")
        print("主题词:", [k.word for k in report.theme_keywords.keywords])
        print("风向:", report.policy_direction.direction, "(" + str(round(report.policy_direction.confidence * 100)) + "%)")
        print("产业:", [(i.name, i.stance) for i in report.industries.industries])
        print("政策:", len(report.policies.policies), "条")
        print("判断:", len(report.outlooks.outlooks), "条")
        print("立场:", len(report.sentiment.items), "条")
        print("事件:", len(report.events.events), "条")
        print("聚类:", len(report.cross_links.links), "个")
        print("自评:", report.self_eval.overall if report.self_eval else "N/A")
        print("JSON dump:")
        print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2)[:1500])
    else:
        print("Mock 自检模式,加 --real 跑真实 API")
        from unittest.mock import patch
        with patch.object(ModelRouter, "chat_json", side_effect=Exception("mock fail")):
            rpt = analyze_for_date("2026-06-25", sample, FakeNLP(), enable_new_tasks=False)
            assert len(rpt.theme_keywords.keywords) >= 1
            assert len(rpt.industries.industries) >= 1
            assert rpt.policy_direction.direction == "中性"
            print("Mock 降级: theme=" + str(len(rpt.theme_keywords.keywords)) + ", industry=" + str(len(rpt.industries.industries)))
        print("Mock self-tests passed")
