"""renderer - 渲染 Markdown 报告 (升级版)

新增 sections:
- 量化指标卡 (情绪指数 / 政策倾向 / 注意力熵 / 头部集中度)
- 跨日对比表 (日环比)
- 热点涌现 / 衰退
- 知识图谱嵌入
- 产业-A股映射
- AI 自评
- 历史报告链接 (仪表盘)
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.ai.schema import AnalysisReport
from src.analysis.metrics import compute as compute_metrics
from src.config import AI_MODEL, AI_PROVIDER, REPORT_DIR
from src.dashboard.builder import render as render_dashboard
from src.nlp.stats import NLPStats
from src.nlp.wordcloud_gen import generate_wordcloud
from src.scraper.pipeline import Article
from src.stocks.mapping import map_industries_to_stocks, related_stocks
from src.storage import repository as repo
from src.trend.comparison import compare as compare_trend
from src.trend.emerging import detect as detect_emerging
from src.utils.date_utils import parse_date, previous_business_day, same_weekday_last_week
from src.utils.logger import get_logger

logger = get_logger("report.renderer")

_TEMPLATE_DIR = Path(__file__).resolve().parent
_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=select_autoescape(enabled_extensions=("j2",)),
    trim_blocks=True,
    lstrip_blocks=True,
)


def _stance_color(stance):
    return {"利好": "🟢", "利空": "🔴", "中性": "🟡"}.get(stance, "⚪")


def _stance_emoji_heat(heat):
    return {"高": "🔥", "中": "🔸", "低": "🔹"}.get(heat, "⚪")


def _direction_arrow(direction):
    return {"up": "↑", "down": "↓", "flat": "→"}.get(direction, "→")


_env.filters["stance_color"] = _stance_color
_env.filters["heat_emoji"] = _stance_emoji_heat
_env.filters["arrow"] = _direction_arrow
_env.filters["sign_color"] = lambda v: ("#7ee787" if (v or 0) > 0 else ("#ff7b72" if (v or 0) < 0 else "#8b949e"))
_env.filters["fmt_pct"] = lambda v: f"{v:+.2f}%" if v is not None else "—"
_env.filters["fmt_num"] = lambda v, d=2: (f"{v:.{d}f}" if isinstance(v, (int, float)) else (v or "—"))
_env.filters["truncate"] = lambda s, n=80: (s if len(s or "") <= n else (s[:n] + "…"))
_env.filters["stance_emoji"] = _stance_color
_env.filters["confidence_bar"] = lambda conf: ("█" * int(round((conf or 0) * 10)) + "░" * (10 - int(round((conf or 0) * 10))))
_env.filters["length"] = lambda v: (0 if v is None else (len(list(v)) if hasattr(v, "__iter__") and not isinstance(v, (str, bytes, dict)) else len(v)))


def _make_sentiment_bars(items):
    if not items:
        return "  (无数据)"
    lines = []
    max_intensity = max((it.intensity for it in items), default=1.0) or 1.0
    for it in items[:8]:
        bar_len = int((it.intensity / max_intensity) * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        lines.append("  " + it.target.ljust(10) + " " + bar + " " + str(round(it.intensity * 100, 1)) + "%  " + it.stance)
    return "\n".join(lines)



def _build_context(articles, nlp_stats, ai_result, report_date, kg_image=None, advanced=None):
    hit_industries = list(nlp_stats.industry_hits.keys())
    total_industries = 12
    metrics = compute_metrics(nlp_stats.word_freq, ai_result, hit_industries, total_industries)

    # 跨日对比
    try:
        prev_dt = previous_business_day(parse_date(report_date))
        week_dt = same_weekday_last_week(parse_date(report_date))
        cmp_res = compare_trend(report_date,
                                prev_date=prev_dt.isoformat(),
                                week_date=week_dt.isoformat())
    except Exception as e:
        logger.warning(f"compare_trend failed: {e}")
        cmp_res = None

    # 涌现 / 衰退
    try:
        trend_report = detect_emerging(report_date, recent_days=3, prev_days=7)
    except Exception as e:
        logger.warning(f"detect_emerging failed: {e}")
        trend_report = None

    # 行业-A股映射
    industry_stocks = map_industries_to_stocks(hit_industries)

    # 写库
    try:
        from src.storage.repository import (
            ArticleRow, AIReportRow, DailyMetric, EntityRow, IndustryRow, KeywordRow,
        )
        # article
        repo.upsert_articles(report_date, [
            ArticleRow(date=report_date, article_id=a.article_id or str(i),
                       title=a.title, url=a.url, source=a.source,
                       publish_time=a.publish_time, channel=a.channel,
                       word_count=a.word_count, content_text=a.content_text)
            for i, a in enumerate(articles)
        ])
        # keyword
        theme_words = {k.word for k in ai_result.theme_keywords.keywords}
        kw_rows = []
        for w, c in nlp_stats.word_freq[:100]:
            kw_rows.append(KeywordRow(
                date=report_date, keyword=w, freq=c,
                is_theme=1 if w in theme_words else 0,
                theme_score=next((k.score for k in ai_result.theme_keywords.keywords if k.word == w), 0.0),
            ))
        repo.upsert_keywords(report_date, kw_rows)
        # industry
        industry_rows = []
        for ind in ai_result.industries.industries:
            industry_rows.append(IndustryRow(
                date=report_date, industry=ind.name, hit_count=ind.article_count,
                article_count=ind.article_count, heat=ind.heat, stance=str(ind.stance),
            ))
        repo.upsert_industries(report_date, industry_rows)
        # entity
        from src.kg.entities import extract as extract_entities
        ent_rows = [EntityRow(date=report_date, entity_type=t, entity_value=v, freq=freq)
                    for (t, v, freq) in extract_entities(articles)]
        repo.upsert_entities(report_date, ent_rows)
        # daily metric
        m = DailyMetric(
            date=report_date,
            article_count=len(articles),
            total_words=nlp_stats.total_words,
            unique_keywords=len(set(w for w, _ in nlp_stats.word_freq)),
            policy_stance_score=metrics.policy_stance_score,
            sentiment_index=metrics.sentiment_index,
            attention_entropy=metrics.attention_entropy,
            attention_top_share=metrics.attention_top_share,
            industry_count=len(hit_industries),
            policy_count=len(ai_result.policies.policies),
            event_count=len(ai_result.events.events),
            raw_json=json.dumps(metrics.as_dict(), ensure_ascii=False),
        )
        repo.upsert_metric(m)
        # ai_report
        ai_row = AIReportRow(
            date=report_date, provider=AI_PROVIDER, model=AI_MODEL,
            payload_json=json.dumps(ai_result.model_dump(), ensure_ascii=False, default=str),
            self_eval_consistency=(ai_result.self_eval.consistency if ai_result.self_eval else 0.0),
            self_eval_groundedness=(ai_result.self_eval.groundedness if ai_result.self_eval else 0.0),
            self_eval_completeness=(ai_result.self_eval.completeness if ai_result.self_eval else 0.0),
            self_eval_overall=(ai_result.self_eval.overall if ai_result.self_eval else 0.0),
            created_at=datetime.now().isoformat(),
        )
        repo.upsert_ai_report(ai_row)
    except Exception as e:
        logger.warning(f"DB persist failed: {e}")

    return {
        "date": report_date,
        "article_count": len(articles),
        "total_words": nlp_stats.total_words,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "core_insights": ai_result.core_insights.insights,
        "theme_keywords": [
            {"word": k.word, "score": k.score, "explain": k.explain}
            for k in ai_result.theme_keywords.keywords
        ],
        "policy_direction": {
            "direction": ai_result.policy_direction.direction,
            "confidence": ai_result.policy_direction.confidence,
            "keywords": ai_result.policy_direction.keywords,
            "interpretation": ai_result.policy_direction.interpretation,
        },
        "industries": [
            {"name": i.name, "heat": i.heat, "article_count": i.article_count,
             "summary": i.summary, "stance": i.stance}
            for i in ai_result.industries.industries
        ],
        "policies": [
            {"title": p.title, "issuer": p.issuer, "content": p.content,
             "source_article": p.source_article}
            for p in ai_result.policies.policies
        ],
        "outlooks": [
            {"topic": o.topic, "judgment": o.judgment, "rationale": o.rationale}
            for o in ai_result.outlooks.outlooks
        ],
        "sentiment": [
            {"target": it.target, "stance": it.stance, "intensity": it.intensity, "evidence": it.evidence}
            for it in ai_result.sentiment.items
        ],
        "sentiment_bars": _make_sentiment_bars(ai_result.sentiment.items),
        "events": [
            {"subject": e.subject, "action": e.action, "object": e.object,
             "event_type": e.event_type, "impact": e.impact}
            for e in ai_result.events.events
        ],
        "cross_links": [
            {"cluster": c.cluster, "article_indices": c.article_indices, "summary": c.summary}
            for c in ai_result.cross_links.links
        ],
        "self_eval": (
            {"consistency": ai_result.self_eval.consistency,
             "groundedness": ai_result.self_eval.groundedness,
             "completeness": ai_result.self_eval.completeness,
             "overall": ai_result.self_eval.overall,
             "comments": ai_result.self_eval.comments}
            if ai_result.self_eval else None
        ),
        "articles": [
            {"title": a.title, "url": a.url, "source": a.source,
             "publish_time": a.publish_time, "channel": a.channel,
             "word_count": a.word_count}
            for a in articles
        ],
        "metrics": metrics.as_dict(),
        "comparison": (cmp_res.as_dict() if cmp_res else None),
        "trend": (
            {
                "emerging": [t.keyword for t in trend_report.emerging[:10]],
                "declining": [t.keyword for t in trend_report.declining[:10]],
                "persistent": [t.keyword for t in trend_report.persistent[:10]],
            }
            if trend_report else None
        ),
        "industry_stocks": industry_stocks,
        "kg_image": kg_image,
        "dashboard_url": f"../dashboard/{report_date}.html",
        "wordcloud_relpath": None,
        # advanced: 9 个前沿模块, dataclass 全部转 dict 给模板
        "advanced": {
            "anomaly": (advanced.get("anomaly").as_dict() if advanced and advanced.get("anomaly") and hasattr(advanced.get("anomaly"), "as_dict") else (advanced or {}).get("anomaly")) if advanced else None,
            "forecast": (advanced.get("forecast").as_dict() if advanced and advanced.get("forecast") and hasattr(advanced.get("forecast"), "as_dict") else (advanced or {}).get("forecast")) if advanced else None,
            "market": (advanced.get("market").as_dict() if advanced and advanced.get("market") and hasattr(advanced.get("market"), "as_dict") else (advanced or {}).get("market")) if advanced else None,
            "macro": (advanced or {}).get("macro"),
            "topics": (advanced or {}).get("topics"),
            "events_study": (advanced or {}).get("events_study"),
            "rag": (advanced or {}).get("rag"),
            "volatility": (advanced or {}).get("volatility"),
            "causal": (advanced or {}).get("causal"),
            # v5 新增三大模块
            "signal": (advanced.get("signal").as_dict() if advanced and advanced.get("signal") and hasattr(advanced.get("signal"), "as_dict") else (advanced or {}).get("signal")) if advanced else None,
            "backtest": (advanced.get("backtest").as_dict() if advanced and advanced.get("backtest") and hasattr(advanced.get("backtest"), "as_dict") else (advanced or {}).get("backtest")) if advanced else None,
            "narrative": (advanced.get("narrative").as_dict() if advanced and advanced.get("narrative") and hasattr(advanced.get("narrative"), "as_dict") else (advanced or {}).get("narrative")) if advanced else None,
            # v6 四大模块
            "risk": (advanced.get("risk").as_dict() if advanced and advanced.get("risk") and hasattr(advanced.get("risk"), "as_dict") else (advanced or {}).get("risk")) if advanced else None,
            "portfolio": (advanced.get("portfolio").as_dict() if advanced and advanced.get("portfolio") and hasattr(advanced.get("portfolio"), "as_dict") else (advanced or {}).get("portfolio")) if advanced else None,
            "scenario": (advanced.get("scenario").as_dict() if advanced and advanced.get("scenario") and hasattr(advanced.get("scenario"), "as_dict") else (advanced or {}).get("scenario")) if advanced else None,
            "qa": (advanced.get("qa").as_dict() if advanced and advanced.get("qa") and hasattr(advanced.get("qa"), "as_dict") else (advanced or {}).get("qa")) if advanced else None,
            # v7 前沿: 多智能体辩论 + SHAP 解释
            "council_debate": (advanced.get("council_debate").as_dict() if advanced and advanced.get("council_debate") and hasattr(advanced.get("council_debate"), "as_dict") else (advanced or {}).get("council_debate")) if advanced else None,
            "shap": (advanced.get("shap").as_dict() if advanced and advanced.get("shap") and hasattr(advanced.get("shap"), "as_dict") else (advanced or {}).get("shap")) if advanced else None,
            # v8 前沿: 量化多因子 + 时序预测增强 + 政策 PDF 解析
            "factor_model": (advanced.get("factor_model").as_dict() if advanced and advanced.get("factor_model") and hasattr(advanced.get("factor_model"), "as_dict") else (advanced or {}).get("factor_model")) if advanced else None,
            "forecast_enhanced": (advanced.get("forecast_enhanced").as_dict() if advanced and advanced.get("forecast_enhanced") and hasattr(advanced.get("forecast_enhanced"), "as_dict") else (advanced or {}).get("forecast_enhanced")) if advanced else None,
            "policy_pdf": (advanced.get("policy_pdf").as_dict() if advanced and advanced.get("policy_pdf") and hasattr(advanced.get("policy_pdf"), "as_dict") else (advanced or {}).get("policy_pdf")) if advanced else None,
            # v9 前沿: Embedding + Hamilton 区制 + GraphRAG + ReAct Agent
            "v9": (advanced or {}).get("v9") if advanced else None,
        },
        "ai_provider": AI_PROVIDER,
        "ai_model": AI_MODEL,
    }


def render(articles, nlp_stats, ai_result, report_date, output_filename=None, kg_image=None, advanced=None):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_filename = output_filename or (report_date + ".md")
    output_path = REPORT_DIR / output_filename

    wordcloud_filename = "wordcloud_" + report_date + ".png"
    wordcloud_path = None
    try:
        wordcloud_path = generate_wordcloud(
            word_freq=nlp_stats.word_freq, output_name=wordcloud_filename,
        )
    except Exception as e:
        logger.warning("词云生成失败: " + str(e))

    context = _build_context(articles, nlp_stats, ai_result, report_date, kg_image=kg_image, advanced=advanced)
    if wordcloud_path:
        context["wordcloud_relpath"] = "../images/" + wordcloud_filename

    template = _env.get_template("template.md.j2")
    rendered = template.render(**context)
    output_path.write_text(rendered, encoding="utf-8")
    logger.info("报告已生成: " + str(output_path) + " (" + str(len(rendered)) + " 字符)")
    return output_path


def render_to_string(articles, nlp_stats, ai_result, report_date):
    context = _build_context(articles, nlp_stats, ai_result, report_date)
    template = _env.get_template("template.md.j2")
    return template.render(**context)


if __name__ == "__main__":
    from src.scraper.pipeline import Article
    from src.nlp.stats import NLPStats
    from src.ai.schema import (
        AnalysisReport, ThemeKeywordsResult, ThemeKeyword,
        PolicyDirectionResult, IndustriesResult, IndustryFocus,
        PoliciesResult, CoreInsightsResult, OutlooksResult, Outlook,
        SentimentResult, SentimentItem, EventsResult, NewsEvent,
        CrossLinksResult, CrossLink, SelfEval, HeatLevel, Stance, Judgment, EventType,
    )
    sample_arts = [
        Article(title="央行降准1万亿支持实体经济", url="http://x", source="人民网",
                publish_time="2026-06-25 10:00", channel="金融", word_count=600,
                content_text="央行决定下调存款准备金率0.5个百分点释放长期资金一亿元支持实体经济"),
        Article(title="新能源汽车销量持续增长", url="http://x", source="中国经济网",
                publish_time="2026-06-25 11:00", channel="汽车", word_count=400,
                content_text="中国新能源汽车销量持续增长,渗透率突破百分之五十"),
    ]
    sample_nlp = NLPStats(total_words=2000, article_count=2,
                          word_freq=[("降准", 8), ("新能源", 5), ("实体经济", 4), ("增长", 3)],
                          industry_hits={"金融": 1, "新能源": 1})
    sample_ai = AnalysisReport(
        theme_keywords=ThemeKeywordsResult(keywords=[
            ThemeKeyword(word="降准", score=0.95, explain="央行降准释放长期资金支持实体经济"),
            ThemeKeyword(word="新能源", score=0.85, explain="新能源汽车销量持续增长渗透率提升"),
        ]),
        policy_direction=PolicyDirectionResult(
            direction="扩张", confidence=0.88,
            keywords=["降准", "扩大内需", "新能源"],
            interpretation="央行降准释放长期流动性一亿元,体现明显的政策扩张倾向,叠加新能源支持政策,信号积极。",
        ),
        industries=IndustriesResult(industries=[
            IndustryFocus(name="新能源", heat=HeatLevel.HIGH, article_count=2,
                          summary="销量持续增长,渗透率突破百分之五十", stance=Stance.POSITIVE),
            IndustryFocus(name="金融", heat=HeatLevel.MEDIUM, article_count=1,
                          summary="央行降准释放流动性利好金融板块", stance=Stance.POSITIVE),
        ]),
        policies=PoliciesResult(policies=[]),
        core_insights=CoreInsightsResult(
            insights="昨日经济新闻聚焦货币政策宽松与产业升级,央行降准释放流动性一亿元,新能源产业持续高速增长。",
        ),
        outlooks=OutlooksResult(outlooks=[
            Outlook(topic="货币宽松延续", judgment=Judgment.SUPPORT,
                    rationale="降准信号明确后续仍可能下调LPR再次降息空间打开"),
            Outlook(topic="新能源持续景气", judgment=Judgment.SUPPORT,
                    rationale="销量与渗透率持续突破,产业进入高质量发展新阶段"),
        ]),
        sentiment=SentimentResult(items=[
            SentimentItem(target="新能源", stance=Stance.POSITIVE, intensity=0.85, evidence="销量增长"),
            SentimentItem(target="金融", stance=Stance.POSITIVE, intensity=0.75, evidence="降准利好"),
        ]),
        events=EventsResult(events=[
            NewsEvent(subject="中国人民银行", action="宣布下调存款准备金率",
                      object="释放长期资金一亿元", event_type=EventType.MONETARY,
                      impact="利好银行地产板块释放流动性"),
        ]),
        cross_links=CrossLinksResult(links=[]),
        self_eval=SelfEval(consistency=0.9, groundedness=0.92, completeness=0.85,
                           comments="整体分析自洽,引用充分,覆盖宏观与产业两个维度。"),
    )
    p = render(sample_arts, sample_nlp, sample_ai, "2026-06-25", output_filename="_test.md")
    print(p)
