"""renderer — 渲染 Markdown 报告 ==============================================
Jinja2 模板 + 词云图嵌入 + 写入 reports/{date}.md。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.ai.analyzer import AnalysisResult
from src.config import AI_MODEL, AI_PROVIDER, REPORT_DIR
from src.nlp.stats import NLPStats
from src.nlp.wordcloud_gen import generate_wordcloud
from src.scraper.pipeline import Article
from src.utils.logger import get_logger

logger = get_logger("report.renderer")

_TEMPLATE_DIR = Path(__file__).resolve().parent
_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=select_autoescape(enabled_extensions=("j2",)),
    trim_blocks=True,
    lstrip_blocks=True,
)


def render(
    articles: List[Article],
    nlp_stats: NLPStats,
    ai_result: AnalysisResult,
    report_date: str,
    output_filename: Optional[str] = None,
) -> Path:
    """渲染并写入 Markdown 报告。

    Args:
        articles: 文章列表
        nlp_stats: NLP 分析结果
        ai_result: AI 分析结果
        report_date: YYYY-MM-DD
        output_filename: 自定义文件名,默认 reports/{date}.md

    Returns:
        报告写入的文件路径
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_filename = output_filename or f"{report_date}.md"
    output_path = REPORT_DIR / output_filename

    # --- 1. 生成词云 ---
    wordcloud_filename = f"wordcloud_{report_date}.png"
    wordcloud_path: Optional[Path] = None
    try:
        wordcloud_path = generate_wordcloud(
            word_freq=nlp_stats.word_freq,
            output_name=wordcloud_filename,
        )
    except Exception as e:
        logger.warning(f"词云生成失败: {e}")

    # --- 2. 准备模板变量 ---
    context = {
        "date": report_date,
        "article_count": len(articles),
        "total_words": nlp_stats.total_words,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "core_insights": ai_result.core_insights,
        "theme_keywords": [
            {"word": k.word, "score": k.score, "explain": k.explain}
            for k in ai_result.theme_keywords
        ],
        "policy_direction": ai_result.policy_direction,
        "industries": [
            {"name": i.name, "heat": i.heat, "article_count": i.article_count, "summary": i.summary}
            for i in ai_result.industries
        ],
        "policies": [
            {"title": p.title, "issuer": p.issuer, "content": p.content, "source_article": p.source_article}
            for p in ai_result.policies
        ],
        "outlooks": [
            {"topic": o.topic, "judgment": o.judgment, "rationale": o.rationale}
            for o in ai_result.outlooks
        ],
        "articles": [
            {
                "title": a.title,
                "url": a.url,
                "source": a.source,
                "publish_time": a.publish_time,
                "channel": a.channel,
                "word_count": a.word_count,
            }
            for a in articles
        ],
        "wordcloud_relpath": (
            f"../images/{wordcloud_filename}" if wordcloud_path else None
        ),
        "ai_provider": AI_PROVIDER,
        "ai_model": AI_MODEL,
    }

    # --- 3. 渲染模板 ---
    template = _env.get_template("template.md.j2")
    rendered = template.render(**context)

    # --- 4. 写入文件 ---
    output_path.write_text(rendered, encoding="utf-8")
    logger.info(f"报告已生成: {output_path} ({len(rendered)} 字符)")
    return output_path


if __name__ == "__main__":
    from src.scraper.pipeline import Article
    from src.nlp.stats import NLPStats
    from src.ai.analyzer import (
        AnalysisResult, ThemeKeyword, PolicyEvent, IndustryFocus, Outlook,
    )

    sample_arts = [
        Article(title="示例文章", url="http://x", source="测试", publish_time="2026-06-11", channel="财经", word_count=100),
    ]
    sample_nlp = NLPStats(total_words=100, article_count=1, word_freq=[("示例", 1)])
    sample_ai = AnalysisResult(
        theme_keywords=[ThemeKeyword(word="示例", score=1.0, explain="测试")],
        policy_direction={"direction": "中性", "confidence": 0.5, "keywords": [], "interpretation": "测试"},
        industries=[IndustryFocus(name="测试产业", heat="高", article_count=1, summary="测试")],
        policies=[],
        core_insights="这是一段测试摘要。",
        outlooks=[],
    )
    p = render(sample_arts, sample_nlp, sample_ai, "2026-06-11", output_filename="_test.md")
    print(p)