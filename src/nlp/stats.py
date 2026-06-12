"""stats — 词频统计与产业匹配 ==============================================
主入口 analyze(articles) -> NLPStats,供上层调用。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from src.config import INDUSTRY_KEYWORDS
from src.nlp.keywords import (
    extract_tfidf_keywords,
    extract_textrank_keywords,
    merge_keywords,
)
from src.nlp.tokenizer import tokenize, tokenize_articles
from src.utils.logger import get_logger

logger = get_logger("nlp.stats")


@dataclass
class NLPStats:
    """NLP 分析结果聚合,贯穿到报告层。"""

    word_freq: List[Tuple[str, int]] = field(default_factory=list)  # top 50
    keywords: List[Tuple[str, float]] = field(default_factory=list)  # 融合后 top 20
    industry_hits: Dict[str, int] = field(default_factory=dict)  # 产业 -> 命中次数
    industry_articles: Dict[str, List[str]] = field(default_factory=dict)  # 产业 -> 涉及文章标题
    top_keywords_by_industry: Dict[str, List[str]] = field(default_factory=dict)
    total_words: int = 0
    article_count: int = 0


def _word_freq(tokens: List[str], top_k: int = 50) -> List[Tuple[str, int]]:
    return Counter(tokens).most_common(top_k)


def _industry_match(articles) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    """扫描所有文章正文,匹配产业关键词。"""
    hits: Dict[str, int] = {}
    art_hits: Dict[str, List[str]] = {}

    for industry, keywords in INDUSTRY_KEYWORDS.items():
        count = 0
        matched_titles: List[str] = []
        for art in articles:
            full = art.content_text
            for kw in keywords:
                if kw in full:
                    count += full.count(kw)
                    if art.title and art.title not in matched_titles:
                        matched_titles.append(art.title)
                    break  # 一篇文章对一个产业只计 1 次
        if count > 0:
            hits[industry] = count
            art_hits[industry] = matched_titles[:5]

    return hits, art_hits


def _top_keywords_per_industry(articles, top_k: int = 5) -> Dict[str, List[str]]:
    """每个产业下,提取该产业相关文章的 top 关键词。"""
    result: Dict[str, List[str]] = {}
    for industry, keywords in INDUSTRY_KEYWORDS.items():
        related_texts: List[str] = []
        for art in articles:
            full = art.content_text
            if any(kw in full for kw in keywords):
                related_texts.extend(art.content)
        if not related_texts:
            continue
        tokens = []
        for p in related_texts:
            tokens.extend(tokenize(p))
        top = Counter(tokens).most_common(top_k)
        result[industry] = [w for w, _ in top]
    return result


def analyze(articles) -> NLPStats:
    """对 Article 列表做完整 NLP 分析。

    Args:
        articles: List[Article](src.scraper.pipeline.Article)

    Returns:
        NLPStats
    """
    stats = NLPStats()
    stats.article_count = len(articles)

    if not articles:
        return stats

    # 收集所有段落
    all_paragraphs = []
    for art in articles:
        all_paragraphs.extend(art.content)

    # 词频
    all_tokens = tokenize_articles(articles, remove_stop=True, min_len=2)
    stats.total_words = len(all_tokens)
    stats.word_freq = _word_freq(all_tokens, top_k=50)

    # 关键词(TF-IDF + TextRank 融合)
    tfidf = extract_tfidf_keywords(all_paragraphs, top_k=30)
    textrank = extract_textrank_keywords(all_paragraphs, top_k=30)
    stats.keywords = merge_keywords(tfidf, textrank, top_k=20)

    # 产业匹配
    stats.industry_hits, stats.industry_articles = _industry_match(articles)
    stats.top_keywords_by_industry = _top_keywords_per_industry(articles)

    logger.info(
        f"NLP 分析完成: {stats.article_count} 篇, {stats.total_words} 词, "
        f"{len(stats.keywords)} 关键词, {len(stats.industry_hits)} 产业命中"
    )
    return stats


if __name__ == "__main__":
    from src.scraper.pipeline import Article

    sample = [
        Article(
            title="新能源汽车产业新阶段",
            content=["新能源汽车销量持续增长", "光伏行业产能扩张"],
            content_text="新能源汽车销量持续增长,光伏行业产能扩张",
        ),
        Article(
            title="央行降准1万亿",
            content=["央行宣布降准释放长期资金", "金融监管持续完善"],
            content_text="央行宣布降准释放长期资金,金融监管持续完善",
        ),
    ]
    s = analyze(sample)
    print("Top 10:", s.word_freq[:10])
    print("Keywords:", s.keywords[:10])
    print("Industries:", s.industry_hits)