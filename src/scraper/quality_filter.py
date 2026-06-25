"""scraper.quality_filter - 内容质量过滤"""
from __future__ import annotations

from typing import List

from src.config import INDUSTRY_KEYWORDS
from src.scraper.pipeline import Article
from src.utils.logger import get_logger

logger = get_logger("scraper.quality_filter")

MIN_CONTENT_CHARS = 60
MIN_TITLE_CHARS = 6

SPAM_SIGNALS = ["点击购买", "扫码加微", "咨询电话", "加我微信",
                "广告投放", "招商加盟", "代理分销", "限时优惠"]


def is_ad_or_spam(text):
    if not text:
        return True
    cnt = sum(text.count(s) for s in SPAM_SIGNALS)
    return cnt >= 2


def finance_relevance(text):
    if not text:
        return 0.0
    hits = 0
    for v in INDUSTRY_KEYWORDS.values():
        for kw in v:
            if kw in text:
                hits += 1
    return min(1.0, hits / 8.0)


def quality_score(article):
    title = article.title or ""
    content = article.content_text or ""
    if len(title) < MIN_TITLE_CHARS:
        return 0.0
    if len(content) < MIN_CONTENT_CHARS:
        return 0.2
    if is_ad_or_spam(content):
        return 0.1
    rel = finance_relevance(content + title)
    base = min(1.0, len(content) / 600.0)
    return round(0.5 * rel + 0.5 * base, 4)


def filter_quality(articles, threshold=0.25):
    out = []
    for a in articles:
        s = quality_score(a)
        if s >= threshold:
            out.append(a)
    return out


if __name__ == "__main__":
    a1 = Article(title="央行降准支持实体经济",
                  content_text="央行决定下调存款准备金率0.5个百分点释放长期资金一亿元支持实体经济")
    a2 = Article(title="广告", content_text="点击购买扫码加微咨询电话")
    a3 = Article(title="短", content_text="hi")
    for a in [a1, a2, a3]:
        print(a.title, "->", quality_score(a))
    print("filtered:", len(filter_quality([a1, a2, a3])))
