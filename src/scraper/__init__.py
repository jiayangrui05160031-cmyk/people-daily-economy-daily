"""scraper — 爬虫层 ==============================================
负责从 finance.people.com.cn 和 finance.ce.cn 采集经济新闻。
"""

from .pipeline import fetch_previous_day_articles, Article
from .ce_source import fetch_ce_articles, CE_LIST_URLS
from .multi_source import fetch_multi_source_articles

__all__ = [
    "fetch_previous_day_articles",
    "fetch_ce_articles",
    "fetch_multi_source_articles",
    "Article",
    "CE_LIST_URLS",
]