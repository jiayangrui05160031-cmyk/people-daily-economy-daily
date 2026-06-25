"""scraper.gov_pbc_source - 央行等官方政策发布源 (示例,默认关闭)

央行新闻发布地址: http://www.pbc.gov.cn/zhengwugongkai/4081330/4081344/index.html
这里是占位实现,需要适配其 JS 渲染结构。
"""
from __future__ import annotations

from typing import List, Optional

from src.config import ENABLE_GOV_PBOC
from src.scraper.fetcher import Fetcher
from src.scraper.pipeline import Article
from src.utils.logger import get_logger

logger = get_logger("scraper.gov_pbc")


def fetch_pbc_articles(target_date: str, lookback_days: int = 2,
                       fetcher: Optional[Fetcher] = None) -> List[Article]:
    if not ENABLE_GOV_PBOC:
        logger.info("[gov_pbc] disabled by config, skip")
        return []
    fetcher = fetcher or Fetcher()
    logger.warning("[gov_pbc] not implemented, returning empty list")
    fetcher.close()
    return []


if __name__ == "__main__":
    arts = fetch_pbc_articles("2026-06-12")
    print(f"pbc: {len(arts)}")
