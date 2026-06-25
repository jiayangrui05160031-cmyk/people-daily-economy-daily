"""scraper.xinhua_source - 新华网财经 (示例,默认关闭)

实际接入需要适配新华网的列表页/详情页结构。
这里给出一个最小可用模板,供二次开发参考。
"""
from __future__ import annotations

from typing import List, Optional

from src.config import ENABLE_XINHUA
from src.scraper.fetcher import Fetcher
from src.scraper.pipeline import Article
from src.utils.logger import get_logger

logger = get_logger("scraper.xinhua")


def fetch_xinhua_articles(target_date: str, lookback_days: int = 2,
                           fetcher: Optional[Fetcher] = None) -> List[Article]:
    if not ENABLE_XINHUA:
        logger.info("[xinhua] disabled by config, skip")
        return []
    fetcher = fetcher or Fetcher()
    logger.warning("[xinhua] not implemented, returning empty list")
    fetcher.close()
    return []


if __name__ == "__main__":
    arts = fetch_xinhua_articles("2026-06-12")
    print(f"xinhua: {len(arts)}")
