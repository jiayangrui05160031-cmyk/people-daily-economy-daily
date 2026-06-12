"""multi_source — 多源合并调度 ==============================================
统一调度 people.com.cn 与 ce.cn 两个数据源,合并去重后返回 Article 列表。
"""

from __future__ import annotations

from typing import List, Optional

from src.scraper.ce_source import fetch_ce_articles
from src.scraper.fetcher import Fetcher
from src.scraper.pipeline import Article, fetch_previous_day_articles
from src.utils.logger import get_logger

logger = get_logger("scraper.multi_source")


def fetch_multi_source_articles(
    target_date: str,
    lookback_days: int = 2,
    fetcher: Optional[Fetcher] = None,
    enable_people: bool = True,
    enable_ce: bool = True,
) -> List[Article]:
    """从多个数据源并行抓取,合并去重。

    Args:
        target_date: YYYY-MM-DD
        lookback_days: 回溯天数
        fetcher: 共用 fetcher(便于测试)
        enable_people: 是否启用人民网源
        enable_ce: 是否启用中国经济网源

    Returns:
        合并去重后的 Article 列表(按时间倒序)
    """
    fetcher = fetcher or Fetcher()
    all_articles: List[Article] = []
    seen_ids: set[str] = set()

    # --- 源 1:人民网(people.com.cn) ---
    if enable_people:
        try:
            logger.info("[multi] 启动人民网源 ...")
            people_arts = fetch_previous_day_articles(
                target_date=target_date,
                lookback_days=lookback_days,
                fetcher=fetcher,
            )
            for art in people_arts:
                if art.article_id not in seen_ids:
                    seen_ids.add(art.article_id)
                    art.source = art.source or "人民网"
                    all_articles.append(art)
            logger.info(f"[multi] 人民网贡献 {len(people_arts)} 篇,累计 {len(all_articles)} 篇")
        except Exception as e:
            logger.warning(f"[multi] 人民网源失败: {e}")

    # --- 源 2:中国经济网(ce.cn) ---
    if enable_ce:
        try:
            logger.info("[multi] 启动中国经济网源 ...")
            ce_arts = fetch_ce_articles(
                target_date=target_date,
                lookback_days=lookback_days,
                fetcher=fetcher,
            )
            for art in ce_arts:
                if art.article_id not in seen_ids:
                    seen_ids.add(art.article_id)
                    all_articles.append(art)
            logger.info(f"[multi] 中国经济网贡献 {len(ce_arts)} 篇,累计 {len(all_articles)} 篇")
        except Exception as e:
            logger.warning(f"[multi] 中国经济网源失败: {e}")

    fetcher.close()

    # 按发布时间倒序(无时间的排最后)
    def sort_key(art: Article):
        return art.publish_time or "0"

    all_articles.sort(key=sort_key, reverse=True)

    logger.info(f"[multi] 合并去重后: {len(all_articles)} 篇")
    return all_articles


if __name__ == "__main__":
    arts = fetch_multi_source_articles("2026-06-12", lookback_days=2)
    print(f"\n总共: {len(arts)} 篇")
    src_count = {"people": 0, "ce": 0}
    for a in arts:
        src = "people" if "people" in a.url else "ce" if "ce.cn" in a.url else "?"
        src_count[src] = src_count.get(src, 0) + 1
        print(f"  [{a.publish_time or '?'}] [{src:6}] {a.title[:50]}")
    print(f"\n源分布: {src_count}")