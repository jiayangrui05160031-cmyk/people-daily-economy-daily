"""multi_source — 多源合并调度 (升级版)

升级点:
- 接入质量过滤 (filter_quality)
- 接入语义去重 (full_dedup)
- 可选接入 xinhua / 21cbh / gov_pbc (默认关闭,需 .env 开启)
- 记录 source_health (成功率/延迟)
"""
from __future__ import annotations

import time
from typing import List, Optional

from src.config import (
    ENABLE_21CBH,
    ENABLE_CE,
    ENABLE_GOV_PBOC,
    ENABLE_PEOPLE,
    ENABLE_XINHUA,
)
from src.scraper.ce_source import fetch_ce_articles
from src.scraper.fetcher import Fetcher
from src.scraper.pipeline import Article, fetch_previous_day_articles
from src.scraper.quality_filter import filter_quality
from src.scraper.semantic_dedup import full_dedup
from src.storage import repository as repo
from src.utils.logger import get_logger

logger = get_logger("scraper.multi_source")


def fetch_multi_source_articles(
    target_date: str,
    lookback_days: int = 2,
    fetcher: Optional[Fetcher] = None,
    enable_people: bool = ENABLE_PEOPLE,
    enable_ce: bool = ENABLE_CE,
    enable_xinhua: bool = ENABLE_XINHUA,
    enable_gov_pbc: bool = ENABLE_GOV_PBOC,
    enable_21cbh: bool = ENABLE_21CBH,
    enable_dedup: bool = True,
    enable_quality_filter: bool = True,
    quality_threshold: float = 0.20,
    router=None,
) -> List[Article]:
    """从多个数据源并行抓取,合并 -> 质量过滤 -> 语义去重 -> 返回。"""
    fetcher = fetcher or Fetcher()
    all_articles: List[Article] = []
    seen_ids: set = set()

    sources = [
        ("people", enable_people, fetch_previous_day_articles),
        ("ce", enable_ce, lambda td, lb, f: fetch_ce_articles(td, lb, fetcher=f)),
        ("xinhua", enable_xinhua, _try_xinhua),
        ("gov_pbc", enable_gov_pbc, _try_pbc),
        ("21cbh", enable_21cbh, _try_21cbh),
    ]

    for name, enabled, fn in sources:
        if not enabled:
            continue
        t0 = time.time()
        try:
            arts = fn(target_date, lookback_days, fetcher)
            latency = int((time.time() - t0) * 1000)
            added = 0
            for art in arts:
                if art.article_id not in seen_ids:
                    seen_ids.add(art.article_id)
                    if not art.source:
                        art.source = name
                    all_articles.append(art)
                    added += 1
            repo.record_source_health(target_date, name, True, latency)
            logger.info(f"[multi] {name} 贡献 {added} 篇, 用时 {latency}ms")
        except Exception as e:
            repo.record_source_health(target_date, name, False, 0)
            logger.warning(f"[multi] {name} 失败: {e}")

    fetcher.close()

    # 按发布时间倒序
    def sort_key(art):
        return art.publish_time or "0"
    all_articles.sort(key=sort_key, reverse=True)
    logger.info(f"[multi] 多源合并: {len(all_articles)} 篇")

    # 质量过滤
    if enable_quality_filter:
        before = len(all_articles)
        all_articles = filter_quality(all_articles, threshold=quality_threshold)
        logger.info(f"[multi] 质量过滤: {before} -> {len(all_articles)}")

    # 语义去重
    if enable_dedup:
        before = len(all_articles)
        all_articles = full_dedup(all_articles, router=router, threshold=0.85)
        logger.info(f"[multi] 语义去重: {before} -> {len(all_articles)}")

    return all_articles


def _try_xinhua(target_date, lookback_days, fetcher):
    try:
        from src.scraper.xinhua_source import fetch_xinhua_articles
        return fetch_xinhua_articles(target_date, lookback_days, fetcher=fetcher)
    except Exception as e:
        logger.warning(f"xinhua import/run failed: {e}")
        return []


def _try_pbc(target_date, lookback_days, fetcher):
    try:
        from src.scraper.gov_pbc_source import fetch_pbc_articles
        return fetch_pbc_articles(target_date, lookback_days, fetcher=fetcher)
    except Exception as e:
        logger.warning(f"gov_pbc import/run failed: {e}")
        return []


def _try_21cbh(target_date, lookback_days, fetcher):
    # 21 世纪经济报道 (占位)
    logger.info("[multi] 21cbh 占位,返回空")
    return []


if __name__ == "__main__":
    arts = fetch_multi_source_articles("2026-06-12", lookback_days=2)
    print(f"\n总共: {len(arts)} 篇")
    src_count = {}
    for a in arts:
        src = a.source or "?"
        src_count[src] = src_count.get(src, 0) + 1
        print(f"  [{a.publish_time or '?'}] [{src:10}] {a.title[:50]}")
    print(f"\n源分布: {src_count}")
