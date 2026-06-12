"""pipeline — 爬虫主流程 ==============================================
串起:列表页 → URL 解析 → 日期过滤 → 去重 → 详情页 → 解析 → Article 列表。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from src.config import LIST_PAGES, LIST_URL_TEMPLATE
from src.scraper.article_parser import parse_article_html
from src.scraper.fetcher import Fetcher
from src.scraper.list_parser import parse_list_html, filter_by_date_token
from src.utils.date_utils import resolve_target_date, path_date_token
from src.utils.logger import get_logger

logger = get_logger("scraper.pipeline")


@dataclass
class Article:
    """统一文章数据结构,贯穿爬虫→NLP→AI→报告。"""

    title: str = ""
    url: str = ""
    article_id: str = ""
    publish_time: str = ""
    source: str = ""
    reporter: str = ""
    editor: str = ""
    channel: str = ""
    content: List[str] = field(default_factory=list)
    content_text: str = ""
    word_count: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "Article":
        return cls(**{k: d.get(k, getattr(cls(), k)) for k in cls.__dataclass_fields__.keys()})


def fetch_previous_day_articles(
    target_date: Optional[str] = None,
    list_pages: int = 1,
    fetcher: Optional[Fetcher] = None,
    skip_details: bool = False,
    lookback_days: int = 7,
) -> List[Article]:
    """抓取指定日期(默认昨天)及向前 N 天的经济新闻。

    Args:
        target_date: YYYY-MM-DD,留空默认昨天
        list_pages: 列表页抓取页数(默认 1 页)。
            实测发现:人民网财经主列表 index1-5 内容完全相同(都是最近 ~11 篇),
            抓多页只会浪费限速时间,默认 1 页。
        fetcher: 自定义 fetcher(便于测试),默认新建
        skip_details: True 时只返回列表页解析结果(不抓详情)
        lookback_days: 向前回溯的天数(含 target_date 自身)。
            实测发现:列表页单日只展示约 11 篇(其中当天 9 篇 + 前几天 2 篇),
            要拿到足够 NLP/AI 分析的内容,默认向前回溯 7 天
            (即报告 [target_date, target_date-1, ..., target_date-6] 七天合计 ~50 篇),
            同时保留"近一周经济热点"语义。

    Returns:
        Article 列表(可能为空,如节假日)
    """
    target = resolve_target_date(target_date or "")
    fetcher = fetcher or Fetcher()

    logger.info(f"目标日期: {target} (向前回溯 {lookback_days} 天)")

    # --- 第一轮:列表页 ---
    all_candidates: List[Dict] = []
    for n in range(1, list_pages + 1):
        list_url = LIST_URL_TEMPLATE.format(n=n)
        try:
            html = fetcher.get(list_url)
            candidates = parse_list_html(html)
            all_candidates.extend(candidates)
            logger.info(f"[列表 {n}/{list_pages}] {list_url} → 提取 {len(candidates)} 个链接")
        except Exception as e:
            logger.warning(f"列表 {n} 抓取失败: {e}")
            continue

    # 去重(article_id)
    seen: set[str] = set()
    unique = []
    for c in all_candidates:
        if c["article_id"] in seen:
            continue
        seen.add(c["article_id"])
        unique.append(c)
    logger.info(f"列表去重后: {len(unique)} 篇")

    # 按日期过滤:target_date 向前 lookback_days 天
    from datetime import timedelta
    valid_tokens = set()
    for offset in range(lookback_days):
        d = target - timedelta(days=offset)
        valid_tokens.add(path_date_token(d))
    filtered = [a for a in unique if a.get("date_token") in valid_tokens]
    logger.info(
        f"过滤 [{', '.join(sorted(valid_tokens, reverse=True))}] 后: {len(filtered)} 篇"
    )

    if skip_details or not filtered:
        return [Article(**{k: c.get(k, "") for k in [
            "title", "url", "article_id"
        ]}) for c in filtered]

    # --- 第二轮:详情页 ---
    articles: List[Article] = []
    for i, c in enumerate(filtered, 1):
        try:
            html = fetcher.get(c["url"])
            parsed = parse_article_html(html, source_url=c["url"])
            # 合并列表页元信息
            article = Article(
                title=parsed.get("title") or c["title"],
                url=c["url"],
                article_id=c["article_id"],
                publish_time=parsed.get("publish_time", ""),
                source=parsed.get("source", ""),
                reporter=parsed.get("reporter", ""),
                editor=parsed.get("editor", ""),
                channel=parsed.get("channel", ""),
                content=parsed.get("content", []),
                content_text=parsed.get("content_text", ""),
                word_count=parsed.get("word_count", 0),
            )
            articles.append(article)
            logger.info(f"[详情 {i}/{len(filtered)}] OK {article.title[:30]} ({article.word_count}字)")
        except Exception as e:
            logger.warning(f"详情失败 {c['url']}: {e}")
            # 仍保留列表页元信息,正文留空
            articles.append(Article(
                title=c["title"], url=c["url"], article_id=c["article_id"]
            ))

    fetcher.close()
    logger.info(f"最终采集: {len(articles)} 篇")
    return articles


if __name__ == "__main__":
    # 开发调试入口
    arts = fetch_previous_day_articles()
    print(f"\n[OK] 共 {len(arts)} 篇")
    for a in arts[:3]:
        print(f"  - {a.title} ({a.publish_time}) [{a.channel}]")