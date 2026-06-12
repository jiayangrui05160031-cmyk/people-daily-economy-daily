"""ce_source — 中国经济网(ce.cn)爬虫源 ==============================================
独立的数据源适配器,与 people_source 并列。

URL 模式:
  列表页: http://finance.ce.cn/  http://www.ce.cn/cysc/  http://www.ce.cn/cysc/zljd/
  文章页: /栏目/路径/YYYYMM/tYYYYMMDD_NNNNNNN.shtml

实测:
  finance.ce.cn 首页: 25 候选(19 当天 + 6 历史)
  cysc 首页: 32 候选,跨 26 个日期
  cysc/zljd 首页: 37 候选,跨 27 个日期

详情页结构(实测):
  标题: h1 (备选 h2, og:title)
  正文: .content 或 #article (39 段示例)
  时间: .time (格式: 2026-06-12 07:24)
  来源: meta[name=NewsArticleSource] (备选 .source)
  作者: meta[name=NewsArticleAuthor]
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from bs4 import BeautifulSoup, Tag

from src.config import REQUEST_INTERVAL_SEC
from src.scraper.pipeline import Article
from src.scraper.fetcher import Fetcher
from src.utils.logger import get_logger

logger = get_logger("scraper.ce")


# 列表页 URL 列表(财经频道各子栏目)
# 注:子栏目页(rolling/bank12/stock)需要翻页参数,解析会返回 0,
#    因此仅保留首页作为稳定数据源;若需扩展,后续可加 page=N 处理
CE_LIST_URLS = [
    "http://finance.ce.cn/",
]

# 文章 URL 正则: /tYYYYMMDD_NNNNNNN.shtml
CE_ARTICLE_URL_RE = re.compile(r"/t(\d{4})(\d{2})(\d{2})_(\d+)\.shtml")


def parse_ce_list_html(html: str, base_url: str = "http://finance.ce.cn") -> List[Dict[str, str]]:
    """解析 ce.cn 列表页,返回 [{title, url, article_id, date_token}, ...]。"""
    soup = BeautifulSoup(html, "lxml")

    candidates: List[Dict[str, str]] = []
    seen_ids: set[str] = set()

    for a in soup.select("a[href]"):
        href = a.get("href", "").strip()
        if not href or "ce.cn" not in href:
            continue

        m = CE_ARTICLE_URL_RE.search(href)
        if not m:
            continue

        year, month, day, article_id = m.groups()
        if article_id in seen_ids:
            continue
        seen_ids.add(article_id)

        date_token = f"/t{year}{month}{day}_"
        title = a.get_text(strip=True)
        # ce.cn 列表页链接的 title 文本常为简短子标题(如"详细/精细化"),
        # 因此放宽到 2 字符下限,信任 URL 正则即可
        if not title or len(title) < 2 or len(title) > 80:
            continue
        # 跳过明显是栏目名/导航的
        if any(kw in title for kw in ["更多", "查看更多", "下一页", "首页", "登录", "注册", "返回"]):
            continue

        candidates.append({
            "title": title,
            "url": href,
            "article_id": article_id,
            "date_token": date_token,
            "publish_date": f"{year}-{month}-{day}",
        })

    return candidates


def parse_ce_article_html(html: str, source_url: str = "") -> Dict:
    """解析 ce.cn 详情页,返回结构化字段。

    Returns:
        dict with title/content/publish_time/source/...
    """
    soup = BeautifulSoup(html, "lxml")

    # --- 标题 ---
    title = ""
    h1 = soup.select_one("h1")
    if h1:
        title = h1.get_text(strip=True)
    if not title:
        h2 = soup.select_one("h2")
        if h2:
            title = h2.get_text(strip=True)
    if not title:
        og = soup.select_one('meta[property="og:title"]')
        if og and og.get("content"):
            title = og["content"].strip()

    # --- 正文 ---
    content_node = soup.select_one("#article") or soup.select_one(".content") or soup.select_one("#articleText")
    paragraphs: List[str] = []
    if content_node:
        # 剔除干扰
        for sel in ["script", "style", "iframe", ".share", "#share", ".ad", ".adv", ".related", ".recommend", ".edit", ".editor"]:
            for n in content_node.select(sel):
                try:
                    n.decompose()
                except Exception:
                    pass
        for p in content_node.find_all("p"):
            txt = p.get_text(separator="", strip=True)
            if not txt or len(txt) < 4:
                continue
            if not re.search(r"[一-鿿A-Za-z0-9]", txt):
                continue
            paragraphs.append(txt)

    # --- 时间 ---
    publish_time = ""
    time_node = soup.select_one(".time")
    if time_node:
        txt = time_node.get_text(strip=True)
        # 提取 YYYY-MM-DD HH:MM
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2})", txt)
        if m:
            publish_time = m.group(0)
    # 备选 meta
    if not publish_time:
        meta_day = soup.select_one('meta[name="NewsArticlePubDay"]')
        if meta_day and meta_day.get("content"):
            publish_time = meta_day["content"].strip()

    # --- 来源 ---
    source = ""
    src_node = soup.select_one(".source") or soup.select_one("#source")
    if src_node:
        src_text = src_node.get_text(strip=True)
        m = re.search(r"来源[：:]\s*(\S+)", src_text)
        if m:
            source = m.group(1)
    if not source:
        meta_src = soup.select_one('meta[name="NewsArticleSource"]')
        if meta_src and meta_src.get("content"):
            source = meta_src["content"].strip()

    # --- 作者/编辑 ---
    author = ""
    editor = ""
    meta_author = soup.select_one('meta[name="NewsArticleAuthor"]')
    if meta_author and meta_author.get("content"):
        author = meta_author["content"].strip()

    # --- 栏目(从 URL 推断) ---
    # /栏目/路径/... 模式
    channel = ""
    m = re.search(r"ce\.cn/([^/]+)/", source_url)
    if m:
        channel_map = {
            "finance": "财经",
            "cysc": "产业经济",
            "bank12": "银行",
            "stock": "证券",
            "home": "金融",
            "zljd": "质量监督",
        }
        channel = channel_map.get(m.group(1), m.group(1))

    content_text = "\n".join(paragraphs)
    return {
        "title": title,
        "content": paragraphs,
        "content_text": content_text,
        "publish_time": publish_time,
        "source": source,
        "reporter": author,
        "editor": editor,
        "channel": channel,
        "url": source_url,
        "word_count": sum(len(p) for p in paragraphs),
    }


def fetch_ce_articles(
    target_date: str,
    lookback_days: int = 7,
    fetcher: Optional[Fetcher] = None,
    list_urls: Optional[List[str]] = None,
) -> List[Article]:
    """抓取 ce.cn 指定日期及向前 N 天的经济新闻。

    Args:
        target_date: YYYY-MM-DD
        lookback_days: 回溯天数
        fetcher: 自定义 fetcher
        list_urls: 自定义列表页 URL 列表

    Returns:
        Article 列表
    """
    from datetime import datetime, timedelta
    target = datetime.strptime(target_date, "%Y-%m-%d").date()
    fetcher = fetcher or Fetcher()
    list_urls = list_urls or CE_LIST_URLS

    # 计算有效日期集合
    valid_dates: set[str] = set()
    valid_path_tokens: set[str] = set()
    for offset in range(lookback_days):
        d = target - timedelta(days=offset)
        valid_dates.add(d.strftime("%Y-%m-%d"))
        # ce.cn 路径格式: /栏目/YYYYMM/tYYYYMMDD_NNNNNNN.shtml
        valid_path_tokens.add(d.strftime("/t%Y%m%d_"))

    # --- 第一轮:列表页 ---
    all_candidates: List[Dict] = []
    for url in list_urls:
        try:
            html = fetcher.get(url)
            cands = parse_ce_list_html(html, base_url=url)
            all_candidates.extend(cands)
            logger.info(f"[ce.cn] {url} → {len(cands)} 候选")
        except Exception as e:
            logger.warning(f"[ce.cn] {url} 失败: {e}")

    # 去重
    seen: set[str] = set()
    unique = []
    for c in all_candidates:
        if c["article_id"] in seen:
            continue
        seen.add(c["article_id"])
        unique.append(c)
    logger.info(f"[ce.cn] 去重后: {len(unique)} 篇")

    # 日期过滤
    filtered = [a for a in unique if a.get("publish_date") in valid_dates]
    logger.info(f"[ce.cn] {target_date} 向前 {lookback_days} 天: {len(filtered)} 篇")

    # --- 第二轮:详情页 ---
    articles: List[Article] = []
    for i, c in enumerate(filtered, 1):
        try:
            html = fetcher.get(c["url"])
            parsed = parse_ce_article_html(html, source_url=c["url"])
            article = Article(
                title=parsed.get("title") or c["title"],
                url=c["url"],
                article_id=f"ce_{c['article_id']}",
                publish_time=parsed.get("publish_time", ""),
                source=parsed.get("source", "") or "中国经济网",
                reporter=parsed.get("reporter", ""),
                editor=parsed.get("editor", ""),
                channel=parsed.get("channel", ""),
                content=parsed.get("content", []),
                content_text=parsed.get("content_text", ""),
                word_count=parsed.get("word_count", 0),
            )
            articles.append(article)
            logger.info(f"[ce.cn 详情 {i}/{len(filtered)}] OK {article.title[:30]} ({article.word_count}字)")
        except Exception as e:
            logger.warning(f"[ce.cn 详情失败] {c['url']}: {e}")
            articles.append(Article(
                title=c["title"], url=c["url"],
                article_id=f"ce_{c['article_id']}",
                source="中国经济网",
            ))

    fetcher.close()
    logger.info(f"[ce.cn] 最终采集: {len(articles)} 篇")
    return articles