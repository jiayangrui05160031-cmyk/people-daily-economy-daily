"""list_parser — 列表页解析 ==============================================
从 index{N}.html 中提取所有候选文章链接(title + url + article_id)。
"""

from __future__ import annotations

import re
from typing import Dict, List
from urllib.parse import urljoin

from bs4 import BeautifulSoup


# 文章 URL 模板正则(用于 article_id 提取):/n1/YYYY/MMDD/cNNNN-NNNNNNNN.html
ARTICLE_URL_RE = re.compile(r"/n1/\d{4}/\d{4}/c\d+-\d+\.html")


def parse_list_html(html: str, base_url: str = "http://finance.people.com.cn") -> List[Dict[str, str]]:
    """解析列表页,返回 [{title, url, article_id, date_token}, ...]。

    容错策略:多组 CSS 选择器依次尝试,任意命中即可。
    """
    soup = BeautifulSoup(html, "lxml")

    # 多组选择器按优先级尝试(不同年份栏目可能用不同模板)
    selectors = [
        "div.box01 ul li a",
        ".hd_news ul li a",
        ".news_list li a",
        "ul.list_14 li a",
        "ul li a",  # 兜底
    ]

    candidates: List[Dict[str, str]] = []
    seen_ids: set[str] = set()

    for selector in selectors:
        anchors = soup.select(selector)
        if not anchors:
            continue
        for a in anchors:
            href = a.get("href", "").strip()
            if not href:
                continue
            # 绝对化
            if not href.startswith("http"):
                href = urljoin(base_url, href)

            # 仅保留匹配文章 URL 模板的链接
            if not ARTICLE_URL_RE.search(href):
                continue

            # 提取 article_id(如 40295339)
            match = re.search(r"c\d+-(\d+)\.html", href)
            if not match:
                continue
            article_id = match.group(1)
            if article_id in seen_ids:
                continue
            seen_ids.add(article_id)

            # 提取日期 token(/n1/2024/0809/)
            date_match = re.search(r"/n1/(\d{4})/(\d{4})/", href)
            date_token = f"/n1/{date_match.group(1)}/{date_match.group(2)}/" if date_match else ""

            title = a.get_text(strip=True) or a.get("title", "").strip()
            if not title or len(title) < 4:
                continue

            candidates.append({
                "title": title,
                "url": href,
                "article_id": article_id,
                "date_token": date_token,
            })

        # 命中一组就停,避免重复
        if candidates:
            break

    return candidates


def filter_by_date_token(articles: List[Dict[str, str]], date_token: str) -> List[Dict[str, str]]:
    """按 URL 路径日期段过滤前一天的文章。"""
    if not date_token:
        return articles
    return [a for a in articles if a.get("date_token") == date_token]