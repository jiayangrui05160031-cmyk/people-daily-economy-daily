"""verify_selectors.py ==============================================
开发期辅助脚本:实爬验证 finance.people.com.cn 的 CSS 选择器是否命中。
不修改任何文件,只打印诊断信息。

用法:
    python scripts/verify_selectors.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# 把项目根加入 import 路径
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import LIST_URL_TEMPLATE, LIST_PAGES, USER_AGENT, REFERER
from src.scraper.fetcher import Fetcher
from src.scraper.list_parser import parse_list_html
from src.scraper.article_parser import parse_article_html


def banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def main() -> int:
    banner("STEP 1: 实测列表页 (index1.html)")
    fetcher = Fetcher()
    try:
        html = fetcher.get(LIST_URL_TEMPLATE.format(n=1))
        print(f"[OK] HTTP 200, length={len(html)}")
    except Exception as e:
        print(f"[X] 列表页抓取失败: {e}")
        return 1

    banner("STEP 2: 解析列表,统计链接数")
    articles = parse_list_html(html, base_url="http://finance.people.com.cn")
    print(f"[OK] 提取到 {len(articles)} 个候选文章链接")
    if articles:
        print("  示例:", articles[0])

    banner("STEP 3: 实测详情页(取列表中第一条)")
    if articles:
        sample = articles[0]
        try:
            article_html = fetcher.get(sample["url"])
            article = parse_article_html(article_html, source_url=sample["url"])
            print(f"[OK] 标题: {article.get('title', '')[:50]}")
            print(f"[OK] 正文段落数: {len(article.get('content', []))}")
            print(f"[OK] 来源: {article.get('source', '')}")
            print(f"[OK] 时间: {article.get('publish_time', '')}")
            print(f"[OK] 栏目: {article.get('channel', '')}")
        except Exception as e:
            print(f"[X] 详情页失败: {e}")
            return 1

    banner("STEP 4: 翻页测试(快速只验 status code)")
    for n in range(1, LIST_PAGES + 1):
        url = LIST_URL_TEMPLATE.format(n=n)
        try:
            fetcher.get(url)
            print(f"[OK] index{n}.html — 200")
        except Exception as e:
            print(f"[X] index{n}.html — {e}")

    print("\n[OK] 所有验证完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())