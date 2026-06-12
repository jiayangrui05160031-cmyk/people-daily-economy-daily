"""archiver — 原始数据 JSON 归档 ==============================================
每日抓取的 Article 列表写入 data/raw/{date}.json,便于复现。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from src.config import DATA_RAW_DIR
from src.scraper.pipeline import Article
from src.utils.logger import get_logger

logger = get_logger("report.archiver")


def archive_raw(articles: List[Article], report_date: str) -> Path:
    """把 Article 列表序列化为 JSON 写入 data/raw/{date}.json。

    Args:
        articles: 文章列表
        report_date: YYYY-MM-DD

    Returns:
        写入的文件路径
    """
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_RAW_DIR / f"{report_date}.json"

    payload = {
        "date": report_date,
        "count": len(articles),
        "articles": [a.to_dict() for a in articles],
    }

    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"原始数据已归档: {output_path}")
    return output_path


def load_raw(report_date: str) -> List[Article]:
    """从 data/raw/{date}.json 反序列化为 Article 列表。"""
    path = DATA_RAW_DIR / f"{report_date}.json"
    if not path.exists():
        raise FileNotFoundError(f"未找到归档: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Article.from_dict(a) for a in payload.get("articles", [])]


if __name__ == "__main__":
    sample = [Article(title="测试", url="http://x", source="测试")]
    p = archive_raw(sample, "2026-06-11")
    loaded = load_raw("2026-06-11")
    print(f"归档: {p},加载: {len(loaded)} 篇")