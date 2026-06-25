"""scraper.semantic_dedup - 跨文章语义去重"""
from __future__ import annotations

import hashlib
from typing import List, Set, Tuple

from src.analysis.similarity import tfidf_cosine
from src.scraper.pipeline import Article
from src.utils.logger import get_logger

logger = get_logger("scraper.semantic_dedup")


def title_hash(article):
    t = (article.title or "").strip().lower()
    t = "".join(c for c in t if c.isalnum() or "\u4e00" <= c <= "\u9fff")
    return hashlib.md5(t.encode("utf-8")).hexdigest()


def dedup_by_title(articles):
    seen = set()
    out = []
    for a in articles:
        h = title_hash(a)
        if h in seen:
            continue
        seen.add(h)
        out.append(a)
    return out


def dedup_by_tfidf(articles, threshold=0.85):
    if not articles:
        return [], 0
    texts = [a.content_text or a.title for a in articles]
    M = tfidf_cosine(texts)
    n = len(articles)
    drop = [False] * n
    dup_count = 0
    for i in range(n):
        if drop[i]:
            continue
        for j in range(i + 1, n):
            if drop[j]:
                continue
            if M[i, j] >= threshold:
                drop[j] = True
                dup_count += 1
    out = [a for a, d in zip(articles, drop) if not d]
    logger.info(f"dedup_by_tfidf: {n} -> {len(out)} (removed {dup_count})")
    return out, dup_count


def full_dedup(articles, router=None, threshold=0.85):
    step1 = dedup_by_title(articles)
    step2, _ = dedup_by_tfidf(step1, threshold=threshold)
    return step2


if __name__ == "__main__":
    a1 = Article(title="央行宣布降准1万亿",
                 content_text="央行决定下调存款准备金率0.5个百分点释放长期资金一亿元支持实体经济",
                 url="u1")
    a2 = Article(title="央行宣布降准1万亿支持实体经济",
                 content_text="央行决定下调存款准备金率0.5个百分点释放长期资金一亿元支持实体经济",
                 url="u2")
    a3 = Article(title="新能源汽车销量增长",
                 content_text="中国新能源汽车销量持续增长,渗透率突破百分之五十",
                 url="u3")
    print("dedup_by_title:", len(dedup_by_title([a1, a2, a3])))
    print("dedup_by_tfidf:", len(dedup_by_tfidf([a1, a2, a3])[0]))
    print("full_dedup:", len(full_dedup([a1, a2, a3])))
