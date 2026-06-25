"""kg.relations - 关系抽取 (共现)"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import List, Tuple


def extract_relations(articles, entities):
    if not entities:
        return []
    ent_set = {v: t for t, v, _ in entities}
    cooccur = defaultdict(int)
    for art in articles:
        text = art.content_text or ""
        present = [v for v in ent_set if v in text]
        for a, b in combinations(sorted(set(present)), 2):
            cooccur[(a, b)] += 1
    rels = []
    for (a, b), w in cooccur.items():
        if w < 2:
            continue
        rels.append((a, b, ent_set[a], ent_set[b], w))
    rels.sort(key=lambda x: x[4], reverse=True)
    return rels


if __name__ == "__main__":
    from src.scraper.pipeline import Article
    arts = [
        Article(title="央行降准1万亿", content=["央行决定降准释放流动性支持实体经济"],
                content_text="央行决定降准释放流动性支持实体经济,工信部同日发布新型储能行动方案"),
        Article(title="工信部储能方案", content=["工信部发布新型储能行动方案"],
                content_text="工信部发布新型储能行动方案,提到央行降准支持新能源"),
    ]
    ents = [("机构", "央行", 2), ("机构", "工信部", 2), ("政策", "降准", 2), ("政策", "储能", 1)]
    print(extract_relations(arts, ents))
