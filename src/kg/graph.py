"""kg.graph - 知识图谱构建"""
from __future__ import annotations

import math
from typing import Dict

import networkx as nx

from src.kg.entities import extract
from src.kg.relations import extract_relations
from src.utils.logger import get_logger

logger = get_logger("kg.graph")


def build(articles, router=None, min_node_freq=1):
    G = nx.Graph()
    entities = extract(articles, router=router)
    if not entities:
        return G

    for t, v, freq in entities:
        if freq < min_node_freq:
            continue
        G.add_node(v, type=t, freq=freq, size=math.log2(freq + 1) * 30)

    rels = extract_relations(articles, entities)
    for a, b, ta, tb, w in rels:
        if a not in G.nodes or b not in G.nodes:
            continue
        if G.has_edge(a, b):
            G[a][b]["weight"] += w
        else:
            G.add_edge(a, b, weight=w, edge_type="cooccur")

    try:
        pr = nx.pagerank(G, alpha=0.85, weight="weight")
        for n, p in pr.items():
            G.nodes[n]["pagerank"] = p
    except Exception as e:
        logger.warning(f"pagerank failed: {e}")
        for n in G.nodes:
            G.nodes[n]["pagerank"] = 0.0

    logger.info(f"KG built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def summarize(G, top_k=8):
    if G.number_of_nodes() == 0:
        return {"nodes": [], "edges": [], "stats": {"nodes": 0, "edges": 0}}

    pr_sorted = sorted(G.nodes(data=True), key=lambda x: x[1].get("pagerank", 0), reverse=True)
    top_nodes = [{"id": n, "type": d.get("type", "其他"), "freq": d.get("freq", 0),
                  "pagerank": round(d.get("pagerank", 0), 4)}
                 for n, d in pr_sorted[:top_k]]

    top_edges = sorted(G.edges(data=True), key=lambda x: x[2].get("weight", 0), reverse=True)
    edge_list = [{"source": a, "target": b, "weight": d.get("weight", 0)}
                 for a, b, d in top_edges[:top_k * 2]]

    return {
        "nodes": top_nodes,
        "edges": edge_list,
        "stats": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()},
    }


if __name__ == "__main__":
    from src.scraper.pipeline import Article
    arts = [
        Article(title="央行降准1万亿", content=["央行决定降准支持实体经济"],
                content_text="央行决定降准释放长期资金支持实体经济"),
        Article(title="工信部新型储能方案", content=["工信部发布新型储能行动方案"],
                content_text="工信部发布新型储能行动方案,工信部强调高质量发展"),
        Article(title="上海市扩大消费", content=["上海市发布扩大消费政策"],
                content_text="上海市发布扩大消费若干措施,新能源汽车以旧换新"),
        Article(title="央行支持新能源", content=["央行降准利好新能源"],
                content_text="央行降准利好新能源板块,工信部协同出台政策"),
    ]
    G = build(arts)
    summary = summarize(G)
    print("nodes:", summary["nodes"])
    print("edges:", summary["edges"])
