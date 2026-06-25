"""kg.viz - 知识图谱可视化 (matplotlib)"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import font_manager

from src.config import CHINESE_FONT_PATH, IMAGE_DIR
from src.utils.logger import get_logger

logger = get_logger("kg.viz")


def _setup_chinese_font():
    try:
        if Path(CHINESE_FONT_PATH).exists():
            font_manager.fontManager.addfont(CHINESE_FONT_PATH)
            prop = font_manager.FontProperties(fname=CHINESE_FONT_PATH)
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["axes.unicode_minus"] = False
    except Exception as e:
        logger.warning(f"font setup failed: {e}")


def render(G, output_name="knowledge_graph.png"):
    if G.number_of_nodes() == 0:
        logger.info("empty graph, skip render")
        return None

    _setup_chinese_font()
    fig, ax = plt.subplots(figsize=(12, 9))

    try:
        pos = nx.spring_layout(G, k=1.5 / math.sqrt(max(G.number_of_nodes(), 1)),
                                seed=42, weight="weight")
    except Exception:
        pos = nx.spring_layout(G, seed=42)

    sizes = [G.nodes[n].get("size", 200) + 100 for n in G.nodes]
    pr = [G.nodes[n].get("pagerank", 0) for n in G.nodes]
    if max(pr) > 0:
        node_colors = [p / max(pr) for p in pr]
    else:
        node_colors = [0.5] * len(pr)

    edges = G.edges()
    weights = [G[u][v].get("weight", 1) for u, v in edges]
    max_w = max(weights) if weights else 1
    edge_widths = [0.5 + 2.0 * (w / max_w) for w in weights]

    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=node_colors,
                            cmap=plt.cm.YlOrRd, alpha=0.85, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.4,
                            edge_color="gray", ax=ax)
    labels = {n: n for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_color="#222", ax=ax)

    edge_labels = {(u, v): G[u][v].get("weight", 1) for u, v in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)

    date_part = output_name.replace("knowledge_graph_", "").replace(".png", "")
    ax.set_title(f"Knowledge Graph of Entities · {date_part}", fontsize=14)
    ax.axis("off")

    out_path = IMAGE_DIR / output_name
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"KG rendered to {out_path}")
    return out_path


if __name__ == "__main__":
    from src.kg.graph import build
    from src.scraper.pipeline import Article
    arts = [
        Article(title="央行降准1万亿", content=["央行降准支持实体经济"],
                content_text="央行决定降准释放长期资金支持实体经济"),
        Article(title="工信部新型储能方案", content=["工信部发布新型储能行动方案"],
                content_text="工信部发布新型储能行动方案强调高质量发展"),
        Article(title="上海市扩大消费", content=["上海市发布扩大消费政策"],
                content_text="上海市发布扩大消费若干措施,新能源汽车以旧换新"),
        Article(title="央行支持新能源", content=["央行降准利好新能源"],
                content_text="央行降准利好新能源板块,工信部协同出台政策"),
    ]
    G = build(arts)
    p = render(G, "knowledge_graph_demo.png")
    print("rendered to:", p)
