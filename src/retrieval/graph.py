"""retrieval.graph - 知识图谱社区检索 (wraps graph_rag.ask_global).

graph_rag.ask_global 输出 dict, 我们 adapter 成 Hit 列表。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger
from .base import Hit

logger = get_logger("retrieval.graph")


class GraphRetriever:
    """Knowledge-graph community-based retrieval."""

    name = "graph"

    def __init__(self, router=None):
        self.router = router

    def search(self, query: str, top_k: int = 5,
               articles: Optional[List] = None, **kwargs) -> List[Hit]:
        if not articles:
            return []
        try:
            from src.analysis.graph_rag import ask_global
            rep = ask_global(
                query, articles=articles, router=self.router,
                top_communities=top_k, use_llm=False,
            )
        except Exception as e:
            logger.debug(f"graph search 失败: {e}")
            return []
        hits: List[Hit] = []
        for src in rep.get("sources", [])[:top_k]:
            hits.append(Hit(
                id=f"graph:community:{src.get('community_id', -1)}",
                text=src.get("summary", ""),
                score=0.5 + 0.1 * (top_k - len(hits)),  # rough ordering
                source=self.name,
                metadata={
                    "label": src.get("label", ""),
                    "node_count": src.get("node_count", 0),
                    "edge_count": src.get("edge_count", 0),
                },
            ))
        return hits
