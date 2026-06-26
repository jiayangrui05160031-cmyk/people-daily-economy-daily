"""retrieval.vector - Embedding 向量检索 (wraps embeddings.EmbeddingStore).

embed_rag.py 整体删除；它的功能（向量化 + 余弦检索）已经被
embeddings.py 完整实现（VectorIndex + EmbeddingStore + SQLite 持久化）。
这里只做一个薄包装：把 embeddings 的 SearchHit 转成统一的 Hit。

embeddings.py 不动；它继续负责 router / index / cache 等底层。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger
from .base import Hit

logger = get_logger("retrieval.vector")


class VectorRetriever:
    """Embedding-based semantic search.

    Wraps EmbeddingStore from src.analysis.embeddings and adapts its
    SearchHit to the unified Hit dataclass.
    """

    name = "vector"

    def __init__(self, store=None, prefer: str = "auto"):
        if store is None:
            from src.analysis.embeddings import get_store
            self.store = get_store()
        else:
            self.store = store

    def add(self, doc_id: str, text: str,
            metadata: Optional[Dict[str, Any]] = None) -> None:
        self.store.add(doc_id, text, metadata=metadata or {})

    def add_many(self, docs: List) -> int:
        return self.store.add_many(docs)

    def backend_info(self) -> Dict[str, Any]:
        return self.store.backend_info()

    def search(self, query: str, top_k: int = 5,
               min_score: float = 0.0, **kwargs) -> List[Hit]:
        try:
            sh = self.store.search(query, top_k=top_k, min_score=min_score)
        except Exception as e:
            logger.debug(f"vector search 失败: {e}")
            return []
        return [
            Hit(
                id=h.id,
                text=h.text,
                score=h.score,
                source=self.name,
                metadata=dict(h.metadata or {}),
            )
            for h in sh
        ]
