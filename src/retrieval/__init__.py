"""retrieval/ - 统一 RAG 检索层 (v9 重构)

取代之前 5 个 RAG 模块的碎片化实现:
  - embed_rag.py        (删除, 与 embeddings.py 重复)
  - rag_history.py      (保留 TF-IDF 部分, 由 LexicalRetriever 包装)
  - embeddings.py       (保留, 由 VectorRetriever 包装)
  - graph_rag.py        (保留, 由 GraphRetriever 包装)
  - qa_assistant.py     (改成编排器, 选/并联 retriever)

设计:
  Retriever(Protocol)
    .search(query, top_k) -> list[Hit]

  三个具体实现:
    - LexicalRetriever   (TF-IDF 词袋; 跨期主题; 中文 jieba)
    - VectorRetriever    (sentence embedding; 语义检索; 持久化)
    - GraphRetriever     (知识图谱社区发现; 关系查询)

  qa_assistant 变成编排器: 按 query 类型选 retriever, 或并联后融合排序。
  server.py 的 /v6/embed/* 路由改指 VectorRetriever。

收益:
  - embed_rag.py 整 285 行重复删除
  - 新增 retriever 类型只需实现一个 protocol
  - 统一返回 list[Hit] 替代五种不同 dataclass
"""
from .base import Retriever, Hit, CitationLike, hit_to_citation
from .lexical import LexicalRetriever
from .vector import VectorRetriever
from .graph import GraphRetriever

__all__ = [
    "Retriever", "Hit", "CitationLike", "hit_to_citation",
    "LexicalRetriever", "VectorRetriever", "GraphRetriever",
]
