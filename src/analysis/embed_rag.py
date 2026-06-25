"""analysis.embed_rag - Embedding 向量 RAG (v6 前沿升级)

qa_assistant 用 TF-IDF 余弦相似度检索, 速度快但有边界:
  - 词汇鸿沟 (同义词 / 翻译 / 改写) 难以匹配
  - 短查询 vs 长文档 相似度偏低

本模块用 Embedding 向量检索 (sentence embedding + cosine):
  - 用 minimax / openai 兼容的 embedding API
  - 失败降级到本地 sentence-transformers (如装了)
  - 终极降级: TF-IDF

向量存储: 简单 in-memory numpy, 文档量 <10k 够用。
生产可换 Qdrant / Milvus / pgvector (单文件接口相同).

调用示例:
    from src.analysis.embed_rag import EmbedRAG
    rag = EmbedRAG(router=router)
    rag.add_document("doc-1", "2026-06-12 央行降准 0.5%", metadata={"date":"2026-06-12"})
    results = rag.query("降准", top_k=5)
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.logger import get_logger

logger = get_logger("analysis.embed_rag")


@dataclass
class EmbedDoc:
    id: str
    text: str
    embedding: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("embedding", None)
        return d


@dataclass
class EmbedHit:
    id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EmbedRAG:
    """Embedding 向量 RAG.

    优先级:
      1) minimax / openai 兼容的 embedding API (如果有 base_url)
      2) 本地 sentence-transformers (如安装)
      3) TF-IDF + 字符 n-gram (终极降级, 无 LLM 也跑)
    """

    # minimax / openai 都暴露 /v1/embeddings
    EMBED_PATH = "/v1/embeddings"

    def __init__(self, router=None, dim: int = 1024,
                 backend: str = "auto"):
        self.router = router
        self.dim = dim
        self.backend = backend  # "auto" | "openai" | "st" | "tfidf"
        self.docs: List[EmbedDoc] = []
        self._tfidf_vec = None
        self._tfidf_matrix = None
        self._st_model = None  # sentence-transformers 模型句柄
        self._last_query_vec: List[float] = []
        self._embed_calls = 0
        self._embed_fail = 0

    # ============================================================
    # Backend 选择
    # ============================================================
    def _resolve_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        # 1. 优先 openai 兼容
        if self.router is not None and getattr(self.router, "api_key", None):
            return "openai"
        # 2. 其次 sentence-transformers
        try:
            import sentence_transformers  # noqa: F401
            return "st"
        except ImportError:
            pass
        return "tfidf"

    # ============================================================
    # Embedding 实现 (3 个后端)
    # ============================================================
    def _embed_openai(self, text: str) -> Optional[List[float]]:
        if self.router is None or not getattr(self.router, "api_key", None):
            return None
        import httpx
        from src.config import AI_BASE_URL
        url = AI_BASE_URL.rstrip("/") + self.EMBED_PATH
        try:
            r = httpx.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.router.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": "embo-01", "input": text[:2000]},
                timeout=30,
            )
            if r.status_code == 200:
                d = r.json()
                emb = d.get("data", [{}])[0].get("embedding", [])
                if emb:
                    self._embed_calls += 1
                    return emb
        except Exception as e:
            logger.debug(f"openai embed 失败: {e}")
        self._embed_fail += 1
        return None

    def _embed_st(self, text: str) -> Optional[List[float]]:
        if self._st_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer(
                    "paraphrase-multilingual-MiniLM-L12-v2"
                )
            except Exception as e:
                logger.debug(f"加载 sentence-transformers 失败: {e}")
                return None
        try:
            emb = self._st_model.encode(text[:2000], normalize_embeddings=True)
            self._embed_calls += 1
            return emb.tolist()
        except Exception:
            self._embed_fail += 1
            return None

    def _embed_tfidf(self, texts: List[str]) -> np.ndarray:
        """批量 TF-IDF 兜底 (全量 1 次算)."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        if self._tfidf_vec is None or len(self.docs) != self._tfidf_matrix.shape[0]:
            self._tfidf_vec = TfidfVectorizer(max_features=512, analyzer="char_wb",
                                              ngram_range=(2, 4))
            self._tfidf_matrix = self._tfidf_vec.fit_transform(
                [d.text for d in self.docs]
            )
        return self._tfidf_vec

    def _embed_one(self, text: str) -> Optional[List[float]]:
        backend = self._resolve_backend()
        if backend == "openai":
            v = self._embed_openai(text)
            if v is not None:
                return v
        if backend in ("st", "auto"):
            v = self._embed_st(text)
            if v is not None:
                return v
        # 终极降级: TF-IDF (单文档 vs 整体 corpus 算)
        return None

    # ============================================================
    # 文档管理
    # ============================================================
    def add_document(self, doc_id: str, text: str,
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        emb = self._embed_one(text)
        doc = EmbedDoc(
            id=doc_id, text=text[:1000],
            embedding=emb or [],
            metadata=metadata or {},
        )
        self.docs.append(doc)
        # TF-IDF 重新 fit
        if self._tfidf_vec is not None:
            self._tfidf_vec = None
            self._tfidf_matrix = None

    def add_documents_batch(self, docs: List[Tuple[str, str, Dict[str, Any]]]) -> None:
        """批量加文档, 1 次 API 调用 (openai 后端)."""
        backend = self._resolve_backend()
        if backend == "openai" and self.router is not None:
            try:
                import httpx
                from src.config import AI_BASE_URL
                url = AI_BASE_URL.rstrip("/") + self.EMBED_PATH
                r = httpx.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {self.router.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"model": "embo-01",
                          "input": [t[:2000] for _, t, _ in docs]},
                    timeout=60,
                )
                if r.status_code == 200:
                    data = r.json().get("data", [])
                    for (doc_id, text, meta), item in zip(docs, data):
                        emb = item.get("embedding", [])
                        self.docs.append(EmbedDoc(
                            id=doc_id, text=text[:1000], embedding=emb,
                            metadata=meta or {},
                        ))
                        self._embed_calls += 1
                    return
            except Exception as e:
                logger.debug(f"批量 embed 失败: {e}")
        # 降级到逐个
        for doc_id, text, meta in docs:
            self.add_document(doc_id, text, meta)

    # ============================================================
    # 查询
    # ============================================================
    def query(self, text: str, top_k: int = 5,
              min_score: float = 0.0) -> List[EmbedHit]:
        """向量检索 top_k."""
        if not self.docs:
            return []
        t0 = time.time()
        backend = self._resolve_backend()
        q_emb = self._embed_one(text)
        if q_emb and any(d.embedding for d in self.docs):
            # 向量余弦
            q = np.array(q_emb, dtype=np.float32)
            q = q / (np.linalg.norm(q) + 1e-9)
            scores = []
            for d in self.docs:
                if not d.embedding:
                    scores.append(0.0)
                    continue
                v = np.array(d.embedding, dtype=np.float32)
                v = v / (np.linalg.norm(v) + 1e-9)
                s = float(np.dot(q, v))
                scores.append(s)
        else:
            # TF-IDF 兜底
            from sklearn.metrics.pairwise import cosine_similarity
            if self._tfidf_vec is None or self._tfidf_matrix is None or self._tfidf_matrix.shape[0] != len(self.docs):
                self._embed_tfidf([d.text for d in self.docs])
            qv = self._tfidf_vec.transform([text])
            sims = cosine_similarity(qv, self._tfidf_matrix).flatten()
            scores = sims.tolist()
        order = np.argsort(scores)[::-1][:top_k]
        hits: List[EmbedHit] = []
        for idx in order:
            s = float(scores[idx])
            if s < min_score:
                continue
            d = self.docs[idx]
            hits.append(EmbedHit(
                id=d.id, text=d.text[:300], score=round(s, 4),
                metadata=d.metadata,
            ))
        self._last_query_vec = q_emb or []
        logger.debug(
            f"EmbedRAG.query({text[:30]}): {len(hits)} hits, "
            f"top={hits[0].score if hits else 0:.3f}, "
            f"backend={backend}, {round((time.time()-t0)*1000,1)}ms"
        )
        return hits

    def stats(self) -> Dict[str, Any]:
        return {
            "docs": len(self.docs),
            "backend": self._resolve_backend(),
            "embed_calls": self._embed_calls,
            "embed_fail": self._embed_fail,
            "last_query_dim": len(self._last_query_vec),
        }
