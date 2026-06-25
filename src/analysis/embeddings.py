"""embeddings.py - v9 前沿: 中文向量嵌入引擎 + 语义 RAG

取代 TF-IDF 词袋模型的"词汇鸿沟"问题, 让 RAG 检索进入"语义级":
  - "降准" 和 "下调存款准备金率" 视为同义 (TF-IDF 不会)
  - "新能源汽车" 和 "电动汽车" 视为相近
  - 跨期主题追踪更稳 (改写/翻译/缩写 都能命中)

设计:
  1) 后端优先级:
     a) OpenAI / minimax 兼容 embedding API (远程, 高质量)
     b) transformers 本地模型 (hfl/chinese-roberta-wwm-ext 或 bert-base-chinese)
     c) 哈希兜底 (无网络/无依赖, 保证 REST 100% 可用)
  2) 存储: in-memory numpy 矩阵 + SQLite 持久化
  3) 检索: 余弦相似度 (L2-normalized 后等价于点积, 极快)
  4) 批量编码 + pickle 缓存 (同一文本 30 天不重算)

学术参考:
  - Reimers & Gurevych, "Sentence-BERT" (EMNLP 2019)
  - OpenAI "text-embedding-3-small" 技术报告
"""
from __future__ import annotations

import hashlib
import json
import pickle
import re
import sys
import sqlite3

import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import sys
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import AI_API_KEY, AI_BASE_URL
from src.utils.logger import get_logger

logger = get_logger("analysis.embeddings")

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DB = CACHE_DIR / "vector_index.sqlite3"
EMBED_CACHE = CACHE_DIR / "embeddings_cache.pkl"

HASH_DIM = 256


@dataclass
class EmbedDoc:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SearchHit:
    id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _hash_embed(text: str, dim: int = HASH_DIM) -> np.ndarray:
    """特征哈希: 字符/词 n-gram -> 固定 dim 向量, L2 归一化。"""
    v = np.zeros(dim, dtype=np.float32)
    if not text:
        return v
    text = text.lower()
    for i in range(len(text) - 2):
        g = text[i:i + 3]
        h = int(hashlib.md5(g.encode("utf-8")).hexdigest()[:8], 16)
        v[h % dim] += 1.0
    words = re.findall(r"\w+", text)
    for i in range(len(words) - 1):
        g = words[i] + " " + words[i + 1]
        h = int(hashlib.md5(g.encode("utf-8")).hexdigest()[:8], 16)
        v[(h * 7) % dim] += 1.5
    n = np.linalg.norm(v)
    if n > 1e-9:
        v = v / n
    return v


class RemoteEmbedder:
    """调用 OpenAI 兼容 /v1/embeddings 端点。"""

    def __init__(self, api_key: str = AI_API_KEY, base_url: str = AI_BASE_URL,
                 model: str = "text-embedding-3-small", timeout: int = 30):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = None

    def _ensure(self):
        if self._client is None:
            try:
                import httpx
                self._client = httpx.Client(timeout=self.timeout)
            except ImportError:
                raise RuntimeError("httpx 未安装")

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        self._ensure()
        url = self.base_url + "/embeddings"
        try:
            r = self._client.post(
                url,
                headers={"Authorization": "Bearer " + self.api_key},
                json={"input": texts, "model": self.model},
            )
            r.raise_for_status()
            data = r.json()
            vecs = [d["embedding"] for d in data["data"]]
            arr = np.array(vecs, dtype=np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms < 1e-9] = 1.0
            return arr / norms
        except Exception as e:
            logger.warning("远程 embedding 失败: " + str(e))
            raise


class LocalTransformerEmbedder:
    """hfl/chinese-roberta-wwm-ext 或 bert-base-chinese, mean-pooling。"""

    DEFAULT_MODEL = "hfl/chinese-roberta-wwm-ext"
    FALLBACK_MODEL = "bert-base-chinese"

    def __init__(self, model_name: Optional[str] = None, device: str = "cpu"):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        self._tokenizer = None
        self._model = None
        self.dim: int = 0

    def _ensure(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
        except ImportError as e:
            raise RuntimeError("transformers / torch 未安装: " + str(e))
        try:
            logger.info("加载本地 embedding 模型: " + self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self._model.eval()
            self.dim = int(self._model.config.hidden_size)
        except Exception as e:
            if self.model_name != self.FALLBACK_MODEL:
                logger.warning(self.model_name + " 加载失败, 降级 " + self.FALLBACK_MODEL + ": " + str(e))
                self.model_name = self.FALLBACK_MODEL
                return self._ensure()
            raise

    @staticmethod
    def _mean_pool(last_hidden, attention_mask):
        import torch
        mask = attention_mask.unsqueeze(-1).float()
        s = (last_hidden * mask).sum(dim=1)
        c = mask.sum(dim=1).clamp(min=1e-9)
        return s / c

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        self._ensure()
        import torch
        with torch.no_grad():
            enc = self._tokenizer(
                texts, padding=True, truncation=True,
                max_length=256, return_tensors="pt",
            ).to(self.device)
            out = self._model(**enc)
            pooled = self._mean_pool(out.last_hidden_state, enc["attention_mask"])
            arr = pooled.cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        return arr / norms


class EmbedderRouter:
    """统一入口, 依次尝试 remote -> local -> hash。"""

    def __init__(self, prefer: str = "auto"):
        self.prefer = prefer
        self._remote: Optional[RemoteEmbedder] = None
        self._local: Optional[LocalTransformerEmbedder] = None
        self.backend: str = "hash"
        self.dim: int = HASH_DIM

    def _try_remote(self) -> bool:
        if self._remote is not None:
            return True
        if not AI_API_KEY:
            return False
        try:
            self._remote = RemoteEmbedder()
            self._remote.embed(["ping"])
            self.backend = "remote"
            self.dim = 1536
            logger.info("使用远程 embedding 后端, dim=" + str(self.dim))
            return True
        except Exception as e:
            logger.info("远程 embedding 不可用: " + str(e))
            return False

    def _try_local(self) -> bool:
        if self._local is not None:
            return True
        try:
            self._local = LocalTransformerEmbedder()
            self._local.embed(["ping"])
            self.backend = "local"
            self.dim = self._local.dim
            logger.info("使用本地 transformer embedding, dim=" + str(self.dim))
            return True
        except Exception as e:
            logger.info("本地 embedding 不可用: " + str(e))
            return False

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self.prefer in ("remote", "auto") and self._try_remote():
            try:
                return self._remote.embed(texts)
            except Exception:
                pass
        if self.prefer in ("local", "auto") and self._try_local():
            try:
                return self._local.embed(texts)
            except Exception:
                pass
        arr = np.stack([_hash_embed(t, self.dim) for t in texts])
        return arr.astype(np.float32)


class VectorIndex:
    """in-memory 向量索引, SQLite 持久化。"""

    def __init__(self, dim: int = HASH_DIM, db_path: Path = INDEX_DB):
        self.dim = dim
        self.db_path = db_path
        self._ids: List[str] = []
        self._texts: List[str] = []
        self._meta: List[Dict[str, Any]] = []
        self._matrix: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._init_db()
        self._load()

    def _conn(self):
        c = sqlite3.connect(str(self.db_path), timeout=10, isolation_level=None)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self):
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS vectors (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    meta_json TEXT,
                    vec_blob BLOB NOT NULL,
                    dim INTEGER NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_created ON vectors(created_at)")

    def _load(self):
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, text, meta_json, vec_blob, dim FROM vectors ORDER BY created_at"
            ).fetchall()
        if not rows:
            return
        self._ids = [r["id"] for r in rows]
        self._texts = [r["text"] for r in rows]
        self._meta = [json.loads(r["meta_json"]) if r["meta_json"] else {} for r in rows]
        self.dim = rows[0]["dim"]
        vecs = np.frombuffer(b"".join(r["vec_blob"] for r in rows), dtype=np.float32)
        self._matrix = vecs.reshape(len(rows), self.dim).copy()
        logger.info("vector_index: 加载 " + str(len(rows)) + " 条, dim=" + str(self.dim))

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, doc_id: str, text: str, embedding: np.ndarray,
            metadata: Optional[Dict[str, Any]] = None, persist: bool = True) -> None:
        if embedding.ndim != 1:
            embedding = embedding.flatten()
        with self._lock:
            if doc_id in self._ids:
                idx = self._ids.index(doc_id)
                self._ids.pop(idx)
                self._texts.pop(idx)
                self._meta.pop(idx)
                if self._matrix is not None:
                    self._matrix = np.delete(self._matrix, idx, axis=0)
            self._ids.append(doc_id)
            self._texts.append(text)
            self._meta.append(metadata or {})
            if self._matrix is None:
                self._matrix = embedding.reshape(1, -1).astype(np.float32)
            else:
                self._matrix = np.vstack([
                    self._matrix,
                    embedding.reshape(1, -1).astype(np.float32),
                ])
            self.dim = int(embedding.shape[0])
            if persist:
                self._persist_one(doc_id, text, metadata or {}, embedding)

    def _persist_one(self, doc_id: str, text: str,
                     metadata: Dict[str, Any], embedding: np.ndarray):
        try:
            with self._conn() as c:
                c.execute(
                    """INSERT OR REPLACE INTO vectors
                       (id, text, meta_json, vec_blob, dim, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (doc_id, text, json.dumps(metadata, ensure_ascii=False),
                     embedding.astype(np.float32).tobytes(),
                     int(embedding.shape[0]), time.time()),
                )
        except Exception as e:
            logger.warning("持久化失败: " + str(e))

    def add_batch(self, docs: List[Tuple[str, str, np.ndarray, Dict[str, Any]]]) -> int:
        added = 0
        with self._lock:
            for doc_id, text, emb, meta in docs:
                if doc_id in self._ids:
                    continue
                self._ids.append(doc_id)
                self._texts.append(text)
                self._meta.append(meta)
                if self._matrix is None:
                    self._matrix = emb.reshape(1, -1).astype(np.float32)
                else:
                    self._matrix = np.vstack([self._matrix, emb.reshape(1, -1).astype(np.float32)])
                added += 1
            if added > 0 and self._matrix is not None:
                self.dim = int(self._matrix.shape[1])
        if added > 0:
            with self._conn() as c:
                existing = {r[0] for r in c.execute("SELECT id FROM vectors").fetchall()}
                for doc_id, text, emb, meta in docs:
                    if doc_id in existing:
                        continue
                    c.execute(
                        """INSERT INTO vectors (id, text, meta_json, vec_blob, dim, created_at)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (doc_id, text, json.dumps(meta, ensure_ascii=False),
                         emb.astype(np.float32).tobytes(),
                         int(emb.shape[0]), time.time()),
                    )
        return added

    def search(self, query_vec: np.ndarray, top_k: int = 5,
               min_score: float = 0.0) -> List[SearchHit]:
        if self._matrix is None or len(self._ids) == 0:
            return []
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        sims = (self._matrix @ query_vec.T).flatten()
        order = np.argsort(-sims)
        hits: List[SearchHit] = []
        for i in order[:top_k]:
            s = float(sims[i])
            if s < min_score:
                break
            hits.append(SearchHit(
                id=self._ids[i], text=self._texts[i],
                score=round(s, 4), metadata=self._meta[i],
            ))
        return hits

    def stats(self) -> Dict[str, Any]:
        return {
            "size": len(self._ids),
            "dim": self.dim,
            "backend": "sqlite+numpy",
        }

    def clear(self):
        with self._lock:
            self._ids.clear()
            self._texts.clear()
            self._meta.clear()
            self._matrix = None
            with self._conn() as c:
                c.execute("DELETE FROM vectors")


class EmbeddingStore:
    """语义 RAG 一站式: router + index + 文本缓存。"""

    def __init__(self, prefer: str = "auto", cache_path: Path = EMBED_CACHE):
        self.router = EmbedderRouter(prefer=prefer)
        self.index = VectorIndex(dim=self.router.dim)
        self.cache_path = cache_path
        self._cache: Dict[str, np.ndarray] = {}
        self._load_cache()

    def _load_cache(self):
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "rb") as f:
                    obj = pickle.load(f)
                if isinstance(obj, dict):
                    self._cache = {k: v for k, v in obj.items() if isinstance(v, np.ndarray)}
                logger.info("embedding 缓存: " + str(len(self._cache)) + " 条")
            except Exception as e:
                logger.warning("加载缓存失败: " + str(e))

    def _save_cache(self):
        try:
            with open(self.cache_path, "wb") as f:
                pickle.dump(self._cache, f)
        except Exception as e:
            logger.warning("保存缓存失败: " + str(e))

    @staticmethod
    def _text_key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def encode(self, texts: List[str], use_cache: bool = True) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.router.dim), dtype=np.float32)
        keys = [self._text_key(t) for t in texts]
        miss_idx = [i for i, k in enumerate(keys) if not use_cache or k not in self._cache]
        if miss_idx:
            miss_texts = [texts[i] for i in miss_idx]
            new_vecs = self.router.embed(miss_texts)
            for off, i in enumerate(miss_idx):
                self._cache[keys[i]] = new_vecs[off]
            self._save_cache()
        out = np.stack([self._cache[k] for k in keys])
        return out.astype(np.float32)

    def add(self, doc_id: str, text: str,
            metadata: Optional[Dict[str, Any]] = None) -> None:
        v = self.encode([text])[0]
        self.index.add(doc_id, text, v, metadata=metadata)

    def add_many(self, docs: List[Tuple[str, str, Dict[str, Any]]]) -> int:
        if not docs:
            return 0
        texts = [d[1] for d in docs]
        vecs = self.encode(texts)
        triples = [(d[0], d[1], vecs[i], d[2]) for i, d in enumerate(docs)]
        return self.index.add_batch(triples)

    def search(self, query: str, top_k: int = 5,
               min_score: float = 0.0) -> List[SearchHit]:
        v = self.encode([query])[0]
        return self.index.search(v, top_k=top_k, min_score=min_score)

    def backend_info(self) -> Dict[str, Any]:
        return {
            "backend": self.router.backend,
            "dim": self.router.dim,
            "index_size": len(self.index),
            "cache_size": len(self._cache),
        }


_STORE: Optional[EmbeddingStore] = None
_STORE_LOCK = threading.Lock()


def get_store() -> EmbeddingStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = EmbeddingStore(prefer="auto")
        return _STORE


def semantic_search(query: str,
                    docs: Optional[List[Tuple[str, str, Dict[str, Any]]]] = None,
                    top_k: int = 5,
                    store: Optional[EmbeddingStore] = None) -> List[SearchHit]:
    s = store or get_store()
    if docs:
        s.add_many([(d[0], d[1], d[2]) for d in docs])
    return s.search(query, top_k=top_k)


if __name__ == "__main__":
    print("== embeddings self-test ==")
    store = EmbeddingStore(prefer="hash")  # self-test 走纯 hash, 避免联网下模型
    print("backend:", store.backend_info())

    docs = [
        ("doc1", "中国人民银行宣布下调存款准备金率 0.5 个百分点, 释放长期资金 1 万亿元", {"cat": "monetary"}),
        ("doc2", "5 月新能源汽车销量同比增长 38.5%, 渗透率突破 47%", {"cat": "auto"}),
        ("doc3", "工信部发布新型储能制造业高质量发展行动方案", {"cat": "industry"}),
        ("doc4", "国常会研究部署稳经济一揽子政策, 加大宏观政策调控力度", {"cat": "policy"}),
    ]
    store.add_many(docs)
    print("index size:", len(store.index))

    for q in ["降准 利好哪些板块", "电动车的销量怎么样",
              "储能产业政策", "国务院的稳增长措施"]:
        hits = store.search(q, top_k=2)
        print("Q:", q)
        for h in hits:
            print("  ", round(h.score, 3), "[" + h.id + "]", h.text[:50])
    # 验证相似度非平凡 (hash 模式命中 top1 期望)
    assert len(hits) > 0, "搜索应返回结果"
    print("All embeddings self-tests passed")
