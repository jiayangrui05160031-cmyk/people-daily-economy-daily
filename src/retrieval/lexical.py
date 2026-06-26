"""retrieval.lexical - TF-IDF 词袋检索 (replaces rag_history top-level).

rag_history.py 仍然保留作为"主入口 (recall + followup 历史走势)"
的兼容壳，但核心 TF-IDF 检索逻辑搬到这里做成 LexicalRetriever。
qa_assistant 通过它做召回，新代码不需要再 import rag_history。
"""
from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import REPORT_DIR, load_stopwords
from src.utils.logger import get_logger
from .base import Hit

logger = get_logger("retrieval.lexical")
_STOP = load_stopwords()


def _tokenize(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[\s\W]+", " ", text)
    toks = [t for t in jieba.cut(text) if t.strip() and t not in _STOP and len(t) > 1]
    return " ".join(toks)


def _chunk_markdown(text: str, max_chars: int = 400) -> List[str]:
    if not text:
        return []
    sections = re.split(r"\n##\s+", text)
    out: List[str] = []
    for s in sections:
        s = s.strip()
        if not s:
            continue
        if len(s) <= max_chars:
            out.append(s)
        else:
            for i in range(0, len(s), max_chars):
                out.append(s[i:i + max_chars])
    return out


class LexicalRetriever:
    """TF-IDF + jieba + cosine. 适合: 跨期主题追踪, 短查询-长文档。"""

    name = "lexical"

    def __init__(self, corpus_dir: Optional[Path] = None,
                 max_files: int = 60, max_features: int = 20000):
        self.corpus_dir = Path(corpus_dir) if corpus_dir else REPORT_DIR
        self.max_files = max_files
        self.max_features = max_features
        self._corpus: List[Tuple[str, str]] = []   # (date, chunk)
        self._corpus_key: Optional[str] = None

    def _load_corpus(self) -> None:
        if not self.corpus_dir.exists():
            self._corpus = []
            return
        files = sorted(self.corpus_dir.glob("*.md"),
                       key=lambda p: p.name, reverse=True)[:self.max_files]
        corpus: List[Tuple[str, str]] = []
        for fp in files:
            try:
                text = fp.read_text(encoding="utf-8")
            except Exception:
                continue
            dmatch = re.search(r"(\d{4}-\d{2}-\d{2})", fp.stem)
            d = dmatch.group(1) if dmatch else fp.stem
            for chunk in _chunk_markdown(text):
                corpus.append((d, chunk))
        self._corpus = corpus
        self._corpus_key = f"{len(files)}|{files[0].name if files else 'none'}"

    def invalidate(self) -> None:
        """Force reload on next search (call after new reports)."""
        self._corpus = []
        self._corpus_key = None

    def _ensure(self) -> None:
        key = "no-files-loaded"
        if self.corpus_dir.exists():
            files = sorted(self.corpus_dir.glob("*.md"),
                           key=lambda p: p.name, reverse=True)[:self.max_files]
            key = f"{len(files)}|{files[0].name if files else 'none'}"
        if key != self._corpus_key:
            self._load_corpus()

    def search(self, query: str, top_k: int = 5,
               min_score: float = 0.05, **kwargs) -> List[Hit]:
        self._ensure()
        if not query or not query.strip() or not self._corpus:
            return []
        try:
            qvec = _tokenize(query)
            cvecs = [_tokenize(c) for _, c in self._corpus]
            if not qvec or not any(cvecs):
                return []
            vec = TfidfVectorizer(max_features=self.max_features)
            X = vec.fit_transform([qvec] + cvecs)
            sims = cosine_similarity(X[0], X[1:]).flatten()
            order = sims.argsort()[::-1][:top_k]
        except Exception as e:
            logger.debug(f"lexical search 失败: {e}")
            return []
        hits: List[Hit] = []
        for idx in order:
            score = float(sims[idx])
            if score < min_score:
                continue
            d, chunk = self._corpus[idx]
            hits.append(Hit(
                id=f"report:{d}:{idx}",
                text=chunk[:300].replace("\n", " "),
                score=round(score, 4),
                source=self.name,
                metadata={"date": d, "chunk_idx": idx, "kind": "report"},
            ))
        return hits
