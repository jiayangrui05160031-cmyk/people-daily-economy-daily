"""keywords — 关键词提取 ==============================================
TF-IDF 与 TextRank 双路融合,综合返回 top-K 关键词。
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer

from src.nlp.tokenizer import tokenize
from src.utils.logger import get_logger

logger = get_logger("nlp.keywords")


def _texts_to_joined(paragraphs: List[str]) -> List[str]:
    """将每段文本用空格连接,作为 TF-IDF 输入。"""
    return [" ".join(tokenize(p, remove_stop=True)) for p in paragraphs if p]


def extract_tfidf_keywords(paragraphs: List[str], top_k: int = 30) -> List[Tuple[str, float]]:
    """TF-IDF 提取关键词。

    Args:
        paragraphs: 段落列表(每篇一文档)
        top_k: 返回前 K 个

    Returns:
        [(word, score), ...]
    """
    docs = _texts_to_joined(paragraphs)
    if not docs:
        return []

    try:
        vec = TfidfVectorizer(
            token_pattern=r"(?u)\S+",  # 已在 _texts_to_joined 里分好词
            max_features=5000,
        )
        matrix = vec.fit_transform(docs)
        # 全局统计:每个词在所有文档里的 TF-IDF 总和
        sums = matrix.sum(axis=0).A1
        vocab = vec.get_feature_names_out()
        pairs = list(zip(vocab, sums))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return [(w, float(s)) for w, s in pairs[:top_k] if s > 0]
    except Exception as e:
        logger.warning(f"TF-IDF 失败: {e}")
        return []


def extract_textrank_keywords(paragraphs: List[str], top_k: int = 30, window: int = 3) -> List[Tuple[str, float]]:
    """TextRank 提取关键词(基于词共现图)。

    Args:
        paragraphs: 段落列表
        top_k: 返回前 K 个
        window: 共现窗口

    Returns:
        [(word, score), ...]
    """
    # 在所有段落上做(整体图而非每段一图)
    all_tokens: List[str] = []
    for p in paragraphs:
        all_tokens.extend(tokenize(p, remove_stop=True))

    if not all_tokens:
        return []

    # 建图:共现窗口内的词构成无向边
    graph = nx.Graph()
    graph.add_nodes_from(set(all_tokens))
    for i in range(len(all_tokens) - window + 1):
        window_tokens = all_tokens[i:i + window]
        for a in window_tokens:
            for b in window_tokens:
                if a != b and graph.has_edge(a, b):
                    graph[a][b]["weight"] += 1
                elif a != b:
                    graph.add_edge(a, b, weight=1)

    try:
        scores = nx.pagerank(graph, alpha=0.85, max_iter=100)
    except Exception as e:
        logger.warning(f"TextRank 失败: {e}")
        return []

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(w, float(s)) for w, s in ranked[:top_k] if s > 0]


def merge_keywords(
    tfidf: List[Tuple[str, float]],
    textrank: List[Tuple[str, float]],
    top_k: int = 20,
    alpha: float = 0.5,
) -> List[Tuple[str, float]]:
    """双路融合:加权合并,去重,排序。

    融合分数 = alpha * 归一化TF-IDF + (1-alpha) * 归一化TextRank
    """
    def normalize(pairs):
        if not pairs:
            return {}
        max_s = max(s for _, s in pairs) or 1.0
        return {w: s / max_s for w, s in pairs}

    tfidf_n = normalize(tfidf)
    tr_n = normalize(textrank)
    merged: Dict[str, float] = {}

    for w, s in tfidf_n.items():
        merged[w] = merged.get(w, 0) + alpha * s
    for w, s in tr_n.items():
        merged[w] = merged.get(w, 0) + (1 - alpha) * s

    ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


if __name__ == "__main__":
    sample = [
        "工业和信息化部对涉嫌非理性竞争汽车生产企业开展约谈",
        "新能源汽车产业进入高质量发展新阶段",
        "央行宣布降准,释放长期资金约1万亿元",
        "光伏行业产能扩张,组件价格持续下行",
    ]
    tfidf = extract_tfidf_keywords(sample, top_k=10)
    tr = extract_textrank_keywords(sample, top_k=10)
    print("TF-IDF:", tfidf[:5])
    print("TextRank:", tr[:5])
    print("Merged:", merge_keywords(tfidf, tr, top_k=10)[:5])