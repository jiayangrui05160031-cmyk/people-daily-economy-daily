"""analysis.similarity - 多路相似度计算

- jaccard: 词集合 Jaccard (快速)
- tfidf_cosine: TF-IDF 向量余弦 (稍慢,语义相关性更好)
- cooccurrence: 同文章共现 (实体关联用)
- llm_similarity: 让 LLM 直接给两段文本 0~1 相似度 (高质量,慢)
"""
from __future__ import annotations

from collections import Counter
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.nlp.tokenizer import tokenize
from src.utils.logger import get_logger

logger = get_logger("analysis.similarity")


def _tokens(text, remove_stop=True):
    return tokenize(text, remove_stop=remove_stop)


def jaccard(a, b):
    sa = set(_tokens(a))
    sb = set(_tokens(b))
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def jaccard_batch(docs):
    n = len(docs)
    M = np.eye(n)
    sets = [set(_tokens(d)) for d in docs]
    for i in range(n):
        for j in range(i + 1, n):
            if not sets[i] and not sets[j]:
                sim = 0.0
            else:
                sim = len(sets[i] & sets[j]) / len(sets[i] | sets[j])
            M[i, j] = M[j, i] = sim
    return M


def tfidf_cosine(docs):
    if not docs:
        return np.zeros((0, 0))
    joined = [" ".join(_tokens(d)) for d in docs]
    if not any(joined):
        return np.zeros((len(docs), len(docs)))
    try:
        vec = TfidfVectorizer(token_pattern=r"(?u)\S+", max_features=5000)
        m = vec.fit_transform(joined)
        return cosine_similarity(m)
    except Exception as e:
        logger.warning(f"tfidf_cosine failed: {e}")
        return np.eye(len(docs))


def cooccurrence(articles_text, entity, window=1):
    counter = Counter()
    for text in articles_text:
        tokens = _tokens(text)
        for i, t in enumerate(tokens):
            if entity in t:
                lo = max(0, i - window)
                hi = min(len(tokens), i + window + 1)
                for j in range(lo, hi):
                    if j != i and entity not in tokens[j]:
                        counter[tokens[j]] += 1
    return counter.most_common(20)


def llm_similarity(router, text_a, text_b, model=None):
    if not router:
        return 0.0
    prompt = (
        "请基于语义内容判断下面两段新闻的相似度,返回 0~1 之间的小数。"
        "只输出数字,不要解释。\n\n"
        f"文本 A:{text_a[:300]}\n\n文本 B:{text_b[:300]}\n\n相似度:"
    )
    try:
        raw, _, _, _ = router.chat(
            messages=[{"role": "system", "content": "你是语义相似度打分器,只输出 0~1 之间的数字。"},
                      {"role": "user", "content": prompt}],
            model=model,
            temperature=0.0,
            max_tokens=20,
            use_json_mode=False,
        )
        import re
        m = re.search(r"\d*\.?\d+", raw or "")
        if m:
            v = float(m.group(0))
            return max(0.0, min(1.0, v))
    except Exception as e:
        logger.warning(f"llm_similarity failed: {e}")
    return 0.0


if __name__ == "__main__":
    a = "央行宣布降准1万亿释放长期资金支持实体经济"
    b = "中国人民银行下调存款准备金率释放流动性一亿元"
    c = "新能源汽车销量增长,渗透率突破百分之五十"
    print("jaccard(a, b):", round(jaccard(a, b), 3))
    print("jaccard(a, c):", round(jaccard(a, c), 3))
    M = tfidf_cosine([a, b, c])
    print("tfidf cosine M:\n", np.round(M, 3))
