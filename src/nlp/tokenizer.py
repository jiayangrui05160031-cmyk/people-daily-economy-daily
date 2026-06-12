"""tokenizer — 中文分词 + 停用词 ==============================================
jieba 分词,过滤停用词、纯数字、纯标点、单字。
"""

from __future__ import annotations

from typing import List

import jieba
import re

from src.config import load_stopwords
from src.utils.logger import get_logger

logger = get_logger("nlp.tokenizer")

# 标点 / 空白字符
_PUNCT_RE = re.compile(r"[\s\W]+", re.UNICODE)


def tokenize(text: str, remove_stop: bool = True, min_len: int = 2) -> List[str]:
    """对单段文本分词,返回干净的词列表。

    Args:
        text: 输入文本
        remove_stop: 是否过滤停用词
        min_len: 最小词长(过滤单字)

    Returns:
        词列表
    """
    if not text:
        return []

    # jieba 分词(精确模式)
    tokens = jieba.lcut(text, cut_all=False)

    # 清洗
    stop_words = load_stopwords() if remove_stop else set()
    cleaned: List[str] = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if len(tok) < min_len:
            continue
        if remove_stop and tok in stop_words:
            continue
        # 过滤纯数字 / 纯英文(无中文)
        if not re.search(r"[一-鿿]", tok):
            continue
        # 过滤纯标点
        if _PUNCT_RE.fullmatch(tok):
            continue
        cleaned.append(tok)

    return cleaned


def tokenize_articles(articles, remove_stop: bool = True, min_len: int = 2) -> List[str]:
    """批量对 Article 列表分词,合并所有正文段落。"""
    all_tokens: List[str] = []
    for art in articles:
        for paragraph in art.content:
            all_tokens.extend(tokenize(paragraph, remove_stop=remove_stop, min_len=min_len))
    return all_tokens


if __name__ == "__main__":
    sample = "工业和信息化部、市场监管总局11日对涉嫌存在非理性竞争汽车生产企业开展约谈"
    print(tokenize(sample))