"""nlp — 中文自然语言处理层 ==============================================
jieba 分词、TF-IDF/TextRank 关键词、词频与词云。
"""

from .stats import analyze, NLPStats

__all__ = ["analyze", "NLPStats"]