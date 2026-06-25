"""analysis.topic_model - LDA 主题建模

基于 sklearn LatentDirichletAllocation + jieba, 从当日文章挖掘潜在话题,
并按主题关键词自动打人类可读标签 + 关联主导产业。

能力:
- 主题抽取 (k = auto 或指定)
- 主题-产业关联 (top_words -> INDUSTRY_KEYWORDS 命中)
- 文档-主题分布 + 熵 (主题多样性)
- 主题强度时间演化 (跨日对比需要 history)

依赖: jieba, sklearn, numpy
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import jieba
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from src.config import INDUSTRY_KEYWORDS, load_stopwords
from src.nlp.stats import NLPStats
from src.scraper.pipeline import Article
from src.storage import repository as repo
from src.utils.date_utils import parse_date
from src.utils.logger import get_logger

logger = get_logger("analysis.topic_model")

_STOPWORDS = load_stopwords()

# 主题标签字典: 关键词 -> 人类可读主题名
_TOPIC_LABELS: List[Tuple[str, str]] = [
    (("降准", "降息", "LPR", "货币政策", "流动性", "央行", "MLF"), "货币与流动性"),
    (("财政", "赤字", "国债", "专项债", "减税", "退税"), "财政与债券"),
    (("房地产", "楼市", "房企", "保障房", "二手房", "保交楼"), "房地产"),
    (("新能源", "光伏", "风电", "锂电池", "储能", "氢能", "充电桩"), "新能源"),
    (("新能源车", "新能源汽车", "比亚迪", "造车新势力", "渗透率"), "新能源汽车"),
    (("芯片", "半导体", "集成电路", "晶圆", "光刻机", "EDA"), "半导体"),
    (("人工智能", "大模型", "AIGC", "生成式", "算力", "AGI", "人形机器人"), "AI 与算力"),
    (("数据要素", "数字经济", "数字人民币", "工业互联网"), "数字经济"),
    (("进出口", "外贸", "跨境电商", "一带一路", "自贸区", "RCEP"), "外贸与开放"),
    (("消费", "内需", "消费券", "以旧换新", "下沉市场"), "消费与内需"),
    (("乡村振兴", "粮食", "种业", "高标准农田", "农村"), "农业农村"),
    (("医药", "创新药", "集采", "医疗器械", "养老", "银发"), "医疗健康"),
    (("基建", "新基建", "5G", "特高压", "城际铁路", "数据中心"), "基础设施"),
    (("就业", "失业", "民生", "收入", "共同富裕"), "就业与民生"),
    (("美联储", "美元", "人民币汇率", "外汇", "G20", "IMF"), "国际宏观"),
    (("国资委", "央企", "国企改革", "重组"), "国资改革"),
]


def _label_for_top_words(top_words: List[str]) -> str:
    """根据主题词匹配一个可读标签。"""
    ws = set(top_words)
    for keys, label in _TOPIC_LABELS:
        if ws & set(keys):
            return label
    return top_words[0] if top_words else "未命名"


def _industry_for_top_words(top_words: List[str]) -> Optional[str]:
    ws = set(top_words)
    best = None
    best_overlap = 0
    for ind, keys in INDUSTRY_KEYWORDS.items():
        overlap = len(ws & set(keys))
        if overlap > best_overlap:
            best_overlap = overlap
            best = ind
    return best if best_overlap > 0 else None


def _tokenize(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[\s\W]+", " ", text)
    tokens = [t for t in jieba.cut(text) if t.strip() and t not in _STOPWORDS and len(t) > 1]
    return " ".join(tokens)


@dataclass
class Topic:
    id: int
    label: str
    top_words: List[str]
    dominance: float  # 0~1, 该主题在所有文档中的占比
    dominant_industry: Optional[str] = None

    def as_dict(self):
        return asdict(self)


@dataclass
class TopicEvolution:
    date: str
    topic_dist: List[Dict[str, float]]  # [{topic_id, strength}, ...]

    def as_dict(self):
        return asdict(self)


@dataclass
class TopicModelReport:
    date: str
    n_topics: int
    n_docs: int
    topics: List[Topic]
    doc_topic_entropy: float
    evolution: List[TopicEvolution] = field(default_factory=list)
    summary: str = ""

    def as_dict(self):
        d = asdict(self)
        d["topics"] = [t.as_dict() for t in self.topics]
        d["evolution"] = [e.as_dict() for e in self.evolution]
        return d


def _auto_k(n_docs: int) -> int:
    if n_docs < 5:
        return 2
    if n_docs < 15:
        return 3
    if n_docs < 30:
        return 5
    if n_docs < 60:
        return 6
    return 8


def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)) / np.log(len(p)))


def fit(articles: List[Article], n_topics: Optional[int] = None,
        max_features: int = 800, min_df: int = 1) -> Optional[TopicModelReport]:
    """对当日文章做 LDA 主题抽取。"""
    docs = []
    for a in articles:
        body = (a.content_text or a.title or "").strip()
        if not body:
            continue
        tokenized = _tokenize(body)
        if tokenized.strip():
            docs.append(tokenized)
    n_docs = len(docs)
    if n_docs < 3:
        logger.warning(f"文档数 {n_docs} 太少, 跳过主题建模")
        return None
    k = n_topics or _auto_k(n_docs)
    try:
        vec = CountVectorizer(
            max_features=max_features, min_df=min_df,
            token_pattern=r"\S+", lowercase=False,
        )
        X = vec.fit_transform(docs)
        if X.shape[1] == 0:
            logger.warning("词表为空, 跳过")
            return None
        lda = LatentDirichletAllocation(
            n_components=k, max_iter=20, learning_method="batch",
            random_state=42, n_jobs=1,
        )
        W = lda.fit_transform(X)  # doc-topic
        H = lda.components_         # topic-word
        vocab = vec.get_feature_names_out()
    except Exception as e:
        logger.warning(f"LDA 失败: {e}")
        return None
    # 主题强度 = 平均 doc-topic 概率
    strengths = W.mean(axis=0)
    s_sum = float(strengths.sum()) or 1.0
    topics: List[Topic] = []
    for i in range(k):
        top_idx = H[i].argsort()[::-1][:8]
        top_words = [vocab[j] for j in top_idx if vocab[j].strip()]
        label = _label_for_top_words(top_words)
        ind = _industry_for_top_words(top_words)
        topics.append(Topic(
            id=i, label=label, top_words=top_words,
            dominance=round(float(strengths[i] / s_sum), 4),
            dominant_industry=ind,
        ))
    # 文档-主题熵: 越接近 1 主题越分散
    ent = float(np.mean([_entropy(W[i]) for i in range(W.shape[0])]))
    return TopicModelReport(
        date="", n_topics=k, n_docs=n_docs,
        topics=topics, doc_topic_entropy=round(ent, 3),
        summary=f"从 {n_docs} 篇文档中提取 {k} 个主题, 主题-文档分布熵 {ent:.3f}",
    )


def fit_with_evolution(articles: List[Article], target_date: str,
                        history_days: int = 7,
                        n_topics: Optional[int] = None) -> Optional[TopicModelReport]:
    """对当日 + 历史 N 天做主题建模, 输出主题强度时间演化。

    注: 跨日建模需要历史 raw JSON, 否则只输出当天主题。
    """
    rep = fit(articles, n_topics=n_topics)
    if rep is None:
        return None
    rep.date = target_date
    # 加载历史 raw 报告, 做主题时间演化
    from src.report.archiver import load_raw
    evolution: List[TopicEvolution] = []
    today = parse_date(target_date)
    for back in range(history_days, 0, -1):
        d = (today - timedelta(days=back)).isoformat()
        try:
            hist_arts = load_raw(d)
        except FileNotFoundError:
            continue
        if len(hist_arts) < 3:
            continue
        hrep = fit(hist_arts, n_topics=rep.n_topics)
        if hrep is None:
            continue
        dist = [{"topic_id": t.id, "strength": t.dominance} for t in hrep.topics]
        evolution.append(TopicEvolution(date=d, topic_dist=dist))
    # 当天
    evolution.append(TopicEvolution(
        date=target_date,
        topic_dist=[{"topic_id": t.id, "strength": t.dominance} for t in rep.topics],
    ))
    rep.evolution = evolution
    return rep


if __name__ == "__main__":
    from src.report.archiver import load_raw
    import json
    arts = load_raw("2026-06-12")
    rep = fit_with_evolution(arts, "2026-06-12", history_days=3)
    if rep:
        print(json.dumps(rep.as_dict(), ensure_ascii=False, indent=2))
    else:
        print("无可建模数据")