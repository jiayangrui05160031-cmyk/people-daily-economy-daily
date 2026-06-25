"""analysis.rag_history - RAG 历史回放 (TF-IDF 余弦相似度)

从历史 reports/*.md + ai_report.payload_json 中检索最相似的历史片段,
给当前报告提供"跨期洞察"——今日主题和历史上哪一天最像, 那一天后来如何。

特性:
- 中文 jieba 分词 + sklearn TfidfVectorizer
- 余弦相似度检索 (Top-K)
- 同时检索 markdown 报告主体 + ai_report 关键内容
- 输出历史片段 + 后 1/3/7 天的指标变化 (走势)

依赖: jieba, sklearn, numpy (jieba + sklearn 已在本项目其它模块使用)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import REPORT_DIR, load_stopwords
from src.storage import repository as repo
from src.utils.date_utils import parse_date
from src.utils.logger import get_logger

logger = get_logger("analysis.rag_history")

_STOP = load_stopwords()


def _tokenize(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[\s\W]+", " ", text)
    toks = [t for t in jieba.cut(text) if t.strip() and t not in _STOP and len(t) > 1]
    return " ".join(toks)


@dataclass
class RecallItem:
    date: str
    score: float
    snippet: str
    followup: str = ""  # 历史片段之后 1-3 天的指标变化

    def as_dict(self):
        return asdict(self)


@dataclass
class RagReport:
    date: str
    recalls: List[RecallItem] = field(default_factory=list)
    top_score: float = 0.0
    summary: str = ""
    corpus_size: int = 0

    def as_dict(self):
        d = asdict(self)
        d["recalls"] = [r.as_dict() for r in self.recalls]
        return d


def _load_corpus(target_date: str) -> Tuple[List[str], List[str], List[str]]:
    """加载历史语料, 返回 (dates, docs_raw, dates_for_ai).

    docs_raw 保留原文片段, dates 与之一一对应。
    """
    target = parse_date(target_date)
    docs: List[str] = []
    dates: List[str] = []
    # 1. 加载 reports/{date}.md
    for md in sorted(REPORT_DIR.glob("*.md")):
        m = re.match(r"(\d{4}-\d{2}-\d{2})", md.name)
        if not m:
            continue
        d = m.group(1)
        if d >= target_date:
            continue  # 不要未来的
        try:
            text = md.read_text(encoding="utf-8")
        except Exception:
            continue
        # 只取前 4000 字提速
        docs.append(text[:4000])
        dates.append(d)
    # 2. 加载 ai_report.payload_json 里的核心内容 (作为补充)
    try:
        from src.storage.db import get_conn
        conn = get_conn()
        for r in conn.execute(
            "SELECT date, payload_json FROM ai_report WHERE date < ? ORDER BY date",
            (target_date,),
        ).fetchall():
            try:
                p = json.loads(r["payload_json"] or "{}")
            except Exception:
                continue
            parts = []
            tk = p.get("theme_keywords") or {}
            for k in (tk.get("keywords") or [])[:8]:
                parts.append(k.get("word", ""))
            ci = p.get("core_insights") or {}
            if ci.get("insights"):
                parts.append(ci["insights"])
            for ind in (p.get("industries") or {}).get("industries", [])[:3]:
                parts.append(ind.get("name", "") + ": " + ind.get("summary", ""))
            for o in (p.get("outlooks") or {}).get("outlooks", [])[:3]:
                parts.append(o.get("topic", "") + ": " + o.get("rationale", ""))
            if not parts:
                continue
            if r["date"] in dates:
                continue  # 已有 markdown
            docs.append(" ".join(parts)[:4000])
            dates.append(r["date"])
    except Exception as e:
        logger.debug(f"ai_report 语料加载失败: {e}")
    return dates, docs, dates


def _followup_for(hist_date: str, target_date: str) -> str:
    """历史日期之后 1-3 天的 sentiment / policy 走向。"""
    try:
        from src.storage.db import get_conn
        conn = get_conn()
        today = parse_date(hist_date)
        rows = []
        for off in (1, 3, 7):
            d = (today + timedelta(days=off)).isoformat()
            if d > target_date:
                break
            r = conn.execute(
                "SELECT sentiment_index, policy_stance_score FROM daily_metric WHERE date = ?",
                (d,),
            ).fetchone()
            if r:
                rows.append(f"+{off}d 情={r['sentiment_index']:.1f} 政={r['policy_stance_score']:+.2f}")
        return "; ".join(rows) if rows else "—"
    except Exception:
        return "—"


def recall(query_text: str, target_date: str, top_k: int = 5) -> Optional[RagReport]:
    """主入口: 给定当前报告的 query_text, 检索历史最相似片段。

    query_text 一般由 theme_keywords + core_insights + industries.summary 拼接。
    """
    if not query_text or not query_text.strip():
        return RagReport(date=target_date, summary="(无 query 文本, 跳过 RAG)")
    dates, docs, _ = _load_corpus(target_date)
    if len(dates) < 2:
        return RagReport(date=target_date,
                          summary=f"(历史语料仅 {len(dates)} 条, 不足 2 条, 跳过 RAG)",
                          corpus_size=len(dates))
    try:
        q_tok = _tokenize(query_text)
        d_toks = [_tokenize(d) for d in docs]
        vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, max_features=5000)
        M = vec.fit_transform(d_toks + [q_tok])
        if M.shape[0] == 0:
            return RagReport(date=target_date, summary="(词表为空, 跳过 RAG)",
                              corpus_size=len(dates))
        sims = cosine_similarity(M[-1], M[:-1]).flatten()
        order = np.argsort(-sims)
    except Exception as e:
        logger.warning(f"RAG 检索失败: {e}")
        return RagReport(date=target_date, summary=f"(检索失败: {e})", corpus_size=len(dates))
    items: List[RecallItem] = []
    top_score = 0.0
    for idx in order[:top_k]:
        s = float(sims[idx])
        if s <= 0.01:
            break
        d = dates[idx]
        snippet_src = docs[idx][:300]
        # 截掉 markdown 标记
        snippet = re.sub(r"[#*`>|]+", " ", snippet_src)
        snippet = re.sub(r"\s+", " ", snippet).strip()
        if len(snippet) > 160:
            snippet = snippet[:160] + "…"
        items.append(RecallItem(
            date=d, score=round(s, 4), snippet=snippet,
            followup=_followup_for(d, target_date),
        ))
        if s > top_score:
            top_score = s
    summary = ""
    if items:
        summary = (
            "与 " + items[0].date + " 报告最相似 (" + str(round(items[0].score, 3)) + "); "
            + "; ".join([it.date for it in items[1:]]) + " 次相似。"
        )
    return RagReport(date=target_date, recalls=items,
                      top_score=round(top_score, 4), summary=summary,
                      corpus_size=len(dates))


if __name__ == "__main__":
    from src.ai.schema import AnalysisReport
    from src.storage import db
    db.get_conn()
    import json
    target = "2026-06-12"
    # 从已存在的 ai_report 拿 query_text
    from src.storage.db import get_conn
    conn = get_conn()
    r = conn.execute(
        "SELECT payload_json FROM ai_report WHERE date = ?", (target,),
    ).fetchone()
    if r:
        p = json.loads(r["payload_json"] or "{}")
        parts = []
        for k in (p.get("theme_keywords") or {}).get("keywords", [])[:8]:
            parts.append(k.get("word", ""))
        if (p.get("core_insights") or {}).get("insights"):
            parts.append(p["core_insights"]["insights"])
        for ind in (p.get("industries") or {}).get("industries", [])[:3]:
            parts.append(ind.get("name", "") + ": " + ind.get("summary", ""))
        query = " ".join(parts)
        rep = recall(query, target, top_k=5)
        print(json.dumps(rep.as_dict(), ensure_ascii=False, indent=2))
    else:
        print("无 ai_report")