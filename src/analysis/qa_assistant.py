"""analysis.qa_assistant - LLM 智能问答 (v9 重构为编排器)

让用户用自然语言提问宏观问题, 系统:
  1) 通过 retrieval/ 子包的 Retriever protocol 召回 Top-K 相关片段
     - 缺省: LexicalRetriever (TF-IDF 跨期主题)
     - 有 router+api_key: VectorRetriever (语义) + GraphRetriever (关系) 并联
  2) 从 daily_metric 拉时序数据 (近 30 天)
  3) 拼装 Prompt, 让 LLM 给出基于证据的答案 + 引用来源

无 LLM 时降级为 extractive 答案 (取 Top-K 片段前 2 句), 保证总能用。

RAG 层已统一到 retrieval/ 子包, 见 src/retrieval/__init__.py。
embedding-rag 重复实现 (embed_rag.py) 已删除。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.storage import db as db_mod
from src.storage import repository as repo
from src.utils.date_utils import parse_date
from src.utils.logger import get_logger

logger = get_logger("analysis.qa_assistant")


# ---------------------------------------------------------------------------
# Citation (back-compat type used in REST responses / reports)
# ---------------------------------------------------------------------------


@dataclass
class Citation:
    source: str       # 'lexical:report:2026-06-12.md' | 'vector:doc-1' | 'graph:community:3'
    date: str
    score: float      # 0~1
    snippet: str

    def as_dict(self):
        return asdict(self)


@dataclass
class QAResult:
    question: str
    answer: str
    citations: List[Citation] = field(default_factory=list)
    context_dates: List[str] = field(default_factory=list)
    metrics_snapshot: Dict[str, float] = field(default_factory=dict)
    generated_by: str = "template"     # 'llm' | 'template'
    model: str = ""
    confidence: float = 0.0
    summary: str = ""

    def as_dict(self):
        d = asdict(self)
        d["citations"] = [c.as_dict() for c in self.citations]
        return d


# ---------------------------------------------------------------------------
# Retriever wiring (v9)
# ---------------------------------------------------------------------------


def _build_default_retrievers(router=None):
    """Build the default Retriever set. Pure-function so it's testable."""
    from src.retrieval import LexicalRetriever, VectorRetriever, GraphRetriever
    retrievers = [LexicalRetriever()]
    if router is not None and getattr(router, "api_key", None):
        retrievers.append(VectorRetriever())
        retrievers.append(GraphRetriever(router=router))
    return retrievers


def _hit_to_citation(hit) -> Citation:
    return Citation(
        source=f"{hit.source}:{hit.id}",
        date=hit.metadata.get("date", ""),
        score=hit.score,
        snippet=hit.text[:200].replace("\n", " "),
    )


# ---------------------------------------------------------------------------
# Time-series snapshot
# ---------------------------------------------------------------------------


def _metrics_snapshot(target_date: str, lookback: int = 30) -> Tuple[Dict[str, float], List[str]]:
    """拉最近 N 天 daily_metric, 返回均值 + 日期列表。"""
    dates: List[str] = []
    snap: Dict[str, float] = {}
    try:
        end = parse_date(target_date)
        from datetime import timedelta
        start = end - timedelta(days=lookback)
        with db_mod.get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM daily_metric WHERE date BETWEEN ? AND ? ORDER BY date",
                (start.isoformat(), end.isoformat()),
            ).fetchall()
        if not rows:
            return snap, dates
        dates = [r["date"] for r in rows]
        for key in ("sentiment_index", "policy_stance_score", "attention_entropy",
                    "industry_count", "policy_count", "event_count"):
            vals = [r[key] for r in rows if r[key] is not None]
            if vals:
                snap[key] = round(sum(vals) / len(vals), 3)
        snap["article_count_total"] = float(sum(r["article_count"] or 0 for r in rows))
    except Exception as e:
        logger.debug(f"metrics snapshot 失败: {e}")
    return snap, dates


# ---------------------------------------------------------------------------
# Back-compat shims: keep the old _retrieve_* names exposed for any
# external callers (server.py imports them only optionally now).
# ---------------------------------------------------------------------------


_SECTION_HEAD_RE = re.compile(r"^##\s+\d+\.\s+", re.MULTILINE)


def _tokenize(text: str) -> str:
    """Back-compat: lexical retriever has its own copy."""
    from src.retrieval.lexical import _tokenize as _t
    return _t(text)


def _chunk_markdown(text: str, max_chars: int = 400) -> List[str]:
    from src.retrieval.lexical import _chunk_markdown as _c
    return _c(text, max_chars)


def _retrieve_reports(query: str, top_k: int = 5) -> List[Citation]:
    """Back-compat shim: returns Citation list using LexicalRetriever."""
    from src.retrieval import LexicalRetriever
    ret = LexicalRetriever()
    return [_hit_to_citation(h) for h in ret.search(query, top_k=top_k)]


def _retrieve_ai_payload(query: str, top_k: int = 3) -> List[Citation]:
    """主题词精确匹配 ai_report.payload_json, 不走 embedding (这是结构化字段匹配)."""
    out: List[Citation] = []
    try:
        import jieba
        tokens = set(t for t in jieba.cut(query) if len(t) > 1)
        with db_mod.get_conn() as conn:
            rows = conn.execute(
                "SELECT date, payload_json FROM ai_report ORDER BY date DESC LIMIT 30"
            ).fetchall()
        for r in rows:
            try:
                p = json.loads(r["payload_json"] or "{}")
            except Exception:
                continue
            kw = (p.get("theme_keywords") or {}).get("keywords", []) or []
            kw_words = {k.get("word", "") for k in kw if isinstance(k, dict)}
            hit = tokens & kw_words
            if not hit:
                continue
            score = min(1.0, len(hit) / max(3, len(tokens)))
            snippet = "主题词: " + ", ".join(list(kw_words)[:6])
            out.append(Citation(
                source=f"ai_report:{r['date']}",
                date=r["date"], score=round(score, 4),
                snippet=snippet,
            ))
        out.sort(key=lambda c: -c.score)
        return out[:top_k]
    except Exception as e:
        logger.debug(f"ai_report retrieval 失败: {e}")
        return []


def _retrieve_embed(query: str, target_date: str = "", top_k: int = 5,
                    router=None) -> List[Citation]:
    """Back-compat shim: uses VectorRetriever instead of deleted EmbedRAG."""
    from src.retrieval import VectorRetriever
    return [_hit_to_citation(h)
            for h in VectorRetriever().search(query, top_k=top_k)]


# ---------------------------------------------------------------------------
# Prompt + LLM
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = (
    "你是中国顶级宏观策略首席,服务大型机构投资人。基于给定的【历史报告片段】+"
    "【AI 主题词命中】+【近30天量化均值】,回答用户的宏观问题。\n"
    "严格要求:\n"
    "1. 答案必须基于提供的证据,绝不编造数字、机构、日期\n"
    "2. 答案不超过 300 字, 用 4~6 句中文回答\n"
    "3. 给出明确的判断 (利多/利空/中性) + 主要依据 + 引用片段编号\n"
    "4. 返回严格 JSON: {\"answer\": \"...\", \"confidence\": 0.0~1.0}\n"
)


def _build_prompt(question: str, citations: List[Citation], snap: Dict[str, float], dates: List[str]) -> str:
    parts = [f"【问题】\n{question}\n"]
    if citations:
        parts.append("【历史报告片段】 (Top-K 检索)")
        for i, c in enumerate(citations, 1):
            parts.append(f"[{i}] {c.source} ({c.date}, sim={c.score:.2f}): {c.snippet}")
    if snap:
        parts.append("【近 30 天量化均值】")
        parts.append(", ".join(f"{k}={v}" for k, v in list(snap.items())[:8]))
        if dates:
            parts.append(f"日期范围: {dates[0]} ~ {dates[-1]} ({len(dates)} 天)")
    parts.append("\n请按 JSON 格式返回 {\"answer\": \"...\", \"confidence\": 0.0~1.0}")
    return "\n\n".join(parts)


def _template_answer(question: str, citations: List[Citation], snap: Dict[str, float]) -> Tuple[str, float]:
    """LLM 不可用时的降级: 用 citations 拼一个简短答案."""
    if not citations:
        answer = (
            f"关于 \"{question}\", 未检索到充分的历史报告证据, "
            f"建议参考近 30 天宏观数据 (情绪={snap.get('sentiment_index', 'N/A')}, "
            f"政策={snap.get('policy_stance_score', 'N/A')})。"
        )
        return answer, 0.2
    top = citations[0]
    answer = (
        f"根据 {len(citations)} 条历史报告片段, 与 \"{question}\" 最相关的是 "
        f"{top.source} (sim={top.score:.2f}):\n"
        f"\"{top.snippet[:120]}\"\n"
        f"近 30 天: 情绪指数 {snap.get('sentiment_index', 'N/A')}, "
        f"政策立场 {snap.get('policy_stance_score', 'N/A')}。"
    )
    return answer, round(min(0.7, top.score), 2)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def ask(question: str, target_date: str = "2026-06-12",
        router: Any = None, top_k: int = 5,
        retrievers: Optional[List] = None) -> QAResult:
    """主入口: 用户问问题 -> 检索 -> 拼装 -> LLM 回答.

    Parameters
    ----------
    retrievers : list[Retriever], optional
        注入自定义 retriever 集合 (方便单测用 mock)。
        None 时按 router 是否可用自动构建。
    """
    question = (question or "").strip()
    if not question:
        return QAResult(question="", answer="问题不能为空", generated_by="template",
                        summary="empty question")

    # 1. 检索: 多个 retriever 并联, 融合排序
    if retrievers is None:
        retrievers = _build_default_retrievers(router=router)
    citations: List[Citation] = []
    for ret in retrievers:
        try:
            for h in ret.search(question, top_k=top_k):
                citations.append(_hit_to_citation(h))
        except Exception as e:
            logger.debug(f"retriever {getattr(ret, 'name', '?')} 失败: {e}")

    # ai_report 主题词召回 (结构化字段, 不走 embedding)
    ai_cits = _retrieve_ai_payload(question, top_k=3)
    citations.extend(ai_cits)

    # 融合: 按 score 降序, 截 top_k
    citations.sort(key=lambda c: -c.score)
    citations = citations[:top_k]

    # 2. 时序
    snap, dates = _metrics_snapshot(target_date)

    # 3. 拼装
    prompt = _build_prompt(question, citations, snap, dates)

    # 4. 调 LLM (如可用)
    answer = ""
    confidence = 0.0
    gen_by = "template"
    model_used = ""
    if router is not None:
        try:
            raw, used, pt, ct = router.chat(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                model=router.default_chain[0] if hasattr(router, "default_chain") else None,
                temperature=0.3, max_tokens=600,
                use_json_mode=True,
            )
            parsed = router.extract_json(raw) or {}
            answer = parsed.get("answer", "").strip()
            confidence = float(parsed.get("confidence", 0.0) or 0.0)
            gen_by = "llm"
            model_used = used
        except Exception as e:
            logger.debug(f"LLM Q&A 失败, 降级: {e}")

    if not answer:
        answer, confidence = _template_answer(question, citations, snap)

    summary = (
        f"问题: {question[:30]}... | 证据: {len(citations)} 条 | "
        f"生成: {gen_by} | 置信: {confidence:.0%}"
    )
    return QAResult(
        question=question, answer=answer, citations=citations,
        context_dates=dates, metrics_snapshot=snap,
        generated_by=gen_by, model=model_used,
        confidence=round(confidence, 3), summary=summary,
    )


if __name__ == "__main__":
    import json
    r = ask("近期降准对新能源板块有什么影响?", target_date="2026-06-12", router=None)
    print(json.dumps(r.as_dict(), ensure_ascii=False, indent=2))
    assert r.answer, "answer 不能为空"
    print("[OK] qa_assistant self-test passed (template mode)")
