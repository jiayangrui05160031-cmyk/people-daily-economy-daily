"""api.streaming - LLM 实时流式输出 (SSE)

把 LLM 生成的 token 流式推给客户端 (Server-Sent Events),
让用户能"边生成边看到", 体验类似 ChatGPT.

典型用例:
  GET /v6/stream/qa?question=...&target_date=...  -> SSE 流
  GET /v6/stream/council?question=...  -> 多智能体 SSE 流

SSE 协议:
  - Content-Type: text/event-stream
  - 帧格式: data: <json>\\n\\n
  - 客户端用 EventSource 自动重连
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger("api.streaming")


def sse_format(event: str, data: Any) -> str:
    """格式化 SSE 帧."""
    if not isinstance(data, str):
        data = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {data}\n\n"


async def stream_qa_answer(question: str, target_date: str,
                            router=None, top_k: int = 5) -> AsyncIterator[str]:
    """流式生成 QA 答案: 先 RAG 检索, 再 LLM 流式输出."""
    from src.analysis.qa_assistant import (
        _retrieve_reports, _retrieve_ai_payload, _build_prompt, _template_answer,
    )
    from src.analysis.qa_assistant import _metrics_snapshot as _q_metrics_snapshot
    # _metrics_snapshot 需要 (date, context_dict) 格式, 我们构造一个最小 context
    def _snapshot_for_date(date: str) -> dict:
        try:
            from src.storage import db as db_mod
            from datetime import timedelta
            from src.utils.date_utils import parse_date
            end = parse_date(date)
            start = end - timedelta(days=30)
            with db_mod.get_conn() as conn:
                rows = conn.execute(
                    "SELECT sentiment_index, policy_stance_score, industry_count, attention_entropy "
                    "FROM daily_metric WHERE date BETWEEN ? AND ? ORDER BY date",
                    (start.isoformat(), end.isoformat()),
                ).fetchall()
            if not rows:
                return {}
            import statistics
            s = statistics.mean([float(r["sentiment_index"] or 50) for r in rows])
            p = statistics.mean([float(r["policy_stance_score"] or 0) for r in rows])
            ind = statistics.mean([float(r["industry_count"] or 0) for r in rows])
            ent = statistics.mean([float(r["attention_entropy"] or 0) for r in rows])
            return {
                "sentiment_index": round(s, 2),
                "policy_stance_score": round(p, 3),
                "industry_count": round(ind, 2),
                "attention_entropy": round(ent, 3),
                "n_days": len(rows),
            }
        except Exception:
            return {}

    t0 = time.time()
    yield sse_format("start", {"question": question, "date": target_date, "ts": t0})

    # 1. 检索证据
    yield sse_format("step", {"stage": "retrieval", "msg": "正在检索历史报告..."})
    citations = _retrieve_reports(question, top_k=top_k)
    citations += _retrieve_ai_payload(question, top_k=3)
    yield sse_format("citations", {
        "count": len(citations),
        "items": [
            {"source": c.source, "date": c.date, "score": c.score, "snippet": c.snippet[:100]}
            for c in citations[:5]
        ],
    })

    snapshot = _snapshot_for_date(target_date)

    # 2. LLM 流式
    if router is not None and getattr(router, "api_key", None):
        yield sse_format("step", {"stage": "llm", "msg": "LLM 流式生成中..."})
        prompt = _build_prompt(question, citations, snapshot, target_date)
        try:
            from src.config import AI_BASE_URL
            from openai import OpenAI
            client = OpenAI(api_key=router.api_key, base_url=AI_BASE_URL)
            stream = client.chat.completions.create(
                model=router.chain[0].name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=1500, stream=True,
                extra_body={"thinking": {"type": "disabled"}},
            )
            full = []
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    full.append(delta)
                    yield sse_format("token", {"text": delta})
            yield sse_format("done", {
                "answer": "".join(full),
                "elapsed_ms": round((time.time() - t0) * 1000, 1),
                "model": router.chain[0].name,
                "generated_by": "llm",
            })
            return
        except Exception as e:
            yield sse_format("error", {"msg": f"LLM 流式失败: {e}"})

    # 3. 模板兜底
    yield sse_format("step", {"stage": "fallback", "msg": "LLM 不可用, 模板降级"})
    if citations:
        ans = f"根据 {len(citations)} 条相关证据:\n" + \
              "\n".join(f"- [{c.source}] {c.snippet[:120]}" for c in citations[:3])
    else:
        ans = (
            f"未检索到 {target_date} 前后 30 天的强相关证据。"
            f"近 30 天: 情绪指数 {snapshot.get('sentiment_index', 50):.2f}, "
            f"政策立场 {snapshot.get('policy_stance_score', 0):+.2f}。"
        )
    # 模拟流式 (按字 chunk)
    for i in range(0, len(ans), 8):
        yield sse_format("token", {"text": ans[i:i+8]})
        await asyncio.sleep(0.02)
    yield sse_format("done", {
        "answer": ans,
        "elapsed_ms": round((time.time() - t0) * 1000, 1),
        "model": "template",
        "generated_by": "template",
    })


async def stream_council(question: str, target_date: str,
                          router=None, use_llm: bool = True,
                          personas: Optional[List[str]] = None) -> AsyncIterator[str]:
    """流式生成多智能体顾问团意见."""
    from src.api.multi_agent import (
        _load_context, _TEMPLATE_GENERATORS, _arbitrate,
        _synthesize, PERSONAS,
    )

    t0 = time.time()
    yield sse_format("start", {"question": question, "date": target_date})
    ctx = _load_context(target_date)

    selected = PERSONAS
    if personas:
        selected = [p for p in PERSONAS if p["name_en"] in personas] or PERSONAS

    opinions = []
    for persona in selected:
        yield sse_format("persona_start", {"name": persona["name"], "icon": persona["icon"]})
        op = None
        if use_llm and router is not None:
            # LLM 模式: 也走流式 (简化: 一段一段吐)
            from src.api.multi_agent import _llm_opinion
            op = _llm_opinion(persona, question, ctx, router)
        if op is None:
            op = _TEMPLATE_GENERATORS[persona["name_en"]](ctx)
            # 流式模拟: 按行吐
            for line in [op.persona + " 立场: " + op.stance,
                         "置信度: " + format(op.confidence, ".0%"),
                         "推理: " + op.reasoning] + ["- " + p for p in op.key_points]:
                yield sse_format("token", {"persona": persona["name_en"], "text": line})
                await asyncio.sleep(0.05)
        else:
            yield sse_format("token", {
                "persona": persona["name_en"],
                "text": op.persona + ": " + op.stance + " (置信 " + format(op.confidence, ".0%") + ")",
            })
        opinions.append(op)
        yield sse_format("persona_done", {
            "name": persona["name"], "stance": op.stance,
            "confidence": op.confidence, "generated_by": op.generated_by,
        })

    final, conf, consensus = _arbitrate(opinions)
    synthesis = _synthesize(question, opinions, final, conf, consensus)
    yield sse_format("synthesis", {
        "final_stance": final, "confidence": conf, "consensus": consensus,
        "text": synthesis,
    })
    yield sse_format("done", {
        "elapsed_ms": round((time.time() - t0) * 1000, 1),
        "n_opinions": len(opinions),
    })


async def stream_health_ticks(interval: float = 2.0) -> AsyncIterator[str]:
    """流式健康心跳 (供前端做实时面板)."""
    from src.storage import db as db_mod
    while True:
        try:
            with db_mod.get_conn() as conn:
                row = conn.execute(
                    "SELECT MAX(date) AS d, COUNT(*) AS n "
                    "FROM daily_metric"
                ).fetchone()
            yield sse_format("tick", {
                "ts": time.time(),
                "last_date": row["d"], "n_rows": row["n"],
            })
        except Exception as e:
            yield sse_format("error", {"msg": str(e)})
        await asyncio.sleep(interval)
