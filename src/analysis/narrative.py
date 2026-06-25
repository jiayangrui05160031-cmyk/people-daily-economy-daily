"""analysis.narrative - LLM 高管摘要自动生成

把当日 9 个前沿分析模块 + AI 维度结果汇总成一个结构化的"高管简报",
由 LLM (miniMax-M3) 输出 5 段结构化 Markdown, 适合作为报告开篇.

5 段结构:
1. 宏观画像 (一句话总结当前宏观状态)
2. 政策与产业风向 (政策 + 主导产业 + AI 判断)
3. 风险与机会 (异常 + 波动率 + 事件)
4. 操作建议 (决策引擎 BUY/HOLD/REDUCE/SELL + 理由)
5. 展望 (预测 + RAG 历史相似)

LLM 失败时降级为模板拼接 (不抛错, 保证报告总能生成).

依赖: src.ai.router (OpenAI 兼容协议)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from datetime import timedelta
from typing import Any, Dict, List, Optional

from src.utils.date_utils import parse_date
from src.utils.logger import get_logger

logger = get_logger("analysis.narrative")


@dataclass
class NarrativeSection:
    heading: str
    body: str

    def as_dict(self):
        return asdict(self)


@dataclass
class NarrativeReport:
    date: str
    title: str
    sections: List[NarrativeSection]
    word_count: int
    generated_by: str  # "llm" | "template"
    model: str = ""
    summary: str = ""

    def as_dict(self):
        # asdict 已递归序列化 dataclass, 不需要再调子对象 as_dict
        return asdict(self)


# ============================================================
# Prompt 构造
# ============================================================
SYSTEM_PROMPT = (
    "你是中国顶级宏观策略首席,服务大型机构投资人。你的写作要求:\n"
    "1. 严格基于给定 JSON 数据,绝不编造指标、数字、机构名\n"
    "2. 措辞简练、句式稳定,每段 100~180 字\n"
    "3. 多空判断明确,不模棱两可\n"
    "4. 输出严格 JSON: {\"sections\": [{\"heading\": \"...\", \"body\": \"...\"}, ...]}\n"
    "5. 5 个 section 标题依次固定为:\n"
    "   - 宏观画像\n"
    "   - 政策与产业风向\n"
    "   - 风险与机会\n"
    "   - 操作建议\n"
    "   - 展望\n"
)


def _safe(obj, *path, default="N/A"):
    cur = obj
    for p in path:
        if cur is None:
            return default
        cur = cur.get(p) if isinstance(cur, dict) else getattr(cur, p, None)
        if cur is None:
            return default
    return cur


def _build_context(advanced: Dict[str, Any], ai_result=None) -> Dict[str, Any]:
    """把 9 模块的输出压平成 prompt 友好的 JSON."""
    anom = advanced.get("anomaly")
    fc = advanced.get("forecast")
    vol = advanced.get("volatility")
    mkt = advanced.get("market")
    macro = advanced.get("macro")
    topics = advanced.get("topics")
    es = advanced.get("events_study")
    rag = advanced.get("rag")
    signal = advanced.get("signal")
    backtest = advanced.get("backtest")
    return {
        "date": _safe(anom, "date", default=""),
        "macro_picture": {
            "sentiment_index": _safe(anom, "current_sentiment", default=None),
            "policy_stance": _safe(ai_result, "policy_direction", "direction", default=None),
            "policy_confidence": _safe(ai_result, "policy_direction", "confidence", default=None),
            "top_industries": [
                {"name": i.name, "stance": _safe(i, "stance", "value", default=str(i.stance)),
                 "heat": _safe(i, "heat", "value", default=str(i.heat))}
                for i in (_safe(ai_result, "industries", "industries", default=[]) or [])[:5]
            ],
            "theme_keywords": [
                k.word for k in (_safe(ai_result, "theme_keywords", "keywords", default=[]) or [])[:8]
            ],
            "anomaly_risk": _safe(anom, "overall_risk", default=None),
            "anomaly_signals_n": len(_safe(anom, "signals", default=[]) or []),
            "panic_index": _safe(vol, "index", default=None),
            "panic_level": _safe(vol, "level", default=None),
            "decision": {
                "action": _safe(signal, "action", default="HOLD"),
                "score": _safe(signal, "score", default=None),
                "confidence": _safe(signal, "confidence", default=None),
            },
        },
        "policy_industry": {
            "core_insight": _safe(ai_result, "core_insights", "insights", default=""),
            "policies": [
                {"title": _safe(p, "title", default=""), "stance": _safe(p, "stance", default="")}
                for p in (_safe(ai_result, "policies", "policies", default=[]) or [])[:3]
            ],
            "topics": [
                {"label": _safe(t, "label", default=""),
                 "share": _safe(t, "dominance", default=None),
                 "industry": _safe(t, "dominant_industry", default=None)}
                for t in (_safe(topics, "topics", default=[]) or [])[:5]
            ],
            "market_overall": _safe(mkt, "market_overall", default=None),
            "macro_indicators": [
                {"name": _safe(i, "name", default=""), "value": _safe(i, "value", default=None),
                 "previous": _safe(i, "previous", default=None)}
                for i in (_safe(macro, "indicators", default=[]) or [])[:5]
            ],
        },
        "risk_opportunity": {
            "anomaly_signals": [
                {"metric": _safe(s, "metric", default=""),
                 "z_score": _safe(s, "z_score", default=None),
                 "direction": _safe(s, "direction", default="")}
                for s in (_safe(anom, "signals", default=[]) or [])[:5]
            ],
            "panic_index": _safe(vol, "index", default=None),
            "panic_level": _safe(vol, "level", default=None),
            "top_events": [
                {"title": _safe(e, "title", default=_safe(e, "action", default="")),
                 "impact": _safe(e, "impact_level", default=None),
                 "industry": _safe(e, "industry", default="")}
                for e in (_safe(es, "events", default=[]) or [])[:3]
            ],
            "backtest_accuracy": _safe(backtest, "horizon_reports", "0", "direction_accuracy", default=None)
                if _safe(backtest, "horizon_reports") else None,
        },
        "actionable": {
            "decision": {
                "action": _safe(signal, "action", default="HOLD"),
                "score": _safe(signal, "score", default=None),
                "confidence": _safe(signal, "confidence", default=None),
            },
            "top_reasons": (_safe(signal, "top_reasons", default=[]) or [])[:5],
            "risks": (_safe(signal, "risks", default=[]) or [])[:3],
            "industry_signals": [
                {"industry": _safe(i, "industry", default=""),
                 "action": _safe(i, "action", default=""),
                 "score": _safe(i, "score", default=None)}
                for i in (_safe(signal, "industry_signals", default=[]) or [])[:5]
            ],
        },
        "outlook": {
            "headline": _safe(fc, "headline", default=""),
            "predicted_sentiment": _safe(fc, "predicted_sentiment", default=None),
            "current_sentiment": _safe(fc, "current_sentiment", default=None),
            "rag_recall_top": [
                {"title": _safe(r, "title", default=""),
                 "date": _safe(r, "date", default=""),
                 "score": _safe(r, "score", default=None)}
                for r in (_safe(rag, "recalls", default=[]) or [])[:3]
            ],
        },
    }


# ============================================================
# 调用 LLM
# ============================================================
def _call_llm(context: Dict[str, Any], router) -> tuple[Optional[List[Dict]], str, str]:
    """调用 LLM, 返回 (sections, model, raw)."""
    if router is None:
        return None, "", ""
    user_prompt = (
        "请基于以下结构化数据生成 5 段中文高管简报 (每段 100~180 字):\n\n"
        "```json\n" + json.dumps(context, ensure_ascii=False, indent=2)[:8000] + "\n```\n\n"
        "输出严格 JSON: {\"sections\": [{\"heading\":\"...\",\"body\":\"...\"},...]}, "
        "5 个 section 标题严格按系统提示。\n"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        raw = router.chat_json(
            messages, task_name="narrative", date=context.get("date", ""),
            temperature=0.4, max_tokens=2500,
        )
        sections = raw.get("sections") if isinstance(raw, dict) else None
        return sections, "MiniMax-M3", json.dumps(raw, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"narrative LLM 失败: {e}")
        return None, "", ""


# ============================================================
# 模板兜底 (LLM 不可用时)
# ============================================================
def _template_narrative(context: Dict[str, Any]) -> List[NarrativeSection]:
    m = context.get("macro_picture", {})
    p = context.get("policy_industry", {})
    r = context.get("risk_opportunity", {})
    a = context.get("actionable", {})
    o = context.get("outlook", {})

    sentiment = m.get("sentiment_index")
    sent_str = f"{sentiment:.1f}" if isinstance(sentiment, (int, float)) else "N/A"
    pol = m.get("policy_stance") or "中性"
    inds = m.get("top_industries", []) or []
    ind_str = "、".join(i["name"] for i in inds[:3]) if inds else "暂未识别"
    themes = "、".join(m.get("theme_keywords", [])[:5])

    body1 = (
        f"今日宏观情绪指数 {sent_str}。政策风向判定为「{pol}」"
        f"{'(' + format(m.get('policy_confidence') or 0, '.0%') + ')' if m.get('policy_confidence') else ''}。"
        f"主题词集中在 {themes or '暂未提取'}, 主导产业为 {ind_str}。"
        f"综合决策信号: {a.get('decision', {}).get('action', 'HOLD')} "
        f"(分数 {a.get('decision', {}).get('score', 0):+.2f}, 置信 {a.get('decision', {}).get('confidence', 0):.0%})。"
    )

    body2 = (
        f"AI 核心洞察: {p.get('core_insight') or 'N/A'}\n\n"
        f"市场联动: {p.get('market_overall') or 'N/A'}。"
        f"近期主题包括 {', '.join(t.get('label', '?') for t in p.get('topics', [])[:3]) or '暂未提取'}。"
    )

    risk_lines = []
    for s in r.get("anomaly_signals", [])[:3]:
        risk_lines.append(f"- {s.get('metric', '')} Z={s.get('z_score', '?')} ({s.get('direction', '')})")
    panic = r.get("panic_index")
    body3 = (
        f"异常检测触发 {len(r.get('anomaly_signals', []))} 项:\n"
        + ("\n".join(risk_lines) or "- (无)")
        + f"\n\n恐慌指数 {panic if panic is not None else 'N/A'} ({r.get('panic_level', 'N/A')}),"
        f" 历史回测方向准确率 {r.get('backtest_accuracy', 'N/A') if r.get('backtest_accuracy') is not None else 'N/A'}。"
    )

    body4 = (
        f"建议动作: **{a.get('decision', {}).get('action', 'HOLD')}**\n\n"
        f"主要依据: {'; '.join(a.get('top_reasons', [])[:3]) or '(无)'}\n\n"
        f"风险点: {'; '.join(a.get('risks', [])[:3]) or '(无)'}\n\n"
        f"行业视角: "
        + (", ".join(f"{i['industry']}={i['action']}" for i in a.get('industry_signals', [])[:4]) or "(无)")
    )

    body5 = (
        f"预测: {o.get('headline') or 'N/A'} "
        f"(明日情绪 {o.get('predicted_sentiment', '?')}, 今日 {o.get('current_sentiment', '?')})\n\n"
        f"历史相似情境: "
        + (", ".join(f"{r.get('title', '?')}@{r.get('date', '?')} (相似度 {r.get('score', '?')})"
                     for r in o.get('rag_recall_top', [])[:3]) or "(暂无)")
    )

    return [
        NarrativeSection("宏观画像", body1),
        NarrativeSection("政策与产业风向", body2),
        NarrativeSection("风险与机会", body3),
        NarrativeSection("操作建议", body4),
        NarrativeSection("展望", body5),
    ]


def generate(target_date: str, advanced: Dict[str, Any],
             ai_result=None, router=None) -> NarrativeReport:
    """生成高管简报."""
    context = _build_context(advanced, ai_result=ai_result)
    sections_raw, model, raw = _call_llm(context, router)
    if sections_raw and isinstance(sections_raw, list) and len(sections_raw) >= 3:
        sections = [
            NarrativeSection(
                heading=str(s.get("heading", ""))[:30],
                body=str(s.get("body", ""))[:600],
            )
            for s in sections_raw[:6]
        ]
        wc = sum(len(s.body) for s in sections)
        return NarrativeReport(
            date=target_date,
            title=f"宏观策略简报 · {target_date}",
            sections=sections,
            word_count=wc,
            generated_by="llm",
            model=model,
            summary=sections[0].body[:120] if sections else "",
        )
    # 兜底模板
    sections = _template_narrative(context)
    wc = sum(len(s.body) for s in sections)
    return NarrativeReport(
        date=target_date,
        title=f"宏观策略简报 · {target_date}",
        sections=sections,
        word_count=wc,
        generated_by="template",
        model="",
        summary=sections[0].body[:120],
    )


if __name__ == "__main__":
    # 自检: 构造伪数据, 跑模板路径
    class _A:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)
    fake_advanced = {
        "anomaly": _A(date="2026-06-12", current_sentiment=62.0, overall_risk="low",
                      signals=[_A(**{"metric":"sentiment_index","z_score":2.1,"direction":"up"})]),
        "forecast": _A(headline="明日偏多", predicted_sentiment=66.0, current_sentiment=62.0),
        "volatility": _A(index=28.0, level="calm"),
        "market": _A(market_overall="bullish"),
        "macro": _A(indicators=[
            _A(**{"name":"PMI","value":50.5,"previous":49.8}),
            _A(**{"name":"CPI","value":0.4,"previous":0.3}),
        ]),
        "topics": _A(topics=[
            _A(**{"label":"新能源","dominance":0.42,"dominant_industry":"新能源"}),
            _A(**{"label":"AI 与算力","dominance":0.31,"dominant_industry":"人工智能"}),
        ]),
        "events_study": _A(events=[
            _A(**{"title":"央行降准","impact_level":5,"industry":"金融"}),
        ]),
        "rag": _A(recalls=[
            _A(**{"title":"2025-Q4 复盘","date":"2026-03-15","score":0.82}),
        ]),
        "signal": _A(action="BUY", score=0.42, confidence=0.62,
                     top_reasons=["政策扩张","AI 联动看多"],
                     risks=["PMI 仍在荣枯线附近"],
                     industry_signals=[_A(**{"industry":"新能源","action":"关注","score":0.7})]),
        "backtest": _A(horizon_reports=[_A(horizon=1, direction_accuracy=0.81)]),
    }
    fake_ai = _A(**{
        "policy_direction": _A(direction="扩张", confidence=0.75),
        "industries": _A(industries=[_A(name="新能源", heat=_A(value="高"), stance=_A(value="利好"))]),
        "theme_keywords": _A(keywords=[_A(word="新能源"), _A(word="AI")]),
        "core_insights": _A(insights="今日新能源板块持续走高。"),
        "policies": _A(policies=[_A(title="央行降准", stance="利好")]),
    })
    import json
    rep = generate("2026-06-12", fake_advanced, ai_result=fake_ai, router=None)
    print(json.dumps(rep.as_dict(), ensure_ascii=False, indent=2))
    assert len(rep.sections) == 5
    assert rep.generated_by in ("llm", "template")
    print("[OK] narrative self-test passed (template mode)")
