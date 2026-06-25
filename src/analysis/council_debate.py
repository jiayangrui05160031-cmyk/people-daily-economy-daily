"""v7 前沿: 多智能体辩论 (Multi-Agent Debate) + LLM-as-a-Judge 仲裁.
Round 1 独立, Round 2 互相阅读后修正立场, LLM Judge 综合。
"""
from __future__ import annotations
import json, re, time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from src.api.multi_agent import (PERSONAS, _load_context, _TEMPLATE_GENERATORS, _llm_opinion, _arbitrate, AgentOpinion)
from src.utils.logger import get_logger
logger = get_logger("analysis.council_debate")

@dataclass
class DebateRound:
    round_no: int
    opinions: List[AgentOpinion] = field(default_factory=list)
    consensus_level: float = 0.0
    elapsed_ms: float = 0.0
    summary: str = ""
    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["opinions"] = [o.as_dict() for o in self.opinions]
        return d

@dataclass
class JudgeVerdict:
    final_stance: str
    final_confidence: float
    consensus: float
    reasoning: str = ""
    dissent: List[str] = field(default_factory=list)
    key_risks: List[str] = field(default_factory=list)
    key_opportunities: List[str] = field(default_factory=list)
    generated_by: str = "template"
    model: str = ""
    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class DebateReport:
    question: str
    date: str
    rounds: List[DebateRound] = field(default_factory=list)
    final: Optional[JudgeVerdict] = None
    stance_evolution: List[str] = field(default_factory=list)
    total_latency_ms: float = 0.0
    used_llm_rounds: int = 0
    used_llm_judge: bool = False
    summary: str = ""
    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["rounds"] = [r.as_dict() for r in self.rounds]
        return d
        return d


def _build_refine_prompt(persona, question, ctx, others):
    metrics = ctx.get("metrics", {})
    events = ctx.get("events", [])[:5]
    industries = ctx.get("industries", [])[:5]
    lines = []
    for o in others:
        lines.append("  - " + o.persona + ": stance=" + o.stance + ", conf=" + format(o.confidence, ".2f"))
        lines.append("    reasoning: " + o.reasoning[:200])
    others_block = "\n".join(lines) if lines else "  (无)"
    return (
        "你是 " + persona["name"] + ", " + persona["stance_hint"] + ".\n"
        "问题: " + question + "\n"
        "日期: " + str(ctx.get("date")) + "\n\n"
        "【量化】sentiment=" + str(metrics.get("sentiment_index", 50))
        + ", policy=" + str(metrics.get("policy_stance_score", 0))
        + ", entropy=" + str(metrics.get("attention_entropy", 0))
        + ", industry=" + str(metrics.get("industry_count", 0)) + "\n"
        "【事件】" + str([(e.get("subject"), e.get("action"), e.get("object")) for e in events]) + "\n"
        "【行业】" + str([(i.get("industry"), i.get("stance")) for i in industries]) + "\n\n"
        "【其他 3 位第 1 轮观点】\n" + others_block + "\n\n"
        "请基于以上证据, 评估其他 3 位的观点, 然后修正你的立场。\n"
        "严格返回 JSON: {\"stance\": \"bullish|bearish|neutral\", \"confidence\": 0.0~1.0, "
        "\"key_points\": [\"3 条以内\"], \"reasoning\": \"2 句以内\", \"agreed_with\": [\"其他 persona\"]}\n"
        "只输出 JSON, 不要其他文字。"
    )

def _llm_refine(persona, question, ctx, others, router):
    if router is None or not getattr(router, "api_key", None):
        return None
    prompt = _build_refine_prompt(persona, question, ctx, others)
    try:
        text, model, pt, ct = router.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=persona["temperature"],
            max_tokens=1200,
            use_json_mode=True,
            disable_thinking=True,
        )
        m = re.search(r"\{[\s\S]*\}", text or "")
        if not m:
            return None
        d = json.loads(m.group(0))
        return AgentOpinion(
            persona=persona["name"], persona_en=persona["name_en"],
            stance=str(d.get("stance", "neutral"))[:8],
            confidence=float(d.get("confidence", 0.5)),
            key_points=[str(x)[:100] for x in d.get("key_points", [])[:3]],
            evidence=[],
            reasoning=str(d.get("reasoning", ""))[:300],
            generated_by="llm",
        )
    except Exception as e:
        logger.debug("refine " + persona["name_en"] + " 失败: " + str(e))
        return None

def _build_judge_prompt(question, date, rounds_list, ctx):
    metrics = ctx.get("metrics", {})
    blocks = []
    for rd in rounds_list:
        lines = ["=== 第 " + str(rd.round_no) + " 轮 (共识度 " + format(rd.consensus_level, ".0%") + ") ==="]
        for o in rd.opinions:
            lines.append("  - " + o.persona + ": stance=" + o.stance + ", conf=" + format(o.confidence, ".2f"))
            lines.append("    reasoning: " + o.reasoning[:150])
        blocks.append("\n".join(lines))
    return (
        "你是资深宏观策略首席, 主持 4 位顾问 (鹰/鸽/量化/务实) 的辩论会。\n"
        "评判标准: 证据强度 > 立场一致性 > 量化指标背书。\n\n"
        "问题: " + question + "\n"
        "日期: " + str(date) + "\n\n"
        "【量化】sentiment=" + str(metrics.get("sentiment_index", 50))
        + ", policy=" + str(metrics.get("policy_stance_score", 0))
        + ", entropy=" + str(metrics.get("attention_entropy", 0))
        + ", industry=" + str(metrics.get("industry_count", 0)) + "\n\n"
        "【辩论记录】\n" + "\n".join(blocks) + "\n\n"
        "请综合判断, 严格返回 JSON:\n"
        "{\n"
        "  \"final_stance\": \"bullish|bearish|neutral\",\n"
        "  \"final_confidence\": 0.0~1.0,\n"
        "  \"consensus\": 0.0~1.0,\n"
        "  \"reasoning\": \"3 句以内, 引用具体证据\",\n"
        "  \"dissent\": [\"持反对意见的 persona 及原因\"],\n"
        "  \"key_risks\": [\"2-3 条核心风险\"],\n"
        "  \"key_opportunities\": [\"2-3 条核心机会\"]\n"
        "}\n只输出 JSON, 不要其他文字。"
    )

def _llm_judge(question, date, rounds_list, ctx, router):
    if router is None or not getattr(router, "api_key", None):
        return None
    prompt = _build_judge_prompt(question, date, rounds_list, ctx)
    try:
        text, model, pt, ct = router.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1500,
            use_json_mode=True,
            disable_thinking=True,
        )
        m = re.search(r"\{[\s\S]*\}", text or "")
        if not m:
            return None
        d = json.loads(m.group(0))
        return JudgeVerdict(
            final_stance=str(d.get("final_stance", "neutral"))[:8],
            final_confidence=float(d.get("final_confidence", 0.5)),
            consensus=float(d.get("consensus", 0.5)),
            reasoning=str(d.get("reasoning", ""))[:400],
            dissent=[str(x)[:100] for x in d.get("dissent", [])[:3]],
            key_risks=[str(x)[:100] for x in d.get("key_risks", [])[:3]],
            key_opportunities=[str(x)[:100] for x in d.get("key_opportunities", [])[:3]],
            generated_by="llm",
            model=model,
        )
    except Exception as e:
        logger.debug("LLM judge 失败: " + str(e))
        return None

def _template_judge(rounds_list):
    if not rounds_list:
        return JudgeVerdict(final_stance="neutral", final_confidence=0.0, consensus=0.0,
                            reasoning="无辩论记录", generated_by="template")
    last = rounds_list[-1]
    final, conf, consensus = _arbitrate(last.opinions)
    risks, opps = [], []
    for o in last.opinions:
        if o.stance == "bearish":
            for kp in o.key_points[:1]:
                if kp and kp not in risks:
                    risks.append(kp)
        elif o.stance == "bullish":
            for kp in o.key_points[:1]:
                if kp and kp not in opps:
                    opps.append(kp)
    return JudgeVerdict(
        final_stance=final, final_confidence=conf, consensus=consensus,
        reasoning="4 位顾问第 " + str(last.round_no) + " 轮加权: " + final
                  + " (置信 " + format(conf, ".0%") + ", 共识 " + format(consensus, ".0%") + ")",
        dissent=[o.persona + ": " + o.stance for o in last.opinions if o.stance != final],
        key_risks=risks[:3], key_opportunities=opps[:3],
        generated_by="template",
    )
def debate(question, target_date, router=None, rounds=2, use_llm=True, use_llm_judge=True):
    """多智能体辩论主入口. 默认 2 轮 + LLM Judge."""
    t0 = time.time()
    ctx = _load_context(target_date)
    rep = DebateReport(question=question, date=target_date)
    stance_evo = []
    used_llm_rounds = 0

    rd1_t0 = time.time()
    rd1_opinions = []
    for persona in PERSONAS:
        op = None
        if use_llm:
            op = _llm_opinion(persona, question, ctx, router)
            if op:
                used_llm_rounds += 1
        if op is None:
            op = _TEMPLATE_GENERATORS[persona["name_en"]](ctx)
        rd1_opinions.append(op)
    final1, conf1, cons1 = _arbitrate(rd1_opinions)
    stance_evo.append(final1)
    rd1 = DebateRound(
        round_no=1, opinions=rd1_opinions,
        consensus_level=cons1, elapsed_ms=round((time.time()-rd1_t0)*1000, 1),
        summary="第 1 轮: 立场 " + final1 + ", 共识 " + format(cons1, ".0%") + ", 置信 " + format(conf1, ".0%"),
    )
    rep.rounds.append(rd1)

    if rounds >= 2:
        rd2_t0 = time.time()
        rd2_opinions = []
        for i, persona in enumerate(PERSONAS):
            others = [o for j, o in enumerate(rd1_opinions) if j != i]
            op = None
            if use_llm:
                op = _llm_refine(persona, question, ctx, others, router)
                if op:
                    used_llm_rounds += 1
            if op is None:
                op = rd1_opinions[i]
            rd2_opinions.append(op)
        final2, conf2, cons2 = _arbitrate(rd2_opinions)
        stance_evo.append(final2)
        rd2 = DebateRound(
            round_no=2, opinions=rd2_opinions,
            consensus_level=cons2, elapsed_ms=round((time.time()-rd2_t0)*1000, 1),
            summary="第 2 轮 (cross-read): 立场 " + final2 + ", 共识 " + format(cons2, ".0%") + ", 置信 " + format(conf2, ".0%"),
        )
        rep.rounds.append(rd2)

    used_judge = False
    verdict = None
    if use_llm_judge and use_llm:
        verdict = _llm_judge(question, target_date, rep.rounds, ctx, router)
        if verdict:
            used_judge = True
    if verdict is None:
        verdict = _template_judge(rep.rounds)

    rep.final = verdict
    rep.stance_evolution = stance_evo
    rep.used_llm_rounds = used_llm_rounds
    rep.used_llm_judge = used_judge
    rep.total_latency_ms = round((time.time() - t0) * 1000, 1)
    rep.summary = (
        str(len(rep.rounds)) + " 轮辩论, LLM 调用 " + str(used_llm_rounds) + " 次"
        + (" + Judge LLM" if used_judge else " (模板 judge)") + "."
        " 最终 " + verdict.final_stance + " (置信 " + format(verdict.final_confidence, ".0%")
        + ", 共识 " + format(verdict.consensus, ".0%") + "), 用时 " + format(rep.total_latency_ms/1000, ".1f") + "s"
    )
    return rep
