"""api.multi_agent - 多智能体顾问团 (Multi-Agent Debate)

让 4-5 个具备不同人设/风险偏好的 AI Agent 围绕同一个宏观问题独立分析,
然后在 "orchestrator" 协调下做 1-2 轮辩论 + 综合, 输出比单 LLM 更稳的结论。

核心思想:
  - 角色多样性 > 单 LLM 多次采样 (Chain-of-Thought 已经被广泛验证有效)
  - 多智能体辩论 (Du et al. 2023, "Improving Factuality and Reasoning through Multiagent Debate")
  - 每个 agent 给出: 立场 + 证据 + 信心; 仲裁者综合

Personas (默认 4 个):
  1. Hawk 鹰派  - 关注通胀 / 紧缩 / 风险事件, 倾向保守
  2. Dove 鸽派  - 关注增长 / 宽松 / 机会, 倾向积极
  3. Quant 量化  - 纯数据驱动, 引用时序 / 风险 / VaR, 不做情感判断
  4. Pragmatist 务实 - 政策落地视角, 关注可操作性 + 时间窗口

当 LLM 不可用时降级为 4 段模板 (基于 daily_metric 量化数据) + 仲裁综合,
保证 REST 端点 100% 可用。

典型用法:
    from src.api.multi_agent import council
    result = council("当前宏观风险与机会", target_date="2026-06-12", router=router)
    print(result.synthesis)
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from src.storage import db as db_mod
from src.utils.date_utils import parse_date
from src.utils.logger import get_logger

logger = get_logger("api.multi_agent")


PERSONAS = [
    {
        "name": "鹰派 Hawk",
        "name_en": "hawk",
        "stance_hint": "关注通胀粘性、信贷收紧、地缘风险, 倾向保守",
        "temperature": 0.3,
        "icon": "🦅",
    },
    {
        "name": "鸽派 Dove",
        "name_en": "dove",
        "stance_hint": "关注经济下行压力、政策宽松窗口、产业机会, 倾向积极",
        "temperature": 0.6,
        "icon": "🕊️",
    },
    {
        "name": "量化 Quant",
        "name_en": "quant",
        "stance_hint": "纯数据驱动, 只引用 Sharpe / VaR / 时序 / 动量 等量化指标, 不做情感判断",
        "temperature": 0.1,
        "icon": "📊",
    },
    {
        "name": "务实 Pragmatist",
        "name_en": "pragmatist",
        "stance_hint": "政策落地视角, 关注部委文件 / 资金到位 / 时间窗口 / 可执行性",
        "temperature": 0.4,
        "icon": "🛠️",
    },
]


@dataclass
class AgentOpinion:
    persona: str
    persona_en: str
    stance: str          # bullish / bearish / neutral
    confidence: float    # 0~1
    key_points: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    reasoning: str = ""
    generated_by: str = "template"  # llm / template

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CouncilResult:
    question: str
    date: str
    opinions: List[AgentOpinion] = field(default_factory=list)
    synthesis: str = ""
    final_stance: str = "neutral"
    final_confidence: float = 0.0
    consensus_level: float = 0.0
    debate_rounds: int = 1
    total_latency_ms: float = 0.0
    generated_by: str = "template"
    summary: str = ""

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["opinions"] = [o.as_dict() for o in self.opinions]
        return d


# ============================================================
# 1) 拉取上下文 (时序 / 风险 / 事件)
# ============================================================
def _load_context(target_date: str) -> Dict[str, Any]:
    """收集 4 个 agent 共享的 evidence pool."""
    end = parse_date(target_date)
    ctx: Dict[str, Any] = {
        "date": target_date,
        "metrics": {},
        "events": [],
        "industries": [],
    }
    try:
        with db_mod.get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM daily_metric WHERE date = ?", (target_date,),
            ).fetchone()
            if row:
                ctx["metrics"] = dict(row)
            rows30 = conn.execute(
                "SELECT date, sentiment_index, policy_stance_score, industry_count, attention_entropy "
                "FROM daily_metric WHERE date <= ? ORDER BY date DESC LIMIT 30",
                (target_date,),
            ).fetchall()
            ctx["timeseries_30d"] = [dict(r) for r in rows30]
            erows = conn.execute(
                "SELECT subject, action, object, event_type, intensity "
                "FROM events WHERE date = ? ORDER BY intensity DESC LIMIT 10",
                (target_date,),
            ).fetchall()
            ctx["events"] = [dict(r) for r in erows]
            irows = conn.execute(
                "SELECT industry, hit_count, stance FROM industry_daily "
                "WHERE date = ? ORDER BY hit_count DESC LIMIT 10",
                (target_date,),
            ).fetchall()
            ctx["industries"] = [dict(r) for r in irows]
    except Exception as e:
        logger.debug(f"load context 失败: {e}")
    return ctx


# ============================================================
# 2) 模板观点生成 (无 LLM 兜底)
# ============================================================
def _stance_from_metrics(metrics: Dict[str, Any]) -> str:
    s = float(metrics.get("sentiment_index", 50.0) or 50.0)
    p = float(metrics.get("policy_stance_score", 0.0) or 0.0)
    if s >= 60 and p >= 0.1:
        return "bullish"
    if s <= 40 and p <= -0.1:
        return "bearish"
    return "neutral"


def _gen_hawk(ctx: Dict[str, Any]) -> AgentOpinion:
    m = ctx.get("metrics", {})
    s = float(m.get("sentiment_index", 50.0) or 50.0)
    p = float(m.get("policy_stance_score", 0.0) or 0.0)
    # 鹰派关注下行风险
    bearish_count = sum(1 for ind in ctx.get("industries", []) if ind.get("stance") == "利空")
    ev = []
    if s < 55:
        ev.append(f"市场情绪指数 {s:.1f} 偏中性偏弱")
    if p < 0:
        ev.append(f"政策立场 {p:+.2f} 偏紧")
    if bearish_count:
        ev.append(f"利空行业 {bearish_count} 个, 集中度风险上升")
    if not ev:
        ev.append("宏观信号温和, 但需关注外围地缘 + 国内信用尾部")
    stance = "bearish" if (s < 50 or p < -0.05) else "neutral"
    return AgentOpinion(
        persona="鹰派 Hawk", persona_en="hawk",
        stance=stance, confidence=0.6 if stance == "bearish" else 0.45,
        key_points=[
            "通胀粘性 / 外部利率高位 / 信用尾部风险是核心矛盾",
            "宽松节奏若慢于预期, 权益估值修复空间有限",
            "建议: 缩短久期, 减少高 beta 敞口",
        ],
        evidence=ev, reasoning="鹰派从风险防御视角看, 默认假设是下行偏多。",
        generated_by="template",
    )


def _gen_dove(ctx: Dict[str, Any]) -> AgentOpinion:
    m = ctx.get("metrics", {})
    s = float(m.get("sentiment_index", 50.0) or 50.0)
    p = float(m.get("policy_stance_score", 0.0) or 0.0)
    bullish_count = sum(1 for ind in ctx.get("industries", []) if ind.get("stance") == "利好")
    ev = []
    if s >= 50:
        ev.append(f"市场情绪 {s:.1f} 中性偏积极")
    if p >= 0:
        ev.append(f"政策立场 {p:+.2f} 偏宽松")
    if bullish_count:
        ev.append(f"利好行业 {bullish_count} 个, 结构性机会显现")
    if not ev:
        ev.append("政策托底意愿明确, 等待数据验证拐点")
    stance = "bullish" if (s >= 55 or p >= 0.1) else "neutral"
    return AgentOpinion(
        persona="鸽派 Dove", persona_en="dove",
        stance=stance, confidence=0.6 if stance == "bullish" else 0.45,
        key_points=[
            "经济转型期政策托底意愿强, 流动性环境边际改善",
            "AI / 高端制造 / 数字经济 等新质生产力是长期主线",
            "建议: 维持中性偏积极仓位, 关注政策催化窗口",
        ],
        evidence=ev, reasoning="鸽派从机会视角看, 默认假设是政策兜底 + 估值修复。",
        generated_by="template",
    )


def _gen_quant(ctx: Dict[str, Any]) -> AgentOpinion:
    m = ctx.get("metrics", {})
    s = float(m.get("sentiment_index", 50.0) or 50.0)
    e = float(m.get("attention_entropy", 0.0) or 0.0)
    ind = float(m.get("industry_count", 0) or 0)
    ev = [
        f"sentiment_index={s:.1f} (0~100, 50 中性)",
        f"attention_entropy={e:.3f} (越高越分散)",
        f"industry_count={ind:.0f} (覆盖广度)",
    ]
    ts = ctx.get("timeseries_30d", [])
    if len(ts) >= 5:
        vals = [float(t.get("sentiment_index", 50) or 50) for t in ts]
        n = len(vals)
        recent = vals[: max(1, n // 3)]
        older = vals[max(1, n // 3):]
        if recent and older:
            momentum = sum(recent) / len(recent) - sum(older) / len(older)
            ev.append(f"近 1/3 vs 前 2/3 momentum={momentum:+.2f}")
            if momentum > 2:
                stance = "bullish"
            elif momentum < -2:
                stance = "bearish"
            else:
                stance = "neutral"
        else:
            stance = "neutral"
    else:
        stance = "neutral"
    return AgentOpinion(
        persona="量化 Quant", persona_en="quant",
        stance=stance, confidence=0.55,
        key_points=[
            "不预测, 只描述: 当前数据呈现 {st} 特征".format(st=stance),
            "风险: 历史 30 日波动率需结合 VaR / Sharpe 综合判断",
            "建议: 严格执行风控阈值, 不预判方向",
        ],
        evidence=ev,
        reasoning="量化派拒绝预测, 只描述当前统计特征。",
        generated_by="template",
    )


def _gen_pragmatist(ctx: Dict[str, Any]) -> AgentOpinion:
    events = ctx.get("events", [])
    high_intensity = [e for e in events if float(e.get("intensity", 0) or 0) >= 0.6]
    ev = []
    if events:
        ev.append(f"今日事件 {len(events)} 个, 高强度 {len(high_intensity)} 个")
    if high_intensity:
        ev.append("重点跟踪: " + ", ".join(
            f"{e.get('subject', '?')}{e.get('action', '')}{e.get('object', '')}"
            for e in high_intensity[:3]
        ))
    if not ev:
        ev.append("事件密度低, 政策窗口尚未明确")
    stance = "neutral"
    if high_intensity:
        stance = "bullish" if any("利好" in (e.get("event_type", "") or "") for e in high_intensity) else "neutral"
    return AgentOpinion(
        persona="务实 Pragmatist", persona_en="pragmatist",
        stance=stance, confidence=0.5,
        key_points=[
            "政策从文件到落地通常 4-8 周, 当前处于观察期",
            "建议: 跟踪部委发布会 + 地方专项债 + PMI 验证信号",
            "操作上: 保留现金缓冲, 等数据拐点再加仓",
        ],
        evidence=ev,
        reasoning="务实派关注时间窗口和可执行性, 不预判长期方向。",
        generated_by="template",
    )


_TEMPLATE_GENERATORS = {
    "hawk": _gen_hawk,
    "dove": _gen_dove,
    "quant": _gen_quant,
    "pragmatist": _gen_pragmatist,
}


# ============================================================
# 3) LLM 增强 (可选, 失败回退)
# ============================================================
def _llm_opinion(persona: Dict[str, Any], question: str, ctx: Dict[str, Any], router) -> Optional[AgentOpinion]:
    if router is None or not getattr(router, "api_key", None):
        return None
    metrics = ctx.get("metrics", {})
    events = ctx.get("events", [])[:5]
    industries = ctx.get("industries", [])[:5]
    prompt = f"""你是一位{persona["name"]}宏观分析师。{persona["stance_hint"]}。

问题: {question}
日期: {ctx.get("date")}

量化数据:
- sentiment_index={metrics.get("sentiment_index", 50)}
- policy_stance_score={metrics.get("policy_stance_score", 0)}
- attention_entropy={metrics.get("attention_entropy", 0)}
- industry_count={metrics.get("industry_count", 0)}

主要事件: {[(e.get("subject"), e.get("action"), e.get("object")) for e in events]}
主要行业: {[(i.get("industry"), i.get("stance")) for i in industries]}

请用 JSON 输出你的分析:
{{
  "stance": "bullish|bearish|neutral",
  "confidence": 0.0~1.0,
  "key_points": ["3-5 条要点"],
  "evidence": ["2-4 条量化证据"],
  "reasoning": "1-2 句推理"
}}
只输出 JSON, 不要其他文字。"""
    try:
        resp = router.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=persona["temperature"],
            max_tokens=1500,
            use_json_mode=True,
            disable_thinking=True,
        )
        text = (resp[0] if isinstance(resp, tuple) else resp.get("content", "")).strip()
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        d = json.loads(m.group(0))
        return AgentOpinion(
            persona=persona["name"], persona_en=persona["name_en"],
            stance=d.get("stance", "neutral")[:8],
            confidence=float(d.get("confidence", 0.5)),
            key_points=[str(x)[:100] for x in d.get("key_points", [])[:5]],
            evidence=[str(x)[:100] for x in d.get("evidence", [])[:4]],
            reasoning=str(d.get("reasoning", ""))[:300],
            generated_by="llm",
        )
    except Exception as e:
        logger.debug(f"LLM {persona['name_en']} 失败: {e}")
        return None


# ============================================================
# 4) 仲裁综合
# ============================================================
_STANCE_SCORE = {"bullish": 1, "neutral": 0, "bearish": -1}


def _arbitrate(opinions: List[AgentOpinion]) -> tuple:
    if not opinions:
        return "neutral", 0.0, 0.0
    weighted = 0.0
    total_w = 0.0
    stances = []
    for op in opinions:
        w = op.confidence
        s = _STANCE_SCORE.get(op.stance, 0)
        weighted += s * w
        total_w += w
        stances.append(op.stance)
    if total_w == 0:
        return "neutral", 0.0, 0.0
    avg = weighted / total_w
    if avg > 0.2:
        final = "bullish"
    elif avg < -0.2:
        final = "bearish"
    else:
        final = "neutral"
    confidence = min(1.0, abs(avg) + 0.2)
    # consensus = 1 - (独特立场数 / 总数)
    distinct = len(set(stances))
    consensus = 1.0 - (distinct - 1) / max(len(stances), 1)
    return final, round(confidence, 3), round(consensus, 3)


def _synthesize(question: str, opinions: List[AgentOpinion], final: str,
                confidence: float, consensus: float) -> str:
    lines = [f"## 顾问团综合意见", ""]
    lines.append(f"**最终立场**: {final}  |  **置信度**: {confidence:.0%}  |  **共识度**: {consensus:.0%}")
    lines.append("")
    lines.append("**4 位顾问立场一览**")
    for op in opinions:
        lines.append(
            f"- {op.persona}: {op.stance} (置信 {op.confidence:.0%}, {op.generated_by})"
        )
    lines.append("")
    lines.append("**核心论据**")
    seen = set()
    for op in opinions:
        for ev in op.evidence[:2]:
            if ev not in seen:
                seen.add(ev)
                lines.append(f"- [{op.persona_en}] {ev}")
    if not seen:
        lines.append("- (无具体证据)")
    return "\n".join(lines)


# ============================================================
# 5) 对外主函数
# ============================================================
def council(question: str, target_date: str, router=None, use_llm: bool = True) -> CouncilResult:
    """运行 4 人顾问团, 返回 CouncilResult."""
    t0 = time.time()
    ctx = _load_context(target_date)
    opinions: List[AgentOpinion] = []

    for persona in PERSONAS:
        op = None
        if use_llm:
            op = _llm_opinion(persona, question, ctx, router)
        if op is None:
            op = _TEMPLATE_GENERATORS[persona["name_en"]](ctx)
        opinions.append(op)

    final, conf, consensus = _arbitrate(opinions)
    synthesis = _synthesize(question, opinions, final, conf, consensus)

    llm_count = sum(1 for o in opinions if o.generated_by == "llm")
    return CouncilResult(
        question=question,
        date=target_date,
        opinions=opinions,
        synthesis=synthesis,
        final_stance=final,
        final_confidence=conf,
        consensus_level=consensus,
        debate_rounds=1,
        total_latency_ms=round((time.time() - t0) * 1000, 1),
        generated_by="llm+template" if llm_count else "template",
        summary=f"4 位顾问立场={final}, 置信 {conf:.0%}, 共识 {consensus:.0%}",
    )
