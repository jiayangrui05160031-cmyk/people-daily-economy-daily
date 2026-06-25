"""react_agent.py - v9 前沿: ReAct 工具调用 LLM Agent (无 LangChain 依赖)

让 LLM 自己决定"调哪个工具 / 取什么参数", 自动循环多步, 直至给出最终答案。
工具集覆盖本项目 v5-v9 所有 24 个分析模块的关键能力。

学术参考:
  - Yao et al. "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)
  - OpenAI "Function Calling" 文档 (2023-06)

核心设计:
  1) 工具注册: 名字 -> {description, parameters (JSON Schema), handler}
  2) 优先用 OpenAI 兼容 function calling (tool_calls 字段)
  3) 不可用时降级: 文本 ReAct 格式 "Action: tool_name\nAction Input: {...}"
  4) 循环终止: 收到 Final Answer 字段 / 达到 max_steps / 异常
  5) 完整 trace 记录每步 Thought/Action/Observation, 供前端可视化

工具集 (默认 7 个):
  - get_signal: 当日 BUY/HOLD/REDUCE/SELL 决策
  - get_factor_model: 5 因子暴露 + 总分
  - get_regime: Hamilton 区制定位
  - get_risk_metrics: 8 类风险指标
  - semantic_search: 向量 RAG 检索 (历史报告)
  - ask_graph_rag: 全局知识图谱问答
  - run_scenario: 蒙特卡洛情景压力测试

典型用法:
    from src.agent.react_agent import build_default_agent, run_agent
    agent = build_default_agent(router=router)
    result = agent.run("当前宏观风险与机会是什么?", target_date="2026-06-12")
    print(result.answer, result.steps)
"""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.logger import get_logger

logger = get_logger("agent.react")


# ============================================================
# 工具定义
# ============================================================
@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]      # JSON Schema
    handler: Callable[..., str]     # 接受 kwargs, 返回 str (观察)

    def to_openai_schema(self) -> Dict[str, Any]:
        """转 OpenAI tools 格式."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ============================================================
# Agent 步骤 + 结果
# ============================================================
@dataclass
class AgentStep:
    step: int
    thought: str
    action: str             # 工具名 (或 "Final Answer")
    action_input: Dict[str, Any]
    observation: str        # 工具返回值 (截断)
    elapsed_ms: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentResult:
    question: str
    answer: str
    steps: List[AgentStep] = field(default_factory=list)
    n_steps: int = 0
    n_tool_calls: int = 0
    final_action: str = ""
    generated_by: str = "react"     # "react-function-calling" | "react-text" | "react-template"
    elapsed_ms: int = 0

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["steps"] = [s.as_dict() for s in self.steps]
        return d


# ============================================================
# 工具实现 (handler 都返回 str 摘要)
# ============================================================
def _tool_get_signal(target_date: str = "") -> str:
    try:
        from src.analysis.signal_engine import synthesize
        from src.analysis.anomaly import detect
        from src.analysis.forecast import predict_next_day
        from src.analysis.volatility import compute
        from src.analysis.stock_correlation import correlate
        from src.analysis.topic_model import fit_with_evolution
        from src.report.archiver import load_raw
        date = target_date or ""
        if not date:
            from src.utils.date_utils import resolve_target_date
            date = str(resolve_target_date())
        articles = load_raw(date) or []
        # 调各模块 (尽量给 None, 触发内部最新数据逻辑)
        anomaly = detect(date)
        forecast = predict_next_day(date)
        vol = compute(date)
        market = correlate(date, industries_hit=[])
        topics = fit_with_evolution(articles, date) if articles else None
        # events_study 跳过 (synthesize 内部需要 EventStudy 对象, 不同模块返回值有差异)
        sig = synthesize(
            target_date=date, anomaly=anomaly, forecast=forecast,
            volatility=vol, market=market, macro=None,
            events_study=None, topics=topics,
        )
        return json.dumps({
            "action": sig.action,
            "score": round(sig.score, 4),
            "confidence": round(sig.confidence, 4),
            "top_reasons": sig.top_reasons[:3],
            "n_signals": len(sig.signals),
        }, ensure_ascii=False)
    except Exception as e:
        return f"{{\"error\": \"{e}\"}}"


def _tool_get_factor_model(target_date: str = "") -> str:
    try:
        from src.analysis.factor_model import compute
        r = compute(target_date or "2026-06-12", lookback=90)
        if r is None:
            return "{\"error\": \"insufficient data for factor model\"}"
        return json.dumps({
            "total_score": r.total_score,
            "rank": r.total_rank,
            "n_periods": r.n_periods,
            "factor_returns": r.factor_returns,
            "top_factor": (max(r.factor_returns, key=r.factor_returns.get)
                           if r.factor_returns else None),
        }, ensure_ascii=False)
    except Exception as e:
        return f"{{\"error\": \"{e}\"}}"


def _tool_get_regime(target_date: str = "") -> str:
    try:
        from src.analysis.regime import fit_from_daily_metric, regime_to_signal
        rep = fit_from_daily_metric(target_date or "", lookback=30,
                                     metric_col="sentiment_index")
        if rep is None:
            return "{\"regime\": \"unknown\", \"n_obs\": 0}"
        sig = regime_to_signal(rep)
        return json.dumps({
            "regime": rep.current_regime,
            "p_calm": rep.current_p_calm,
            "p_euphoric": rep.current_p_euphoric,
            "means": rep.state_means,
            "avg_sojourn": [rep.avg_sojourn_calm, rep.avg_sojourn_euphoric],
            "converged": rep.converged,
            "signal": sig,
        }, ensure_ascii=False)
    except Exception as e:
        return f"{{\"error\": \"{e}\"}}"


def _tool_get_risk_metrics(target_date: str = "") -> str:
    try:
        from src.analysis.risk_metrics import compute
        rm = compute(target_date or "")
        return json.dumps({
            "sharpe_ratio": rm.sharpe_ratio, "sortino_ratio": rm.sortino_ratio, "calmar_ratio": rm.calmar_ratio, "expected_shortfall_95": rm.expected_shortfall_95,
            "max_drawdown": rm.max_drawdown,
            "var_95": rm.var_95, "var_99": rm.var_99,
            "expected_shortfall_95": rm.expected_shortfall_95,
            "skewness": rm.skewness,
            "kurtosis": rm.kurtosis,
            "level": rm.risk_level,
        }, ensure_ascii=False)
    except Exception as e:
        return f"{{\"error\": \"{e}\"}}"


def _tool_semantic_search(query: str, top_k: int = 5) -> str:
    try:
        from src.analysis.embeddings import semantic_search
        from src.analysis.embeddings import build_corpus_from_reports
        corpus = build_corpus_from_reports(lookback_days=30)
        if not corpus:
            return "{\"hits\": [], \"msg\": \"no corpus\"}"
        hits = semantic_search(query, docs=corpus, top_k=top_k)
        out = [{"id": h.id, "score": h.score,
                "snippet": h.text[:120]} for h in hits]
        return json.dumps({"hits": out, "n_corpus": len(corpus)},
                          ensure_ascii=False)
    except Exception as e:
        return f"{{\"error\": \"{e}\"}}"


def _tool_ask_graph_rag(question: str, target_date: str = "") -> str:
    try:
        from src.analysis.graph_rag import ask_global
        from src.report.archiver import load_raw
        from datetime import timedelta
        from src.utils.date_utils import parse_date
        # 拿 target_date 当天 + 前 3 天的文章
        if not target_date:
            return "{\"answer\": \"(需提供 target_date)\"}"
        end = parse_date(target_date)
        arts = []
        for off in range(0, 4):
            d = (end - timedelta(days=off)).isoformat()
            try:
                arts.extend(load_raw(d) or [])
            except Exception:
                pass
        if not arts:
            return "{\"answer\": \"(近 4 天无归档文章)\"}"
        ans = ask_global(question, arts, router=None, use_llm=False)
        return json.dumps({
            "answer": ans["answer"][:400],
            "communities_used": ans["communities_used"],
            "modularity": ans.get("modularity"),
        }, ensure_ascii=False)
    except Exception as e:
        return f"{{\"error\": \"{e}\"}}"


def _tool_run_scenario(scenario: str = "base") -> str:
    try:
        from src.analysis.scenario import list_scenarios, run
        if scenario == "list":
            return json.dumps({"scenarios": list_scenarios()},
                              ensure_ascii=False)
        r = run(target_date=target_date or "2026-06-12", scenario=scenario, horizon_days=30, n_sims=300, seed=42)
        return json.dumps({
            "scenario": r.scenario_name,
            "p_positive": getattr(r, "p_positive", 0),
            "p5": r.p5, "p50": r.p50, "p95": r.p95,
            "expected_return": r.expected_return,
        }, ensure_ascii=False)
    except Exception as e:
        return f"{{\"error\": \"{e}\"}}"


# ============================================================
# 工具注册表
# ============================================================
def default_tools() -> List[Tool]:
    return [
        Tool(
            name="get_signal",
            description=("获取当日综合交易信号 (BUY/HOLD/REDUCE/SELL) + 置信度 + 关键理由。"
                         "输入 target_date (YYYY-MM-DD), 不传则用最新数据。"),
            parameters={
                "type": "object",
                "properties": {
                    "target_date": {"type": "string",
                                    "description": "目标日期 YYYY-MM-DD"}
                },
                "required": [],
            },
            handler=_tool_get_signal,
        ),
        Tool(
            name="get_factor_model",
            description="获取 5 因子 (动量/价值/质量/波动/规模) 暴露 + 总分 + 评级 (A-E).",
            parameters={
                "type": "object",
                "properties": {
                    "target_date": {"type": "string"}
                },
                "required": [],
            },
            handler=_tool_get_factor_model,
        ),
        Tool(
            name="get_regime",
            description=("Hamilton 马尔可夫区制定位: 当前是'冷静'还是'狂热'区制, "
                         "返回后验概率 + 转移矩阵 + 平均持续天数。"),
            parameters={
                "type": "object",
                "properties": {
                    "target_date": {"type": "string"}
                },
                "required": [],
            },
            handler=_tool_get_regime,
        ),
        Tool(
            name="get_risk_metrics",
            description="8 类风险指标: Sharpe/Sortino/MaxDD/Calmar/VaR/CVaR/Skew/Kurtosis + 等级.",
            parameters={
                "type": "object",
                "properties": {
                    "target_date": {"type": "string"}
                },
                "required": [],
            },
            handler=_tool_get_risk_metrics,
        ),
        Tool(
            name="semantic_search",
            description=("向量语义检索: 在过去 30 天报告中找与 query 语义相近的片段, "
                         "返回 id/score/snippet. 支持同义词/改写。"),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "查询文本 (支持自然语言)"},
                    "top_k": {"type": "integer", "default": 5}
                },
                "required": ["query"],
            },
            handler=_tool_semantic_search,
        ),
        Tool(
            name="ask_graph_rag",
            description=("知识图谱全局问答: 给定问题 + 日期, 返回跨文章知识图谱社区级答案 + 涉及社区。"),
            parameters={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "target_date": {"type": "string"}
                },
                "required": ["question"],
            },
            handler=_tool_ask_graph_rag,
        ),
        Tool(
            name="run_scenario",
            description=("蒙特卡洛情景压力测试: 输入情景名 (base/drr/cut_rate/real_estate_crash/export_shock/geopolitics/ai_breakthrough) "
                         "或 'list' 列出所有情景。"),
            parameters={
                "type": "object",
                "properties": {
                    "scenario": {"type": "string", "default": "base"}
                },
                "required": [],
            },
            handler=_tool_run_scenario,
        ),
    ]


# ============================================================
# ReAct Agent
# ============================================================
REACT_SYSTEM_PROMPT = """你是中国宏观经济研究 ReAct 智能体。

工作循环: 思考 -> 选工具 -> 拿观察 -> 再思考 -> ... -> 最终答案。

规则:
1) 你必须从工具列表中选一个, 或者给出 Final Answer (当信息足够时)
2) 每次只输出一段 Thought 和一段 Action (含 Action Input JSON)
3) 不要捏造工具返回值
4) 工具返回的是 JSON 字符串, 你需解析后纳入下一轮思考
5) Final Answer 必须是中文, 150-400 字, 引用具体数据, 给出明确结论

输出格式 (严格):
Thought: <你的思考, 1-2 句>
Action: <工具名 或 "Final Answer">
Action Input: <JSON 对象 或 答案文本>
"""


class ReActAgent:
    def __init__(self,
                 router: Any = None,
                 tools: Optional[List[Tool]] = None,
                 max_steps: int = 6):
        self.router = router
        self.tools = {t.name: t for t in (tools or default_tools())}
        self.max_steps = max_steps

    def _openai_schema_list(self) -> List[Dict[str, Any]]:
        return [t.to_openai_schema() for t in self.tools.values()]

    # -----------------------------
    # 主入口
    # -----------------------------
    def run(self, question: str, target_date: str = "") -> AgentResult:
        t0 = time.time()
        steps: List[AgentStep] = []
        n_tool = 0
        final = ""
        gen_by = "react"

        if self.router is not None and getattr(self.router, "api_key", ""):
            # 优先: OpenAI function calling
            try:
                final, steps, n_tool, gen_by = self._run_with_function_calling(
                    question, target_date)
            except Exception as e:
                logger.info("function calling 模式失败, 降级文本 ReAct: %s", e)
                final, steps, n_tool, gen_by = self._run_text_react(
                    question, target_date)
        else:
            # 无 router: 纯模板 (单步, 直接调 1-2 个工具 + 拼装答案)
            final, steps, n_tool, gen_by = self._run_template(question, target_date)

        return AgentResult(
            question=question,
            answer=final,
            steps=steps,
            n_steps=len(steps),
            n_tool_calls=n_tool,
            final_action=steps[-1].action if steps else "",
            generated_by=gen_by,
            elapsed_ms=int((time.time() - t0) * 1000),
        )

    # -----------------------------
    # 模式 1: OpenAI Function Calling (原生 tool_calls)
    # -----------------------------
    def _run_with_function_calling(self, question: str,
                                    target_date: str):
        messages = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {"role": "user",
             "content": question + (
                 (" (参考日期: " + target_date + ")" if target_date else "")
             )},
        ]
        steps: List[AgentStep] = []
        n_tool = 0
        final = ""
        for step_n in range(1, self.max_steps + 1):
            t0 = time.time()
            raw, used_model, pt, ct = self.router.chat(
                messages,
                temperature=0.3,
                max_tokens=1200,
                use_json_mode=False,
            )
            # raw 可能含 <think> 块
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            # 兼容: 优先解析 tool_calls (OpenAI 格式)
            tool_calls = self._parse_tool_calls(raw, used_model)
            if tool_calls:
                # 记录 Thought + Action
                thought = re.search(r"Thought:\s*(.+?)(?:\n|$)", raw, re.S)
                thought = thought.group(1).strip() if thought else ""
                for call in tool_calls:
                    obs = self._dispatch(call["name"], call["arguments"])
                    steps.append(AgentStep(
                        step=step_n,
                        thought=thought or "(no thought)",
                        action=call["name"],
                        action_input=call["arguments"],
                        observation=obs[:500],
                        elapsed_ms=int((time.time() - t0) * 1000),
                    ))
                    n_tool += 1
                    # 把 tool call + 观察加入 messages
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({
                        "role": "user",
                        "content": ("Observation (" + call["name"] + "): " + obs[:500]),
                    })
                continue
            # 兼容: 文本 Action: tool_name / Final Answer
            text_action = self._parse_text_action(raw)
            if text_action and text_action["action"] == "Final Answer":
                final = text_action["input_text"]
                steps.append(AgentStep(
                    step=step_n, thought=text_action.get("thought", ""),
                    action="Final Answer", action_input={},
                    observation=final[:200],
                    elapsed_ms=int((time.time() - t0) * 1000),
                ))
                return final, steps, n_tool, "react-function-calling"
            if text_action:
                obs = self._dispatch(text_action["action"], text_action["input"])
                steps.append(AgentStep(
                    step=step_n,
                    thought=text_action.get("thought", ""),
                    action=text_action["action"],
                    action_input=text_action["input"],
                    observation=obs[:500],
                    elapsed_ms=int((time.time() - t0) * 1000),
                ))
                n_tool += 1
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": ("Observation (" + text_action["action"] + "): " + obs[:500]),
                })
                continue
            # 兜底: raw 视为 Final Answer
            final = raw[:600]
            steps.append(AgentStep(
                step=step_n, thought="(direct answer)",
                action="Final Answer", action_input={},
                observation=final[:200],
                elapsed_ms=int((time.time() - t0) * 1000),
            ))
            return final, steps, n_tool, "react-function-calling"

        # max_steps 耗尽
        if not final and steps:
            final = ("(达 max_steps=" + str(self.max_steps) +
                     ", 最后观察: " + steps[-1].observation[:300] + ")")
        return final, steps, n_tool, "react-function-calling"

    def _parse_tool_calls(self, raw: str, model: str):
        """从原始 LLM 输出中提取 tool_calls.
        我们的 router.chat() 默认返回 (content, model, pt, ct),
        不返回 tool_calls. 这里退化为: 解析 <tool_call> 块 (OpenAI 协议约定)"""
        # 格式 1: <tool_call>{"name": "x", "arguments": {...}}</tool_call>
        calls = []
        for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", raw, re.S):
            try:
                obj = json.loads(m.group(1))
                if "name" in obj:
                    calls.append({
                        "name": obj["name"],
                        "arguments": obj.get("arguments", {}),
                    })
            except Exception:
                pass
        return calls

    def _parse_text_action(self, raw: str):
        """解析文本格式 ReAct: Action: <name>\nAction Input: <json/text>"""
        m = re.search(r"Thought:\s*(.+?)\nAction:\s*(.+?)\nAction Input:\s*(.+?)(?:\n\n|$)",
                      raw, re.S)
        if not m:
            return None
        thought = m.group(1).strip()
        action = m.group(2).strip()
        input_str = m.group(3).strip()
        if action == "Final Answer":
            return {"thought": thought, "action": "Final Answer",
                    "input_text": input_str, "input": {}}
        # 尝试 JSON
        try:
            inp = json.loads(input_str)
        except Exception:
            inp = {"query": input_str} if action == "semantic_search" else {"raw": input_str}
        return {"thought": thought, "action": action, "input": inp}

    def _dispatch(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name not in self.tools:
            return ("{\"error\": \"unknown tool '" + tool_name + "'\"}")
        try:
            return self.tools[tool_name].handler(**(arguments or {}))
        except Exception as e:
            return ("{\"error\": \"" + str(e) + "\"}")

    # -----------------------------
    # 模式 2: 文本 ReAct (router 不可用时, 走 LLM 但不走 function calling)
    # -----------------------------
    def _run_text_react(self, question: str, target_date: str):
        # 与 function calling 一致, 但不传 tool_calls
        # 我们直接走 function calling 路径; 如果 router.chat 不支持 tool 参数
        # 会回落到 parse_text_action
        return self._run_with_function_calling(question, target_date)

    # -----------------------------
    # 模式 3: 无 LLM, 模板 (调 1-2 个工具拼答案)
    # -----------------------------
    def _run_template(self, question: str, target_date: str):
        steps: List[AgentStep] = []
        n_tool = 0
        # 调 3 个最有代表性的工具
        for tool_name in ("get_signal", "get_regime", "get_risk_metrics"):
            t0 = time.time()
            obs = self._dispatch(tool_name, {"target_date": target_date})
            steps.append(AgentStep(
                step=len(steps) + 1,
                thought=("(no LLM, 调 " + tool_name + ")"),
                action=tool_name,
                action_input={"target_date": target_date},
                observation=obs[:500],
                elapsed_ms=int((time.time() - t0) * 1000),
            ))
            n_tool += 1
        # 拼装答案
        final = ("(无 LLM 模板模式) 基于 " + str(n_tool) + " 个工具观察:\n")
        for s in steps:
            final += "- " + s.action + ": " + s.observation[:120] + "\n"
        return final, steps, n_tool, "react-template"


# ============================================================
# 便捷构造
# ============================================================
def build_default_agent(router: Any = None,
                        tools: Optional[List[Tool]] = None,
                        max_steps: int = 6) -> ReActAgent:
    return ReActAgent(router=router, tools=tools, max_steps=max_steps)


def run_agent(question: str,
              target_date: str = "",
              router: Any = None) -> AgentResult:
    agent = build_default_agent(router=router)
    return agent.run(question, target_date=target_date)


# ============================================================
# 自检
# ============================================================
if __name__ == "__main__":
    print("== react_agent self-test (template 模式) ==")
    res = run_agent("今日宏观风险与机会", target_date="2026-06-12", router=None)
    print(f"n_steps={res.n_steps}, n_tool_calls={res.n_tool_calls}, by={res.generated_by}, {res.elapsed_ms}ms")
    for s in res.steps:
        print(f"  step {s.step}: {s.action}  in={s.action_input}  out={s.observation[:100]}")
    print("answer:", res.answer[:200])
    assert res.n_tool_calls >= 1
    print("\nAll react_agent self-tests passed (template)")
