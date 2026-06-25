"""analysis.llm_judge - LLM-as-Judge 多维质量门控 (v9 前沿模块).

设计目标:
  - 报告生成后, 由多个 "虚拟 Judge" 角色各自独立打分
  - 每个 Judge 在 5 个维度上评估 (0~1): 一致性 / 有据性 / 完整性 / 可行动性 / 新颖性
  - 多 Judge 投票: 平均分 + 共识度 + 离散度 (consensus & disagreement)
  - 出报告前自评, 不达标则给出改进建议, 阻止低质量外发

Judge 角色 (4 个, 来自 README 架构图: 政策 / 产业 / 市场 / 战略):
  1. PolicyJudge    政策导向 - 关注与中央/部委口径一致性、风险措辞、负面表述
  2. IndustryJudge  产业导向 - 关注产业链 / 主题分布 / 上下游联动
  3. MarketJudge    市场导向 - 关注数据点 vs 实际行情 / 情绪 vs 量化指标
  4. StrategyJudge  战略导向 - 关注可执行性 / 时间窗口 / 决策路径

LLM 不可用时降级为规则打分 (关键词密度 + 章节完整性 + 引用统计), 保证总能跑通.

典型用法:
    from src.analysis.llm_judge import judge_report, JudgeReport
    result = judge_report(report_text=open("reports/2026-06-12.md").read(),
                          target_date="2026-06-12")
    print(result.overall_score, result.pass_, result.improvements)

依赖:
    - 可选: src.ai.client (LLM 调用)
    - 可选: src.storage.db (打分历史落库)
    - 必要: src.utils.logger
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger("analysis.llm_judge")

# ============================================================
# 1. 常量: 维度 + Judge 角色
# ============================================================

# 5 个评估维度 (来自 README)
DIMENSIONS: List[str] = ["consistency", "groundedness", "completeness", "actionability", "novelty"]

DIMENSION_CN: Dict[str, str] = {
    "consistency":   "一致性",    # 全文立场/数据/结论是否自洽, 不自相矛盾
    "groundedness":  "有据性",    # 结论是否有数据/事件/引用支撑, 不空喊口号
    "completeness":  "完整性",    # 31 节结构是否齐, 关键数据点是否覆盖
    "actionability": "可行动性",  # 是否给出明确的政策/产业/操作建议 + 时间窗口
    "novelty":       "新颖性",    # 是否识别了非常规信号 / 冷门主题 / 反共识判断
}

DIMENSION_WEIGHTS: Dict[str, float] = {
    "consistency":   0.20,
    "groundedness":  0.25,
    "completeness":  0.20,
    "actionability": 0.20,
    "novelty":       0.15,
}

# 4 个 Judge 角色
JUDGE_PERSONAS: List[Dict[str, Any]] = [
    {
        "name": "PolicyJudge",
        "name_cn": "政策导向",
        "icon": "🏛️",
        "weight": 0.30,
        "focus": "与中央/部委文件口径一致性、风险措辞审慎度、政策延续性",
        "temperature": 0.2,
        "rubric": {
            "consistency":   "是否与最近 3 个月国务院/部委口径一致?是否出现明显矛盾立场?",
            "groundedness":  "政策结论是否引用了具体文件/会议/数据?",
            "completeness":  "是否覆盖了主要政策领域 (货币/财政/产业/外贸)?",
            "actionability": "是否给出明确的政策落地路径 / 时间窗口 / 责任部门?",
            "novelty":       "是否识别了非常规政策信号 (如跨部门协同/新工具)?",
        },
    },
    {
        "name": "IndustryJudge",
        "name_cn": "产业导向",
        "icon": "🏭",
        "weight": 0.25,
        "focus": "产业链完整性、上下游联动、主题热度、替代/互补关系",
        "temperature": 0.4,
        "rubric": {
            "consistency":   "产业判断在上下游环节是否一致?有无自相矛盾?",
            "groundedness":  "是否引用了具体公司/项目/订单/产能数据?",
            "completeness":  "是否覆盖了主流 + 新兴两条产业链?",
            "actionability": "是否给出选股/选赛道 / 仓位建议 + 触发条件?",
            "novelty":       "是否识别了非共识的细分赛道 / 拐点信号?",
        },
    },
    {
        "name": "MarketJudge",
        "name_cn": "市场导向",
        "icon": "📈",
        "weight": 0.25,
        "focus": "数据点 vs 实际行情、情绪 vs 量化指标、估值 vs 历史区间",
        "temperature": 0.3,
        "rubric": {
            "consistency":   "文中情绪判断与量化指标 (Sharpe/VaR/动量) 是否吻合?",
            "groundedness":  "市场判断是否引用了具体指数/板块/资金流向数据?",
            "completeness":  "是否覆盖了股/债/汇/商品 4 大类资产?",
            "actionability": "是否给出明确的进场/出场条件 + 仓位 + 止损?",
            "novelty":       "是否识别了异常成交 / 资金异动 / 持仓变化?",
        },
    },
    {
        "name": "StrategyJudge",
        "name_cn": "战略导向",
        "icon": "🧭",
        "weight": 0.20,
        "focus": "可执行性、时间窗口、决策路径、风险预案",
        "temperature": 0.3,
        "rubric": {
            "consistency":   "战略建议在不同章节是否一致 (顶层与执行层不打架)?",
            "groundedness":  "战略建议是否基于文中已展示的数据/事件?",
            "completeness":  "是否给出了主选/备选/对冲三档方案?",
            "actionability": "是否给出 1 周 / 1 月 / 1 季的时间窗口 + 触发指标?",
            "novelty":       "是否提出非传统的反共识战略视角?",
        },
    },
]


# ============================================================
# 2. 数据结构
# ============================================================

@dataclass
class DimensionScore:
    """单个维度的多 Judge 评分."""
    dimension: str                # 一致性 / 有据性 / ...
    dimension_cn: str
    scores: Dict[str, float]      # {judge_name: 0~1}
    mean: float = 0.0
    std: float = 0.0              # 离散度 (越小越共识)
    consensus: float = 0.0        # 1 - std, 越大越共识
    weighted: float = 0.0         # 维度权重 × judge 权重 加权后的分数

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class JudgeOpinion:
    """单个 Judge 的总体意见."""
    judge: str
    judge_cn: str
    icon: str
    overall_score: float = 0.0    # 0~1, 加权后
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    reasoning: str = ""           # LLM 推理过程 / 模板说明
    key_concerns: List[str] = field(default_factory=list)
    key_strengths: List[str] = field(default_factory=list)
    generated_by: str = "template"  # 'llm' | 'template'
    model: str = ""
    elapsed_ms: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Improvement:
    """针对未达标维度的具体改进建议."""
    dimension: str
    dimension_cn: str
    current_score: float
    target_score: float
    suggestions: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class JudgeReport:
    """LLM-as-Judge 最终评估报告."""
    target_date: str
    source: str                   # 'report:2026-06-12.md' | 'inline'
    text_length: int              # 报告字符数
    section_count: int            # 解析出的章节数

    # 维度级
    dimension_scores: List[DimensionScore] = field(default_factory=list)

    # Judge 级
    judge_opinions: List[JudgeOpinion] = field(default_factory=list)

    # 总体
    overall_score: float = 0.0    # 0~1 加权总分
    consensus_level: float = 0.0  # 0~1, 所有维度 consensus 的平均
    pass_threshold: float = 0.65 # 通过门槛 (可调)
    pass_: bool = False           # 是否通过

    # 改进
    improvements: List[Improvement] = field(default_factory=list)

    # 元信息
    generated_by: str = "template"
    judges_used: int = 0
    dimensions_used: int = 0
    total_latency_ms: float = 0.0
    model: str = ""
    summary: str = ""

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["dimension_scores"] = [s.as_dict() for s in self.dimension_scores]
        d["judge_opinions"] = [o.as_dict() for o in self.judge_opinions]
        d["improvements"] = [i.as_dict() for i in self.improvements]
        return d


# ============================================================
# 3. 规则打分 (无 LLM 时的降级实现)
# ============================================================

# 章节头部 (中文报告常见的章节模式, 用于 completeness 维度)
_SECTION_PATTERNS = [
    re.compile(r"^#+\s+第?\d+[\.、\s]", re.MULTILINE),
    re.compile(r"^#+\s+[一二三四五六七八九十]+[\.、\s]", re.MULTILINE),
    re.compile(r"^#+\s+\d+\.\s+\*\*", re.MULTILINE),
    re.compile(r"^##\s+[A-Z]", re.MULTILINE),
]

# 数据引用关键词 (groundedness 维度)
_GROUND_KEYWORDS = [
    r"\d+\.\d+%", r"\d+\%", r"\d+亿", r"\d+万亿", r"\d+\.\d+亿",
    r"YoY", r"MoM", r"QoQ", r"环比", r"同比",
    r"GDP", r"CPI", r"PPI", r"PMI", r"M2",
    r"基准利率", r"LPR", r"MLF", r"DR007",
    r"据.+?统计", r"据.+?数据", r"据.+?消息",
    r"来源[:：]", r"央行", r"财政部", r"统计局",
    r"会议", r"文件", r"通知", r"讲话",
]

# 行动/建议关键词 (actionability 维度)
_ACTION_KEYWORDS = [
    r"建议", r"推荐", r"应当", r"需要", r"必须",
    r"关注", r"警惕", r"回避", r"加仓", r"减仓", r"建仓", r"清仓",
    r"时间窗口", r"短期", r"中期", r"长期",
    r"配置", r"仓位", r"权重", r"比例",
    r"止损", r"止盈", r"目标价", r"支撑位", r"压力位",
    r"触发条件", r"入场", r"出场",
]

# 新颖信号关键词 (novelty 维度)
_NOVEL_KEYWORDS = [
    r"首次", r"突破", r"新低", r"新高",
    r"超预期", r"低于预期", r"高于预期",
    r"拐点", r"反转", r"反弹", r"反转信号",
    r"异常", r"异动", r"突袭", r"突现",
    r"非共识", r"反共识", r"小众", r"非主流",
    r"潜在", r"隐藏", r"被忽视", r"未充分定价",
]

# 一致性负面信号 (矛盾/自我否定)
_INCONSISTENCY_PATTERNS = [
    re.compile(r"(乐观|积极|扩张).{0,40}(悲观|消极|收缩)", re.DOTALL),
    re.compile(r"(看多|看涨).{0,40}(看空|看跌)", re.DOTALL),
    re.compile(r"(上涨|增长).{0,40}(下跌|下滑)", re.DOTALL),
]

# 政策一致性正面信号
_POLICY_ALIGN_KEYWORDS = [
    r"中央", r"国务院", r"政治局", r"二十大", r"两会",
    r"新质生产力", r"高质量发展", r"供给侧", r"需求侧",
    r"双循环", r"共同富裕", r"乡村振兴", r"科技自立",
    r"碳中和", r"碳达峰", r"数字经济",
]


def _count_sections(text: str) -> int:
    """统计章节数."""
    if not text:
        return 0
    seen = set()
    for pat in _SECTION_PATTERNS:
        for m in pat.finditer(text):
            start = m.start()
            # 用前 80 字符做去重 key (避免同一章节多次匹配)
            key = text[start:start + 80].strip()
            seen.add(key)
    return len(seen)


def _count_keyword_hits(text: str, patterns: List[str]) -> int:
    """统计关键词命中次数."""
    if not text:
        return 0
    cnt = 0
    for p in patterns:
        cnt += len(re.findall(p, text))
    return cnt


def _normalize(value: float, cap: float = 50.0) -> float:
    """归一化到 0~1 (用 cap 做饱和)."""
    if cap <= 0:
        return 0.0
    return min(1.0, max(0.0, value / cap))


def _rule_score_dimension(dimension: str, text: str, sections: int) -> Tuple[float, str]:
    """用规则给单个维度打分 (0~1). 返回 (分数, 简短理由)."""
    if not text:
        return 0.0, "无文本"

    if dimension == "consistency":
        # 一致性: 检查文内矛盾 (低 = 好) + 立场密度 (高 = 一致)
        contradictions = sum(1 for p in _INCONSISTENCY_PATTERNS if p.search(text))
        stance_density = _count_keyword_hits(text, [
            r"我们认为", r"判断", r"预期", r"预计", r"观点"
        ])
        # 矛盾越少越好; 立场密度越高越好
        score = 1.0 - _normalize(contradictions, 3.0) * 0.6
        score = score * (0.5 + 0.5 * _normalize(stance_density, 8.0))
        reason = f"矛盾信号={contradictions}, 立场密度={stance_density}"
        return max(0.0, min(1.0, score)), reason

    if dimension == "groundedness":
        hits = _count_keyword_hits(text, _GROUND_KEYWORDS)
        score = _normalize(hits, 25.0)
        reason = f"数据/引用关键词={hits} 处"
        return score, reason

    if dimension == "completeness":
        # 章节数 + 是否含 31 节结构
        sec_score = _normalize(sections, 31.0)
        # 加分: 包含 5 大领域关键词
        domains = ["货币", "财政", "产业", "外贸", "就业", "通胀"]
        hit = sum(1 for d in domains if d in text)
        score = sec_score * 0.7 + _normalize(hit, len(domains)) * 0.3
        reason = f"章节={sections}, 五大领域覆盖={hit}/{len(domains)}"
        return max(0.0, min(1.0, score)), reason

    if dimension == "actionability":
        hits = _count_keyword_hits(text, _ACTION_KEYWORDS)
        score = _normalize(hits, 12.0)
        reason = f"行动建议关键词={hits} 处"
        return score, reason

    if dimension == "novelty":
        hits = _count_keyword_hits(text, _NOVEL_KEYWORDS)
        score = _normalize(hits, 6.0)
        reason = f"新颖信号关键词={hits} 处"
        return score, reason

    return 0.5, f"未知维度 {dimension}"


def _rule_judge_score(persona: Dict[str, Any], text: str, sections: int,
                      dim_scores: Dict[str, float]) -> Tuple[float, Dict[str, float], List[str], List[str]]:
    """单个 Judge 用规则打分 (考虑 Judge 的 focus/rubric 偏好)."""
    weights: Dict[str, float] = {
        "consistency":   0.20,
        "groundedness":  0.25,
        "completeness":  0.20,
        "actionability": 0.20,
        "novelty":       0.15,
    }
    # 政策导向的 Judge 对一致性更看重 (再加权)
    focus = persona.get("focus", "")
    if "政策" in focus or "中央" in focus:
        weights["consistency"] = 0.35
        weights["groundedness"] = 0.30
        weights["completeness"] = 0.15
        weights["actionability"] = 0.10
        weights["novelty"] = 0.10
    elif "产业" in focus:
        weights["consistency"] = 0.15
        weights["groundedness"] = 0.30
        weights["completeness"] = 0.25
        weights["actionability"] = 0.20
        weights["novelty"] = 0.10
    elif "市场" in focus:
        weights["consistency"] = 0.20
        weights["groundedness"] = 0.30
        weights["completeness"] = 0.15
        weights["actionability"] = 0.25
        weights["novelty"] = 0.10
    elif "战略" in focus:
        weights["consistency"] = 0.25
        weights["groundedness"] = 0.15
        weights["completeness"] = 0.15
        weights["actionability"] = 0.30
        weights["novelty"] = 0.15

    overall = sum(dim_scores[d] * weights[d] for d in DIMENSIONS)

    concerns: List[str] = []
    strengths: List[str] = []
    for d in DIMENSIONS:
        s = dim_scores[d]
        if s < 0.4:
            concerns.append(f"{DIMENSION_CN[d]}偏低 ({s:.2f})")
        elif s >= 0.7:
            strengths.append(f"{DIMENSION_CN[d]}表现好 ({s:.2f})")

    return max(0.0, min(1.0, overall)), dim_scores, concerns, strengths


def _build_rule_judge_opinions(text: str, dim_scores: Dict[str, float], sections: int) -> List[JudgeOpinion]:
    """用规则给 4 个 Judge 生成 opinion."""
    opinions: List[JudgeOpinion] = []
    for persona in JUDGE_PERSONAS:
        t0 = time.time()
        overall, dims, concerns, strengths = _rule_judge_score(persona, text, sections, dim_scores)
        op = JudgeOpinion(
            judge=persona["name"],
            judge_cn=persona["name_cn"],
            icon=persona["icon"],
            overall_score=overall,
            dimension_scores=dims,
            reasoning=f"规则打分 (focus={persona['focus'][:30]}...)",
            key_concerns=concerns,
            key_strengths=strengths,
            generated_by="template",
            elapsed_ms=(time.time() - t0) * 1000,
        )
        opinions.append(op)
    return opinions


# ============================================================
# 4. LLM 打分 (有 LLM 时的真实实现)
# ============================================================

def _llm_judge_dimension(persona: Dict[str, Any], dimension: str,
                         text: str, model_router=None) -> Tuple[float, str]:
    """用 LLM 给单个 Judge 的单个维度打分.

    返回 (0~1 分数, 推理过程).
    若 model_router 为 None 或 LLM 不可用, 抛 RuntimeError, 让上层降级到规则.
    """
    if model_router is None:
        raise RuntimeError("model_router is None")

    rubric = persona.get("rubric", {}).get(dimension, "")
    if not rubric:
        return 0.5, "无 rubric"

    # 截断长文本, 避免 LLM context 爆掉
    snippet = text[:6000] if len(text) > 6000 else text

    prompt = (
        f"你是【{persona['name_cn']}】(关注: {persona['focus']}).\n"
        f"请评估以下宏观研究报告的【{DIMENSION_CN[dimension]}】维度.\n"
        f"评估问题: {rubric}\n\n"
        f"报告片段 (前 6000 字):\n---\n{snippet}\n---\n\n"
        "请输出严格 JSON (无多余文字):\n"
        '{"score": 0.0~1.0, "reasoning": "<50字内的判断理由>"}\n'
    )

    try:
        # 尝试调用 LLM (model_router 是 src.ai.router.Router 实例)
        # chat 返回 (text, model, prompt_tokens, completion_tokens) tuple
        # 关闭 thinking 模式, 节省 token + 加快响应 (我们只要 JSON score)
        resp = model_router.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=persona.get("temperature", 0.3),
            max_tokens=200,
            disable_thinking=True,
        )
        # resp 是 tuple: (text, model, pt, ct)
        content = (resp[0] if isinstance(resp, tuple) else resp).strip()
        # 提取 JSON (可能被包在 ```json ... ``` 里)
        m = re.search(r"\{[^{}]*\"score\"[^{}]*\}", content, re.DOTALL)
        if not m:
            raise RuntimeError(f"LLM 输出无 JSON: {content[:80]}")
        obj = json.loads(m.group(0))
        score = float(obj.get("score", 0.5))
        score = max(0.0, min(1.0, score))
        return score, obj.get("reasoning", "")[:120]
    except Exception as e:
        raise RuntimeError(f"LLM 调用失败: {e}")


def _llm_judge_persona(persona: Dict[str, Any], text: str,
                       model_router=None) -> Tuple[float, Dict[str, float], str]:
    """用 LLM 给单个 Judge 评所有维度 + 给出总分 + 推理."""
    dim_scores: Dict[str, float] = {}
    reasoning_parts: List[str] = []

    for d in DIMENSIONS:
        try:
            s, r = _llm_judge_dimension(persona, d, text, model_router)
            dim_scores[d] = s
            reasoning_parts.append(f"{DIMENSION_CN[d]}={s:.2f}")
        except Exception as e:
            logger.warning("LLM judge 失败 (persona=%s, dim=%s): %s", persona["name"], d, e)
            # 降级为规则
            s, r = _rule_score_dimension(d, text, _count_sections(text))
            dim_scores[d] = s
            reasoning_parts.append(f"{DIMENSION_CN[d]}={s:.2f}(rule)")

    # 用默认权重算总分
    overall = sum(dim_scores[d] * DIMENSION_WEIGHTS[d] for d in DIMENSIONS)
    reasoning = " | ".join(reasoning_parts)
    return max(0.0, min(1.0, overall)), dim_scores, reasoning


# ============================================================
# 5. 主入口
# ============================================================

def judge_report(
    report_text: str,
    target_date: str = "",
    source: str = "inline",
    pass_threshold: float = 0.65,
    model_router: Any = None,
    persist: bool = True,
) -> JudgeReport:
    """对宏观研究报告做 LLM-as-Judge 多维质量评估.

    参数:
        report_text:     报告全文 (Markdown 文本)
        target_date:     报告日期 (YYYY-MM-DD)
        source:          报告来源标识 (e.g. 'report:2026-06-12.md')
        pass_threshold:  通过门槛 (默认 0.65)
        model_router:    LLM 路由 (src.ai.router.Router 实例); None = 用规则降级
        persist:         是否把评分结果写入 SQLite (失败也不抛异常)

    返回:
        JudgeReport (含 5 维分数 / 4 Judge 意见 / 总体通过状态 / 改进建议)
    """
    t0 = time.time()
    text = report_text or ""
    sections = _count_sections(text)

    # --- 1. 计算 5 维分数 (优先 LLM, 降级规则) ---
    dim_scores: Dict[str, float] = {}
    dim_reasons: Dict[str, str] = {}
    use_llm = model_router is not None and _check_router_available(model_router)

    for d in DIMENSIONS:
        if use_llm:
            # 让第一个 Judge 用 LLM 算的维度分作为基准, 其它 Judge 再投票微调
            # 简化: 维度分 = 4 个 Judge 用 LLM 打分的均值
            scores_for_dim: List[float] = []
            for persona in JUDGE_PERSONAS:
                try:
                    s, _ = _llm_judge_dimension(persona, d, text, model_router)
                    scores_for_dim.append(s)
                except Exception as e:
                    logger.warning("LLM dim=%s judge=%s 失败, 降级: %s", d, persona["name"], e)
            if scores_for_dim:
                dim_scores[d] = sum(scores_for_dim) / len(scores_for_dim)
                dim_reasons[d] = f"LLM, {len(scores_for_dim)} judges"
                continue
        # 规则降级
        s, r = _rule_score_dimension(d, text, sections)
        dim_scores[d] = s
        dim_reasons[d] = r

    # --- 2. 各 Judge opinion ---
    judge_opinions: List[JudgeOpinion] = []
    if use_llm:
        for persona in JUDGE_PERSONAS:
            t1 = time.time()
            try:
                overall, dims, reasoning = _llm_judge_persona(persona, text, model_router)
                concerns: List[str] = []
                strengths: List[str] = []
                for d in DIMENSIONS:
                    s = dims[d]
                    if s < 0.4:
                        concerns.append(f"{DIMENSION_CN[d]}偏低 ({s:.2f})")
                    elif s >= 0.7:
                        strengths.append(f"{DIMENSION_CN[d]}表现好 ({s:.2f})")
                op = JudgeOpinion(
                    judge=persona["name"],
                    judge_cn=persona["name_cn"],
                    icon=persona["icon"],
                    overall_score=overall,
                    dimension_scores=dims,
                    reasoning=reasoning,
                    key_concerns=concerns,
                    key_strengths=strengths,
                    generated_by="llm",
                    elapsed_ms=(time.time() - t1) * 1000,
                )
            except Exception as e:
                logger.warning("LLM judge %s 整体失败, 用规则: %s", persona["name"], e)
                overall, dims, concerns, strengths = _rule_judge_score(
                    persona, text, sections, dim_scores
                )
                op = JudgeOpinion(
                    judge=persona["name"],
                    judge_cn=persona["name_cn"],
                    icon=persona["icon"],
                    overall_score=overall,
                    dimension_scores=dims,
                    reasoning=f"LLM 失败降级为规则: {e}",
                    key_concerns=concerns,
                    key_strengths=strengths,
                    generated_by="template",
                    elapsed_ms=(time.time() - t1) * 1000,
                )
            judge_opinions.append(op)
    else:
        judge_opinions = _build_rule_judge_opinions(text, dim_scores, sections)

    # --- 3. 维度级 DimensionScore (聚合 judge opinions) ---
    dim_score_list: List[DimensionScore] = []
    for d in DIMENSIONS:
        judge_vals = {op.judge: op.dimension_scores.get(d, 0.0) for op in judge_opinions}
        n = len(judge_vals)
        mean = sum(judge_vals.values()) / n if n else 0.0
        # std
        if n > 1:
            var = sum((v - mean) ** 2 for v in judge_vals.values()) / n
            std = var ** 0.5
        else:
            std = 0.0
        consensus = max(0.0, 1.0 - std * 2.5)  # std 0.4 → consensus 0
        weighted = mean * DIMENSION_WEIGHTS.get(d, 0.2)
        dim_score_list.append(DimensionScore(
            dimension=d,
            dimension_cn=DIMENSION_CN[d],
            scores=judge_vals,
            mean=mean,
            std=std,
            consensus=consensus,
            weighted=weighted,
        ))

    # --- 4. 总体 ---
    overall = sum(ds.weighted for ds in dim_score_list)
    consensus_level = sum(ds.consensus for ds in dim_score_list) / len(dim_score_list) if dim_score_list else 0.0
    pass_ = overall >= pass_threshold and consensus_level >= 0.4  # 既要够分, 也要有共识

    # --- 5. 改进建议 (针对未达标维度) ---
    improvements: List[Improvement] = []
    for ds in dim_score_list:
        if ds.mean < pass_threshold:
            target = max(pass_threshold, ds.mean + 0.2)
            sugs = _suggest_improvements(ds.dimension, text)
            improvements.append(Improvement(
                dimension=ds.dimension,
                dimension_cn=ds.dimension_cn,
                current_score=ds.mean,
                target_score=target,
                suggestions=sugs,
            ))

    # --- 6. summary ---
    if pass_:
        summary = f"✓ 通过 ({overall:.2f}, 共识 {consensus_level:.2f}) - {len(judge_opinions)} 个 Judge 一致认可"
    else:
        fail_reasons: List[str] = []
        if overall < pass_threshold:
            fail_reasons.append(f"总分 {overall:.2f} < 门槛 {pass_threshold}")
        if consensus_level < 0.4:
            fail_reasons.append(f"共识度 {consensus_level:.2f} 过低")
        summary = f"✗ 未通过 ({'; '.join(fail_reasons)})"

    elapsed_ms = (time.time() - t0) * 1000

    # --- 7. 落库 (失败不影响主流程) ---
    if persist:
        try:
            _persist_to_db(target_date, source, overall, consensus_level, pass_, dim_scores)
        except Exception as e:
            logger.warning("落库失败 (不影响评分): %s", e)

    report = JudgeReport(
        target_date=target_date,
        source=source,
        text_length=len(text),
        section_count=sections,
        dimension_scores=dim_score_list,
        judge_opinions=judge_opinions,
        overall_score=overall,
        consensus_level=consensus_level,
        pass_threshold=pass_threshold,
        pass_=pass_,
        improvements=improvements,
        generated_by="llm" if use_llm else "template",
        judges_used=len(judge_opinions),
        dimensions_used=len(dim_score_list),
        total_latency_ms=elapsed_ms,
        summary=summary,
    )
    logger.info("judge_report 完成: target=%s overall=%.2f pass=%s elapsed=%.0fms",
                target_date, overall, pass_, elapsed_ms)
    return report


def _check_router_available(model_router: Any) -> bool:
    """检查 model_router 是否真的可用 (有 chat 方法 + 不抛异常)."""
    if model_router is None:
        return False
    if not hasattr(model_router, "chat"):
        return False
    try:
        # 试探一下: 用极小 prompt, 看是否成功
        # 关闭 thinking 模式, 避免 max_tokens=5 被 <think>...</think> 占满
        resp = model_router.chat(
            messages=[{"role": "user", "content": "ping"}],
            temperature=0.0,
            max_tokens=20,
            disable_thinking=True,
        )
        # chat 返回 (text, model, pt, ct) tuple
        if not resp:
            return False
        text = resp[0] if isinstance(resp, tuple) else resp
        return bool(text and text.strip())
    except Exception:
        return False


def _suggest_improvements(dimension: str, text: str) -> List[str]:
    """针对未达标维度, 给出改进建议 (规则版)."""
    sugs: List[str] = []
    if dimension == "consistency":
        if not text:
            sugs.append("增加报告主体内容")
        else:
            contradictions = sum(1 for p in _INCONSISTENCY_PATTERNS if p.search(text))
            if contradictions > 0:
                sugs.append(f"检测到 {contradictions} 处可能的文内矛盾, 建议统一立场表达")
            sugs.append("确保章节间立场一致, 用 '综上' / '本报告判断' 等显式锚点句")

    elif dimension == "groundedness":
        hits = _count_keyword_hits(text, _GROUND_KEYWORDS)
        if hits < 10:
            sugs.append(f"当前数据/引用仅 {hits} 处, 建议补充至 15+ (GDP/CPI/PMI/利率/同比环比 等)")
        sugs.append("每个核心结论后加具体数据出处 (来源: 统计局 / 央行 / 部委文件 / Wind)")
        sugs.append("避免 '大幅' / '显著' 等空洞副词, 用数字代替")

    elif dimension == "completeness":
        sections = _count_sections(text)
        if sections < 20:
            sugs.append(f"当前仅 {sections} 节, 建议按 v9 模板补全至 31 节")
        missing = [d for d in ["货币", "财政", "产业", "外贸", "就业", "通胀"] if d not in text]
        if missing:
            sugs.append(f"五大领域缺失: {', '.join(missing)}")
        sugs.append("补全: 风险章节 / 情景章节 / 战略章节")

    elif dimension == "actionability":
        hits = _count_keyword_hits(text, _ACTION_KEYWORDS)
        if hits < 6:
            sugs.append(f"行动建议关键词仅 {hits} 处, 建议加 8+ 明确动作")
        sugs.append("为每条建议加 时间窗口 (短期 1 周 / 中期 1 月 / 长期 1 季) + 触发条件 + 风险预案")

    elif dimension == "novelty":
        hits = _count_keyword_hits(text, _NOVEL_KEYWORDS)
        if hits < 3:
            sugs.append(f"新颖信号仅 {hits} 处, 建议至少 4 处 '首次/拐点/超预期/异动' 类识别")
        sugs.append("尝试提出 1-2 条反共识判断 + 论证, 提升报告锐度")

    return sugs


def _persist_to_db(target_date: str, source: str, overall: float,
                   consensus: float, pass_: bool, dim_scores: Dict[str, float]) -> None:
    """把评分结果写入 SQLite (复用 economy_timeseries 表, key='llm_judge_overall' 等)."""
    try:
        from src.storage import db as db_mod
        rows = [
            ("llm_judge_overall", overall),
            ("llm_judge_consensus", consensus),
            ("llm_judge_pass", 1.0 if pass_ else 0.0),
        ]
        for d, s in dim_scores.items():
            rows.append((f"llm_judge_{d}", s))
        # 用 daily_metric 表, metric_date=target_date
        db_mod.upsert_daily_metrics(
            metric_date=target_date or "1970-01-01",
            metrics=rows,
        )
        logger.info("llm_judge 落库完成 (%d 行)", len(rows))
    except Exception as e:
        logger.warning("llm_judge 落库失败: %s", e)


# ============================================================
# 6. 便捷函数
# ============================================================

def quick_score(report_text: str, target_date: str = "") -> Dict[str, Any]:
    """快速打分, 返回简化 dict (无 Judge 细节)."""
    r = judge_report(report_text, target_date=target_date)
    return {
        "target_date": r.target_date,
        "overall": round(r.overall_score, 3),
        "consensus": round(r.consensus_level, 3),
        "pass": r.pass_,
        "dimensions": {ds.dimension: round(ds.mean, 3) for ds in r.dimension_scores},
        "judges": {op.judge: round(op.overall_score, 3) for op in r.judge_opinions},
        "summary": r.summary,
        "improvements": [i.as_dict() for i in r.improvements],
    }


def list_judges() -> List[Dict[str, Any]]:
    """返回所有 Judge 配置 (供 API / Dashboard 展示)."""
    return [
        {
            "name": p["name"],
            "name_cn": p["name_cn"],
            "icon": p["icon"],
            "focus": p["focus"],
            "weight": p["weight"],
        }
        for p in JUDGE_PERSONAS
    ]


def list_dimensions() -> List[Dict[str, Any]]:
    """返回所有维度配置 (供 API / Dashboard 展示)."""
    return [
        {
            "key": d,
            "name_cn": DIMENSION_CN[d],
            "weight": DIMENSION_WEIGHTS[d],
        }
        for d in DIMENSIONS
    ]


__all__ = [
    "DIMENSIONS", "DIMENSION_CN", "DIMENSION_WEIGHTS",
    "JUDGE_PERSONAS",
    "DimensionScore", "JudgeOpinion", "Improvement", "JudgeReport",
    "judge_report", "quick_score", "list_judges", "list_dimensions",
]