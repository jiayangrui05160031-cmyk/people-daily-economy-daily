"""v8 前沿: 政策 PDF 解析器 (Policy PDF Parser)

对政策文件 PDF 做深度结构化提取, 输出事件/产业/资金/时间窗四元组。
支持:
- pdfplumber 文本提取 (含表格)
- pypdf 兜底
- LLM 辅助结构化 (minimax-M3)
- 关键字段识别: 政策名 / 发布机构 / 文号 / 发布日期 / 涉及产业 / 资金规模 / 生效日期

依赖: pdfplumber, pypdf, jieba
"""
from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jieba

from src.utils.logger import get_logger

logger = get_logger("analysis.policy_pdf")


# ============================================================
# 数据类
# ============================================================
@dataclass
class PolicyClause:
    """政策条款 (子条目)."""
    clause_no: str
    text: str
    keywords: List[str] = field(default_factory=list)
    industries: List[str] = field(default_factory=list)
    amount: str = ""
    effective_date: str = ""
    stance: str = "中性"     # 利好 / 利空 / 中性
    confidence: float = 0.0

    def as_dict(self):
        return asdict(self)


@dataclass
class PolicyDocument:
    """解析后的政策文件."""
    source_path: str
    title: str = ""
    issuer: str = ""          # 发布机构 (如 国务院 / 中国人民银行)
    doc_number: str = ""      # 文号 (如 国发〔2024〕15号)
    publish_date: str = ""    # 发布日期
    effective_date: str = ""  # 生效日期
    total_pages: int = 0
    total_chars: int = 0
    full_text: str = ""
    clauses: List[PolicyClause] = field(default_factory=list)
    industries_mentioned: List[str] = field(default_factory=list)
    amount_total: str = ""
    key_metrics: Dict[str, str] = field(default_factory=dict)   # 资金/比例/规模
    overall_stance: str = "中性"
    confidence: float = 0.0
    parse_method: str = "regex"   # regex / llm / hybrid
    parse_warnings: List[str] = field(default_factory=list)
    summary: str = ""

    def as_dict(self):
        d = asdict(self)
        d["clauses"] = [c.as_dict() for c in self.clauses]
        return d


# ============================================================
# 1) 文本提取
# ============================================================
def _extract_text_from_pdf(pdf_path: str) -> Tuple[str, int, List[str]]:
    """返回 (full_text, total_pages, warnings). 优先 pdfplumber."""
    warnings: List[str] = []
    text_parts: List[str] = []
    pages = 0
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            pages = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                t = page.extract_text() or ""
                if t:
                    text_parts.append(t)
                # 表格
                tables = page.extract_tables() or []
                for tab in tables:
                    for row in tab:
                        if row and any(c for c in row):
                            text_parts.append(" | ".join(str(c) for c in row if c))
    except Exception as e:
        warnings.append(f"pdfplumber 失败: {e}")
        # 兜底 pypdf
        try:
            import pypdf
            reader = pypdf.PdfReader(pdf_path)
            pages = len(reader.pages)
            for p in reader.pages:
                t = p.extract_text() or ""
                if t:
                    text_parts.append(t)
        except Exception as e2:
            warnings.append(f"pypdf 也失败: {e2}")
    return "\n".join(text_parts), pages, warnings


# ============================================================
# 2) 字段正则识别
# ============================================================
_TITLE_RE = re.compile(r"^.{0,40}?(?:关于|印发|通知|意见|办法|规划|方案|纲要|指引|措施|条例).{0,80}", re.MULTILINE)
_ISSUER_HINTS = (
    "国务院", "中共中央", "中央办公厅", "国务院办公厅",
    "中国人民银行", "财政部", "国家发改委", "国家发展改革委",
    "工信部", "工信部", "商务部", "证监会", "银保监会", "国家金融监督管理总局",
    "农业农村部", "住建部", "国家统计局", "海关总署", "国家税务总局", "国资委",
    "中国证券监督管理委员会", "国家市场监督管理总局", "国家能源局", "国家网信办",
    "工业和信息化部", "国家卫生健康委员会", "国家医疗保障局", "国家知识产权局",
    "中国人民银行", "中共中央", "全国人大常委会",
)
_DOCNUM_RE = re.compile(r"([\u4e00-\u9fff]+[〔【]?\d{4}[〕】]?\s?\d+\s?号)")
_DATE_RE = re.compile(r"(d{4}s*年s*d{1,2}s*月s*d{1,2}s*日)")
_EFFECT_RE = re.compile(r"(?:自|从|于|自发布之日起施行|自.{0,15}起施行|自d{4}年d{1,2}月d{1,2}日起施行)")
_AMOUNT_RE = re.compile(r"([d,.]+s*(?:亿元|万元|万亿|百万元|美元|元))")
_INDUSTRY_KEYWORDS = {
    "新能源": ("新能源", "光伏", "风电", "储能", "氢能", "锂电池", "充电桩"),
    "新能源汽车": ("新能源汽车", "电动汽车", "造车新势力", "渗透率"),
    "半导体": ("半导体", "集成电路", "芯片", "晶圆", "EDA", "光刻机", "封测"),
    "人工智能": ("人工智能", "大模型", "AIGC", "算力", "AGI", "人形机器人", "AI"),
    "数字经济": ("数字经济", "数据要素", "数字人民币", "工业互联网", "区块链"),
    "房地产": ("房地产", "楼市", "房企", "保交楼", "保障房", "城中村"),
    "金融": ("金融", "银行", "证券", "基金", "保险", "资本市场", "LPR", "MLF"),
    "消费": ("消费", "内需", "以旧换新", "消费券", "下沉市场"),
    "医疗健康": ("医药", "创新药", "集采", "医疗器械", "养老", "银发经济"),
    "制造业": ("制造业", "工业", "高端装备", "智能制造"),
    "基础设施": ("基建", "新基建", "5G", "特高压", "城际铁路", "数据中心"),
    "外贸": ("外贸", "进出口", "跨境电商", "一带一路", "RCEP", "自贸区"),
    "农业": ("农业", "种业", "粮食", "高标准农田", "乡村振兴"),
}
_STANCE_KEYWORDS = {
    "利好": ("支持", "鼓励", "促进", "扩大", "提升", "加快", "推动", "落实", "实施", "加强", "完善", "推进", "深化", "优化", "降低", "减免", "补贴", "奖励", "落实"),
    "利空": ("限制", "禁止", "淘汰", "出清", "整顿", "处罚", "罚款", "约谈", "收紧", "从严", "压减", "遏制"),
}


def _detect_industries(text: str) -> List[str]:
    found = []
    for ind, keys in _INDUSTRY_KEYWORDS.items():
        if any(k in text for k in keys):
            found.append(ind)
    return found


def _detect_stance(text: str) -> Tuple[str, float]:
    pos = sum(text.count(k) for k in _STANCE_KEYWORDS["利好"])
    neg = sum(text.count(k) for k in _STANCE_KEYWORDS["利空"])
    total = pos + neg
    if total == 0:
        return "中性", 0.0
    p_pos = pos / total
    if p_pos > 0.65:
        return "利好", min(1.0, (p_pos - 0.5) * 2)
    if p_pos < 0.35:
        return "利空", min(1.0, (0.5 - p_pos) * 2)
    return "中性", 1.0 - abs(p_pos - 0.5) * 2


def _split_clauses(text: str) -> List[Tuple[str, str]]:
    """按 '第X条' / '一、' / '1.' 切分, 返回 (clause_no, clause_text)."""
    pattern = re.compile(r"(?:^|\n)\s*(第[一二三四五六七八九十百千]+条|[一二三四五六七八九十]+、|\d+\.|（\d+）)")
    parts = pattern.split(text)
    out: List[Tuple[str, str]] = []
    if not parts:
        return [("", text[:2000])]
    # parts 格式: [pre, no1, body1, no2, body2, ...]
    pre = parts[0].strip()
    if pre:
        out.append(("", pre[:1500]))
    for i in range(1, len(parts) - 1, 2):
        cno = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if body and len(body) > 5:
            out.append((cno, body[:2000]))
    return out or [("", text[:2000])]


# ============================================================
# 3) 主入口
# ============================================================
def parse(pdf_path: str, use_llm: bool = False, router=None) -> PolicyDocument:
    """解析政策 PDF. 可选 LLM 增强结构化."""
    doc = PolicyDocument(source_path=str(pdf_path))
    if not Path(pdf_path).exists():
        doc.parse_warnings.append(f"文件不存在: {pdf_path}")
        return doc
    # 1) 提取文本
    full_text, pages, warnings = _extract_text_from_pdf(pdf_path)
    doc.full_text = full_text
    doc.total_pages = pages
    doc.total_chars = len(full_text)
    doc.parse_warnings.extend(warnings)
    if not full_text.strip():
        doc.parse_warnings.append("PDF 文本提取为空 (可能为扫描件)")
        return doc
    # 2) 字段识别
    m = _TITLE_RE.search(full_text)
    if m:
        doc.title = m.group(0).strip()[:80]
    if not doc.title:
        doc.title = Path(pdf_path).stem[:60]
    for hint in _ISSUER_HINTS:
        if hint in full_text[:3000]:
            doc.issuer = hint
            break
    m = _DOCNUM_RE.search(full_text)
    if m:
        doc.doc_number = m.group(1).strip()[:30]
    m = _DATE_RE.search(full_text[:3000])
    if m:
        doc.publish_date = m.group(1).replace(" ", "")
    m = _EFFECT_RE.search(full_text)
    if m:
        doc.effective_date = m.group(0)[:30]
    # 3) 资金识别 (前 10 个)
    amounts = _AMOUNT_RE.findall(full_text)
    if amounts:
        doc.amount_total = amounts[0]
        doc.key_metrics["金额_提及"] = " | ".join(amounts[:5])
    # 4) 行业识别
    doc.industries_mentioned = _detect_industries(full_text)
    # 5) 立场判定
    stance, conf = _detect_stance(full_text)
    doc.overall_stance = stance
    doc.confidence = round(conf, 3)
    # 6) 条款切分
    raw_clauses = _split_clauses(full_text)[:20]
    for cno, body in raw_clauses:
        ks = [w for w, _ in __import__("collections").Counter(jieba.cut(body)).most_common(8) if len(w) > 1]
        inds = _detect_industries(body)
        amt = ""
        m = _AMOUNT_RE.search(body)
        if m:
            amt = m.group(1)
        eff = ""
        m = _EFFECT_RE.search(body)
        if m:
            eff = m.group(0)[:20]
        c_stance, c_conf = _detect_stance(body)
        doc.clauses.append(PolicyClause(
            clause_no=cno, text=body[:600],
            keywords=ks, industries=inds,
            amount=amt, effective_date=eff,
            stance=c_stance, confidence=round(c_conf, 3),
        ))
    # 7) LLM 增强 (可选)
    if use_llm and router is not None and getattr(router, "api_key", None):
        try:
            _llm_enhance(doc, router)
        except Exception as e:
            doc.parse_warnings.append(f"LLM 增强失败: {e}")
    doc.parse_method = "llm" if "LLM" in doc.parse_method else "regex"
    # 8) 摘要
    doc.summary = (
        f"《{doc.title or '(无标题)'}》发布于 {doc.publish_date or '?'}, "
        f"主体: {doc.issuer or '?'}, 文号: {doc.doc_number or '无'}, "
        f"立场: {doc.overall_stance} (置信 {doc.confidence:.0%}), "
        f"涉及产业 {len(doc.industries_mentioned)} 个, "
        f"条款 {len(doc.clauses)} 条, 资金: {doc.amount_total or '无'}"
    )
    return doc


def _llm_enhance(doc: PolicyDocument, router) -> None:
    """用 LLM 做关键摘要 + 行业映射 + 影响传导."""
    if len(doc.full_text) > 8000:
        text = doc.full_text[:4000] + "\n...\n" + doc.full_text[-4000:]
    else:
        text = doc.full_text
    prompt = (
        "你是宏观政策分析专家。给定以下政策文本, 严格返回 JSON:\n"
        "{\n"
        '  "summary": "80 字以内政策核心",\n'
        '  "key_targets": ["具体扶持对象/行业/企业, 3-5 条"],\n'
        '  "investment_thesis": "对 A 股哪些板块利好/利空, 1-2 句",\n'
        '  "implementation_window": "时间窗 (立即/3 个月/1 年等)",\n'
        '  "macro_impact": "对 GDP/CPI/就业 的潜在影响 1 句"\n'
        "}\n只输出 JSON, 不要其他文字。\n\n"
        f"政策文本:\n{text}"
    )
    text_out, model, pt, ct = router.chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1200,
        use_json_mode=True,
        disable_thinking=True,
    )
    import re as _re
    m = _re.search(r"\{[\s\S]*\}", text_out or "")
    if not m:
        return
    d = json.loads(m.group(0))
    doc.parse_method = "llm"
    doc.summary = d.get("summary", doc.summary) or doc.summary
    extras = doc.key_metrics
    extras["llm_thesis"] = d.get("investment_thesis", "")[:200]
    extras["llm_window"] = d.get("implementation_window", "")
    extras["llm_macro_impact"] = d.get("macro_impact", "")
    if d.get("key_targets"):
        extras["llm_key_targets"] = " | ".join(d["key_targets"][:5])


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.analysis.policy_pdf <pdf_path>")
        sys.exit(1)
    doc = parse(sys.argv[1])
    print(f"Title: {doc.title}")
    print(f"Issuer: {doc.issuer}  Date: {doc.publish_date}")
    print(f"Industries: {doc.industries_mentioned}")
    print(f"Stance: {doc.overall_stance}  Amount: {doc.amount_total}")
    print(f"Summary: {doc.summary}")
    print(f"Clauses: {len(doc.clauses)}")
