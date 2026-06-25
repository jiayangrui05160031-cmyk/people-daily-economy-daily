"""prompts - 10 个分析任务的提示词模板 (v2)

设计原则:
- JSON schema 必须显式,使用完整示例 (避免 M2.5 把对象列表退化成字符串列表)
- 关键洞察类任务要求带引用标号 [文章 i],供 citations 模块解析
- 6 个原任务 + 4 个新任务 (sentiment / events / cross_links / self_eval)
"""
from __future__ import annotations

from typing import Any, Dict, List

# ============================================================
# 通用系统提示词
# ============================================================
SYSTEM_PROMPT = """你是中国宏观经济政策分析专家,负责解读《人民日报》每日经济新闻。
输出要求:
1) 客观、准确、有数据支撑;
2) 专业财经语言但通俗易懂;
3) 严格按用户指定的 JSON 格式返回,字段名、字段类型、嵌套结构必须完全一致;
4) 不输出 JSON 之外的任何文字、解释或 markdown 围栏。"""

# ============================================================
# 强制 JSON 输出约束 (追加到每个用户 prompt 末尾)
# ============================================================
JSON_GUARD = """

【重要 - 输出格式约束】
1) 你的回复必须是合法 JSON 对象,不允许任何额外文字。
2) 字段名、字段类型、嵌套结构必须与下方示例完全一致。
3) 数组元素必须为对象,不能退化为字符串列表。
4) 数值字段不要加引号,布尔字段输出 true/false。
5) 中文内容字段中不要包含未转义的双引号。"""

# ============================================================
# 任务 1:昨日主题词
# ============================================================
THEME_KEYWORDS_PROMPT = """任务:基于以下《人民日报》昨日经济新闻标题与正文,提炼 8-12 个主题词。

要求:
- 每个主题词 2-4 字,能代表一篇文章的核心议题
- 按重要性排序 (score 越高越重要)
- 附 1 句话解释 (来自原文事实)
- 严禁只返回字符串数组,keywords 的每个元素必须是含 word/score/explain 的对象

返回 JSON 格式示例 (字段类型必须完全匹配):
{{
  "keywords": [
    {{"word": "降准", "score": 0.95, "explain": "央行降准 1 万亿支持实体经济"}},
    {{"word": "新能源汽车", "score": 0.88, "explain": "5 月销量同比 +38.5%,渗透率破 47%"}}
  ]
}}

文章列表:
{articles_text}
""" + JSON_GUARD

# ============================================================
# 任务 2:政策风向
# ============================================================
POLICY_DIRECTION_PROMPT = """任务:分析以下《人民日报》昨日经济新闻,判断当前政策风向。

要求:
- direction 必须是以下三个值之一: "扩张" / "收紧" / "中性"
  · 扩张: 刺激经济、加大投资、降准降息、促消费
  · 收紧: 去杠杆、收紧信贷、调控降温
  · 中性: 日常监管、结构优化、规则完善
- keywords 数组 3-5 个政策关键词 (字符串)
- interpretation 100 字以内

返回 JSON 格式示例:
{{
  "direction": "扩张",
  "confidence": 0.85,
  "keywords": ["降准", "消费券", "设备更新"],
  "interpretation": "降准释放长期资金 1 万亿,叠加消费刺激,体现明显的政策扩张倾向。"
}}

文章列表:
{articles_text}
""" + JSON_GUARD

# ============================================================
# 任务 3:重点产业
# ============================================================
INDUSTRY_FOCUS_PROMPT = """任务:识别以下《人民日报》昨日经济新闻中重点报道的产业/行业。

要求:
- 挑选 3-6 个重点产业
- heat 字段值必须是: "高" / "中" / "低"
- article_count 必须是正整数 (1-999)
- industries 数组的每个元素必须为含 5 个字段的对象,不能简化为字符串

返回 JSON 格式示例:
{{
  "industries": [
    {{"name": "新能源", "heat": "高", "article_count": 8, "summary": "销量持续增长,渗透率破 47%", "stance": "利好"}},
    {{"name": "半导体", "heat": "中", "article_count": 5, "summary": "国产化加速推进", "stance": "利好"}}
  ]
}}

stance 取值: "利好" / "利空" / "中性"

文章列表:
{articles_text}

预扫描提示 (NLP 已识别出命中产业):
{industry_hint}
""" + JSON_GUARD

# ============================================================
# 任务 4:重点政策出台
# ============================================================
POLICY_DETAIL_PROMPT = """任务:从以下《人民日报》昨日经济新闻中,提取 3-5 条"重点政策出台"事件。

要求:
- 每条政策必须包含 4 个字段
- title 15 字以内,issuer 2-20 字,content 80-150 字
- 若当日无明确政策,返回空数组 {{"policies": []}}

返回 JSON 格式示例:
{{
  "policies": [
    {{
      "title": "央行降准 1 万亿",
      "issuer": "中国人民银行",
      "content": "下调金融机构存款准备金率 0.5 个百分点,释放长期资金约 1 万亿元,支持实体经济。",
      "source_article": "央行宣布降准 1 万亿"
    }}
  ]
}}

文章列表:
{articles_text}
""" + JSON_GUARD

# ============================================================
# 任务 5:核心信息
# ============================================================
CORE_INSIGHTS_PROMPT = """任务:基于以下《人民日报》昨日经济新闻,提炼一段 150-200 字的"核心信息"摘要。

要求:
- 用一段话概述昨日经济新闻的核心主线
- 突出政策、产业、市场三类关键信息
- 语言精炼,适合决策者 30 秒阅读
- 不要用列表,只用流畅段落

返回 JSON 格式示例:
{{
  "insights": "昨日经济新闻聚焦货币政策宽松与产业升级两条主线。央行降准 1 万亿向市场注入长期流动性,..."
}}

文章列表:
{articles_text}
""" + JSON_GUARD

# ============================================================
# 任务 6:未来发展判断
# ============================================================
FUTURE_OUTLOOK_PROMPT = """任务:基于以下《人民日报》昨日经济新闻的核心政策与产业动态,对未来 1-3 个月的发展趋势做判断。

要求:
- 给出 3-5 条具体判断 (不是泛泛而谈)
- topic 15 字以内,judgment 取值: "支持" / "中性" / "警惕"
- rationale 50-80 字
- outlooks 数组的每个元素必须为含 3 个字段的对象

返回 JSON 格式示例:
{{
  "outlooks": [
    {{"topic": "货币宽松延续", "judgment": "支持", "rationale": "降准信号明确,后续 LPR 仍有下调空间。"}},
    {{"topic": "地产去库存", "judgment": "中性", "rationale": "城中村改造提速,但销售端仍待观察。"}}
  ]
}}

文章列表:
{articles_text}
""" + JSON_GUARD

# ============================================================
# 任务 7 (新):立场 / 情感
# ============================================================
SENTIMENT_PROMPT = """任务:从以下《人民日报》昨日经济新闻中,识别主要产业/主体的市场立场与情感倾向。

要求:
- 抽取 3-6 个目标 (target): 产业名 (新能源、半导体) 或政策主体 (央行、地产)
- stance 取值: "利好" / "利空" / "中性"
- intensity 0.0-1.0,evidence 4-60 字 (引用原文中支持该判断的关键事实)
- items 数组每个元素必须是 4 字段对象

返回 JSON 格式示例:
{{
  "items": [
    {{"target": "新能源汽车", "stance": "利好", "intensity": 0.85, "evidence": "销量同比 +38.5%,渗透率破 47%"}},
    {{"target": "房地产", "stance": "利空", "intensity": 0.70, "evidence": "投资持续下滑,销售尚未回暖"}}
  ]
}}

文章列表:
{articles_text}
""" + JSON_GUARD

# ============================================================
# 任务 8 (新):事件抽取 (主谓宾三元组)
# ============================================================
EVENTS_PROMPT = """任务:从以下《人民日报》昨日经济新闻中,抽取 4-8 条核心事件 (主谓宾三元组)。

要求:
- subject (主语): 2-20 字,通常是机构或主体 (央行、工信部、某产业)
- action (谓语): 2-30 字,具体动作 (宣布降准、发布指引、销量增长)
- object (宾语): 2-30 字,动作的承受者 (释放 1 万亿资金、新能源汽车销量)
- event_type 取值: "货币政策" / "财政政策" / "产业政策" / "贸易政策" / "监管政策" / "市场动态" / "其他"
- impact 0-60 字,可能影响

返回 JSON 格式示例:
{{
  "events": [
    {{
      "subject": "中国人民银行",
      "action": "宣布下调存款准备金率 0.5 个百分点",
      "object": "释放长期资金约 1 万亿元",
      "event_type": "货币政策",
      "impact": "利好银行、地产、消费板块"
    }},
    {{
      "subject": "工信部",
      "action": "发布新型储能制造业高质量发展行动方案",
      "object": "到 2027 年培育 3-5 家生态主导型企业",
      "event_type": "产业政策",
      "impact": "利好新型储能产业链"
    }}
  ]
}}

文章列表:
{articles_text}
""" + JSON_GUARD

# ============================================================
# 任务 9 (新):跨文章关联 (聚类)
# ============================================================
CROSS_LINKS_PROMPT = """任务:对以下多篇《人民日报》经济新闻做聚类,识别哪些文章在报道同一主题或同一事件。

要求:
- 找出 2-5 个聚类,每个聚类至少包含 2 篇文章
- article_indices 是文章序号数组 (从 1 开始),例如 [1, 3, 5]
- cluster 字段是该聚类主题 (2-20 字)
- summary 10-100 字,说明这些文章的共同点

返回 JSON 格式示例:
{{
  "links": [
    {{
      "cluster": "央行降准",
      "article_indices": [1, 3, 7],
      "summary": "三篇文章分别从总量、利率、银行影响三角度报道央行降准"
    }},
    {{
      "cluster": "新能源汽车产业",
      "article_indices": [2, 5],
      "summary": "报道 5 月销量与出口高增长"
    }}
  ]
}}

文章列表:
{articles_text}
""" + JSON_GUARD

# ============================================================
# 任务 10 (新):AI 自评 (第二阶段,依赖前 9 个任务结果)
# ============================================================
SELF_EVAL_PROMPT = """任务:你是一个严谨的 AI 质量评审员。请基于以下"前 9 个任务的输出",从 3 个维度打分并写评语。

评分维度 (0.0-1.0):
- consistency (一致性): 各任务结论是否自洽 (例如政策风向"扩张"是否与重点政策、核心信息一致)
- groundedness (有据性): 每条结论是否能追溯到原文 (引用是否充分)
- completeness (完整性): 是否覆盖了政策 / 产业 / 市场 / 立场 / 事件 / 关联 6 个视角

要求:
- 3 个分值必须独立给出,允许存在差异
- comments 10-200 字,既要肯定优点,也要点出 1-2 条可改进之处

返回 JSON 格式示例:
{{
  "consistency": 0.88,
  "groundedness": 0.92,
  "completeness": 0.85,
  "comments": "整体自洽,主线清晰。政策风向与重点政策呼应良好。需注意 cross_links 覆盖面可进一步扩大。"
}}

前 9 个任务的输出:
{prior_results}
""" + JSON_GUARD

# ============================================================
# 辅助函数:把 Article 列表格式化为 LLM 输入
# ============================================================
def format_articles_for_llm(articles, max_chars: int = 8000, per_article_limit: int = 400) -> str:
    """把 Article 列表格式化为纯文本,限制总长度以避免超 token。"""
    blocks: List[str] = []
    total_len = 0
    for i, art in enumerate(articles, 1):
        title = getattr(art, "title", None) or "(无标题)"
        time_str = getattr(art, "publish_time", None) or "?"
        source = getattr(art, "source", None) or "?"
        channel = getattr(art, "channel", None) or "?"
        content = (getattr(art, "content_text", None) or "")[:per_article_limit]
        block = f"[{i}] {title}\n时间: {time_str} | 栏目: {channel} | 来源: {source}\n{content}\n"
        if total_len + len(block) > max_chars:
            break
        blocks.append(block)
        total_len += len(block)
    return "\n".join(blocks)


def format_industry_hint(nlp_stats) -> str:
    """把 NLP 识别的产业命中作为提示给 LLM,辅助二次归纳。"""
    if not nlp_stats or not getattr(nlp_stats, "industry_hits", None):
        return "无(请根据内容自行判断)"
    items = sorted(nlp_stats.industry_hits.items(), key=lambda x: x[1], reverse=True)
    lines = [f"- {name}: 命中 {count} 次" for name, count in items[:10]]
    return "\n".join(lines)


# ============================================================
# 任务注册表 (供 analyzer 调度)
# ============================================================
TASK_REGISTRY: Dict[str, str] = {
    "theme_keywords": THEME_KEYWORDS_PROMPT,
    "policy_direction": POLICY_DIRECTION_PROMPT,
    "industries": INDUSTRY_FOCUS_PROMPT,
    "policies": POLICY_DETAIL_PROMPT,
    "core_insights": CORE_INSIGHTS_PROMPT,
    "outlooks": FUTURE_OUTLOOK_PROMPT,
    "sentiment": SENTIMENT_PROMPT,
    "events": EVENTS_PROMPT,
    "cross_links": CROSS_LINKS_PROMPT,
}


if __name__ == "__main__":
    for name, tmpl in TASK_REGISTRY.items():
        rendered = tmpl.format(articles_text="(test)", industry_hint="(test)")
        assert "{{" not in rendered and "}}" not in rendered, f"{name} 含未替换占位符"
    print(f"prompts 自检: {len(TASK_REGISTRY)} 个任务模板全部可格式化")
    rendered = SELF_EVAL_PROMPT.format(prior_results="(test)")
    assert "{{" not in rendered and "}}" not in rendered
    print("SELF_EVAL_PROMPT 模板可格式化")
    print("\nAll prompts self-tests passed")
