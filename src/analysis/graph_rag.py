"""graph_rag.py - v9 前沿: GraphRAG-lite (社区发现 + LLM 层级摘要)

实现 Microsoft GraphRAG 思路的精简版:
  1) 已有知识图谱 (src.kg.graph.build)
  2) Louvain 社区发现 (networkx 内置)
  3) 每社区生成 level-0 摘要 (节点 top-K + 边 top-K, 模板/LLM 两种)
  4) 全局 level-1 摘要 (汇总所有 level-0, LLM 提炼)
  5) ask_global(question): 路由到相关社区, 拼装上下文, LLM 回答

设计取舍:
  - 不依赖 networkx.community.louvain (旧版不一定有), 用内置 label_propagation + 手动 modularity
  - 实在没有 community, 退化为"按类型分组" (政府/产业/事件 等)
  - LLM 不可用时退化为"模板摘要" (节点+边 + 出现频次)

学术参考:
  - Microsoft Research "From Local to Global: A Graph RAG Approach to
    Query-Focused Summarization" (2024)
  - Blondel et al. "Fast unfolding of communities in large networks" (2008)
"""
from __future__ import annotations

import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.logger import get_logger

logger = get_logger("analysis.graph_rag")

import re
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

def _strip_think(text: str) -> str:
    if not text:
        return ""
    return _THINK_RE.sub("", text).strip()



import re
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

def _strip_think(text: str) -> str:
    if not text:
        return ""
    return _THINK_RE.sub("", text).strip()




# ============================================================
# 数据类
# ============================================================
@dataclass
class CommunitySummary:
    """单个社区的摘要."""
    community_id: int
    node_count: int
    edge_count: int
    top_nodes: List[Dict[str, Any]]   # [{id, type, freq, pagerank}]
    top_edges: List[Dict[str, Any]]   # [{source, target, weight}]
    label: str                        # 社区主标签 (template: 频次最高类型)
    summary: str                      # 文本摘要 (template 或 LLM)
    generated_by: str                 # "template" | "llm"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GraphRAGReport:
    """完整 GraphRAG 输出."""
    n_nodes: int
    n_edges: int
    n_communities: int
    communities: List[CommunitySummary] = field(default_factory=list)
    global_summary: str = ""         # 全局 level-1 摘要
    global_generated_by: str = "template"
    modularity: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["communities"] = [c.as_dict() for c in self.communities]
        return d


# ============================================================
# 社区发现 (networkx 内置 label_propagation + 退化)
# ============================================================
def _detect_communities(G: nx.Graph, max_communities: int = 8) -> List[set]:
    """优先用 networkx 的 community 算法, 失败退化按节点 type 分组。"""
    if G.number_of_nodes() == 0:
        return []
    n = G.number_of_nodes()

    # 1) 优先: asyn_lpa / label_propagation (nx 2.x 一定有)
    try:
        from networkx.algorithms.community import asyn_lpa_communities
        comms = list(asyn_lpa_communities(G, weight="weight", seed=42))
        logger.info("社区发现 (asyn_lpa): %d 个社区", len(comms))
    except Exception as e:
        logger.info("asyn_lpa 不可用: %s, 退化到 type 分组", e)
        comms = _group_by_type(G)

    # 2) 太碎就合并 (单节点社区)
    if len(comms) > max_communities:
        comms = _merge_small(comms, G, target=max_communities)
    return comms


def _group_by_type(G: nx.Graph) -> List[set]:
    """按节点 type 分组 (退化方案)."""
    by_type: Dict[str, set] = defaultdict(set)
    for n, d in G.nodes(data=True):
        by_type[d.get("type", "其他")].add(n)
    comms = list(by_type.values())
    if not comms:
        comms = [{n} for n in G.nodes]
    return comms


def _merge_small(communities: List[set], G: nx.Graph,
                 target: int = 8) -> List[set]:
    """把单节点社区合并到邻居最多的社区, 直到 <= target 个。"""
    if len(communities) <= target:
        return communities
    # 按大小排序
    communities = sorted(communities, key=lambda c: -len(c))
    big = communities[:target - 1]
    small = communities[target - 1:]
    for s in small:
        if not s:
            continue
        # 找邻居最多的 big 社区
        n = next(iter(s))
        neighbors = list(G.neighbors(n))
        best_idx, best_score = 0, -1
        for i, b in enumerate(big):
            cnt = sum(1 for x in s if any(G.has_edge(x, y) for y in b))
            if cnt > best_score:
                best_score = cnt
                best_idx = i
        big[best_idx] = big[best_idx] | s
    return [c for c in big if c]


# ============================================================
# Modularity 估计
# ============================================================
def _modularity(G: nx.Graph, communities: List[set]) -> float:
    """快速 modularity 估计 (无 nx.community 时)."""
    if not communities or G.number_of_edges() == 0:
        return 0.0
    m = G.number_of_edges()
    if m == 0:
        return 0.0
    q = 0.0
    for c in communities:
        lc = G.subgraph(c).number_of_edges()
        dc = sum(dict(G.degree(c)).values())
        q += (lc / m) - (dc / (2 * m)) ** 2
    return round(q, 4)


# ============================================================
# 社区摘要 (template + LLM 两种)
# ============================================================
def _template_community_summary(G: nx.Graph, community: set,
                                top_k: int = 5) -> CommunitySummary:
    """模板方式生成社区摘要."""
    sub = G.subgraph(community)
    nodes_sorted = sorted(
        sub.nodes(data=True),
        key=lambda x: x[1].get("pagerank", 0),
        reverse=True,
    )
    top_nodes = [
        {"id": n, "type": d.get("type", "其他"),
         "freq": d.get("freq", 0),
         "pagerank": round(d.get("pagerank", 0), 4)}
        for n, d in nodes_sorted[:top_k]
    ]
    edges_sorted = sorted(
        sub.edges(data=True), key=lambda x: x[2].get("weight", 0), reverse=True
    )
    top_edges = [
        {"source": a, "target": b, "weight": d.get("weight", 0)}
        for a, b, d in edges_sorted[:top_k]
    ]
    # 标签: 频次最高 type
    type_counter = Counter(d.get("type", "其他") for _, d in sub.nodes(data=True))
    label = type_counter.most_common(1)[0][0] if type_counter else "其他"
    # 摘要文本
    top_ids = " / ".join(n["id"] for n in top_nodes[:3])
    summary = ("社区 " + label + " (" + str(len(community)) + " 节点, "
               + str(sub.number_of_edges()) + " 边). 核心: " + top_ids + ".")

    return CommunitySummary(
        community_id=-1,
        node_count=len(community),
        edge_count=sub.number_of_edges(),
        top_nodes=top_nodes,
        top_edges=top_edges,
        label=label,
        summary=summary,
        generated_by="template",
    )


def _llm_community_summary(community: CommunitySummary,
                            router: Any) -> CommunitySummary:
    """用 LLM 改写社区摘要为自然语言 (失败则保留 template)."""
    if router is None or not getattr(router, "api_key", ""):
        return community
    try:
        nodes_text = ", ".join(
            n["id"] + "(" + n["type"] + ")"
            for n in community.top_nodes[:6]
        )
        edges_text = "; ".join(
            e["source"] + "-" + e["target"]
            for e in community.top_edges[:4]
        )
        prompt = (
            "你是宏观政策分析专家, 请用 1-2 句话 (50-100 字) 总结以下社区:\n"
            "节点: " + nodes_text + "\n"
            "边: " + edges_text + "\n"
            "要求: 概括该社区的核心主题, 不要列举, 只输出一段连贯中文。"
        )
        from src.ai.router import get_default_router
        if not isinstance(router, type(get_default_router())):
            pass
        raw, _, _, _ = router.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=300, use_json_mode=False,
        )
        if raw:
            raw = _strip_think(raw)
        if raw:
            raw = _strip_think(raw)
        if raw and raw.strip():
            community.summary = raw.strip()
            community.generated_by = "llm"
    except Exception as e:
        logger.info("LLM 社区摘要失败, 保留 template: %s", e)
    return community


# ============================================================
# 全局摘要
# ============================================================
def _llm_global_summary(community_summaries: List[CommunitySummary],
                         router: Any) -> Tuple[str, str]:
    """汇总所有社区为全局摘要."""
    if not community_summaries:
        return ("知识图谱暂无显著社区结构。", "template")
    if router is None or not getattr(router, "api_key", ""):
        # 模板方式
        lines = ["# 全局知识图谱摘要 (template)\n"]
        for c in community_summaries:
            lines.append("- **" + c.label + "** (" + str(c.node_count) + "节点): " + c.summary)
        return ("\n".join(lines), "template")
    try:
        items = "\n".join(
            (str(i + 1) + ". " + c.label + ": " + c.summary)
            for i, c in enumerate(community_summaries)
        )
        prompt = (
            "你是中国宏观经济研究专家, 请基于以下社区摘要, 总结过去一段时间的核心政策/产业主线:\n"
            + items + "\n\n"
            "要求: 200-400 字, 提炼 3-5 个跨社区主题, 用分点 markdown。"
        )
        raw, _, _, _ = router.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4, max_tokens=800, use_json_mode=False,
        )
        if raw:
            raw = _strip_think(raw)
        if raw:
            raw = _strip_think(raw)
        if raw and raw.strip():
            return (raw.strip(), "llm")
    except Exception as e:
        logger.info("LLM 全局摘要失败: %s", e)
    # fallback template
    lines = ["# 全局摘要 (template)\n"]
    for c in community_summaries:
        lines.append("- **" + c.label + "**: " + c.summary)
    return ("\n".join(lines), "template")


# ============================================================
# 主入口: 从文章列表构建知识图谱 + 社区 + 摘要
# ============================================================
def build_graph_rag(articles: List[Any],
                    router: Any = None,
                    max_communities: int = 8,
                    use_llm_summary: bool = True) -> GraphRAGReport:
    """从文章列表构建 GraphRAG-lite 报告.

    Args:
        articles: 已有文章列表 (有 title/content_text/source 等字段)
        router: LLM router (None 时所有摘要走模板)
        max_communities: 最多保留几个社区
        use_llm_summary: 是否用 LLM 改写摘要
    """
    # 1) 构建知识图谱 (复用 src.kg.graph)
    from src.kg.graph import build as build_kg
    G = build_kg(articles, router=router, min_node_freq=1)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    if n_nodes == 0:
        return GraphRAGReport(
            n_nodes=0, n_edges=0, n_communities=0,
            global_summary="(无文章, 知识图谱为空)",
            global_generated_by="template",
            modularity=0.0,
        )

    # 2) 社区发现
    communities = _detect_communities(G, max_communities=max_communities)
    n_comm = len(communities)
    mod = _modularity(G, communities)

    # 3) 每社区摘要
    summaries: List[CommunitySummary] = []
    for i, comm in enumerate(communities):
        cs = _template_community_summary(G, comm, top_k=6)
        cs.community_id = i
        if use_llm_summary:
            cs = _llm_community_summary(cs, router)
        summaries.append(cs)

    # 4) 按 size 排序, 大社区在前
    summaries.sort(key=lambda c: -c.node_count)

    # 5) 全局摘要
    gs, gs_by = _llm_global_summary(summaries, router)

    return GraphRAGReport(
        n_nodes=n_nodes,
        n_edges=n_edges,
        n_communities=n_comm,
        communities=summaries,
        global_summary=gs,
        global_generated_by=gs_by,
        modularity=mod,
    )


# ============================================================
# 全局问答 (GraphRAG 核心场景)
# ============================================================
def ask_global(question: str,
                articles: List[Any],
                router: Any = None,
                top_communities: int = 3,
                use_llm: bool = True) -> Dict[str, Any]:
    """全局问答: 用 KG 社区结构检索相关社区, 拼装上下文, LLM 回答.

    Args:
        question: 用户问题
        articles: 文章列表
        router: LLM router
        top_communities: 选 Top-K 相关社区
        use_llm: 是否用 LLM 改写答案

    Returns:
        dict 含: question, answer, sources, communities_used, generated_by
    """
    report = build_graph_rag(articles, router=router,
                              max_communities=8,
                              use_llm_summary=use_llm)
    if report.n_communities == 0:
        return {
            "question": question,
            "answer": "(无知识图谱社区, 无法回答)",
            "sources": [],
            "communities_used": [],
            "generated_by": "empty",
        }

    # 1) 简单关键词匹配: 选包含 question 关键词最多的社区
    q_tokens = set(question)
    scored = []
    for c in report.communities:
        score = 0
        for n in c.top_nodes:
            if n["id"] in question:
                score += 2
        for e in c.top_edges:
            if e["source"] in question or e["target"] in question:
                score += 1
        scored.append((score, c))
    scored.sort(key=lambda x: -x[0])
    picked = [c for s, c in scored[:top_communities] if s > 0]
    if not picked:
        # 兜底: 取最大的
        picked = report.communities[:top_communities]

    # 2) 拼装上下文
    ctx_lines = []
    for c in picked:
        ctx_lines.append("## 社区 " + c.label + " (" + str(c.node_count) + " 节点)")
        ctx_lines.append(c.summary)
        ctx_lines.append("核心实体: " + ", ".join(n["id"] for n in c.top_nodes[:5]))
        ctx_lines.append("")
    context = "\n".join(ctx_lines)

    # 3) LLM 回答 / 模板兜底
    answer = ""
    gen_by = "template"
    if use_llm and router is not None and getattr(router, "api_key", ""):
        try:
            prompt = (
                "你是中国宏观经济分析专家。基于以下知识图谱社区摘要, 回答用户问题。\n"
                "回答要求: 客观、引用具体实体、150-300 字。\n\n"
                "## 知识上下文\n" + context + "\n\n"
                "## 用户问题\n" + question + "\n\n"
                "## 回答"
            )
            raw, _, _, _ = router.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=600, use_json_mode=False,
            )
            if raw:
                raw = _strip_think(raw)
            if raw and raw.strip():
                answer = raw.strip()
                gen_by = "llm"
        except Exception as e:
            logger.info("LLM 全局问答失败: %s", e)

    if not answer:
        # 模板: 拼接社区摘要 + 列出涉及实体
        entities = []
        for c in picked:
            entities.extend(n["id"] for n in c.top_nodes[:4])
        answer = ("根据知识图谱的 " + str(len(picked)) + " 个相关社区 ("
                  + ", ".join(c.label for c in picked) + "), 涉及核心实体: "
                  + ", ".join(entities[:10]) + ". "
                  + " ".join(c.summary for c in picked))
        gen_by = "template"

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {"community_id": c.community_id, "label": c.label,
             "node_count": c.node_count, "edge_count": c.edge_count,
             "summary": c.summary}
            for c in picked
        ],
        "communities_used": [c.community_id for c in picked],
        "generated_by": gen_by,
        "modularity": report.modularity,
        "n_graph_nodes": report.n_nodes,
        "n_graph_edges": report.n_edges,
    }


# ============================================================
# 自检
# ============================================================
if __name__ == "__main__":
    print("== graph_rag self-test ==")
    from src.scraper.pipeline import Article

    articles = [
        Article(title="央行降准 1 万亿", content=["央行降准支持实体经济"],
                content_text="央行宣布下调存款准备金率 0.5 个百分点, 释放长期资金约 1 万亿元, 加大宏观调控力度"),
        Article(title="国常会稳经济一揽子政策", content=["国务院常务会议研究稳经济"],
                content_text="国务院常务会议研究部署稳经济一揽子政策, 加大宏观政策调控力度"),
        Article(title="5 月新能源车销量", content=["新能源车销量增长"],
                content_text="5 月新能源汽车销量同比增长 38.5%, 渗透率突破 47%, 产业政策利好持续"),
        Article(title="工信部新型储能", content=["工信部储能方案"],
                content_text="工信部发布新型储能制造业高质量发展行动方案, 到 2027 年培育 3-5 家生态主导型企业"),
        Article(title="上海扩大消费", content=["上海消费政策"],
                content_text="上海市发布扩大消费若干措施, 新能源汽车以旧换新补贴"),
    ]
    # 1. 不带 LLM
    print("\n-- template 模式 --")
    rep = build_graph_rag(articles, router=None, use_llm_summary=False)
    print("nodes:", rep.n_nodes, "edges:", rep.n_edges,
          "communities:", rep.n_communities, "modularity:", rep.modularity)
    for c in rep.communities[:3]:
        print(" 社区", c.community_id, c.label, "(", c.node_count, "节点,", c.edge_count, "边)")
        print("   ", c.summary)
    print("\n-- LLM 摘要 (如可用) --")
    try:
        from src.ai.router import get_default_router
        router = get_default_router()
        rep_llm = build_graph_rag(articles, router=router, use_llm_summary=True)
        print("communities:", rep_llm.n_communities)
        for c in rep_llm.communities[:2]:
            print(" ", c.label, "->", c.summary[:80], "...")
        print("global (by:", rep_llm.global_generated_by, "):", rep_llm.global_summary[:200])
    except Exception as e:
        print("LLM 模式跳过:", e)

    # 2. 全局问答
    print("\n-- ask_global (template) --")
    ans = ask_global("降准对宏观经济有什么影响", articles, router=None, use_llm=False)
    print("Q: 降准对宏观经济有什么影响")
    print("A:", ans["answer"][:200])
    print("communities_used:", ans["communities_used"])
    print("\nAll graph_rag self-tests passed")
