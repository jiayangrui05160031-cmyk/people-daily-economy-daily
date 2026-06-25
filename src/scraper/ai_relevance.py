"""scraper.ai_relevance - LLM 驱动的主题相关性评分 (v6 前沿升级)

传统 quality_filter 用规则 (长度 / 关键词密度) 过滤文章, 只能粗筛。
本模块用 LLM 做"是否宏观经济 / 金融 / 政策类新闻"的精准判断:

  输入: 标题 + 摘要 + URL
  输出: relevance_score (0~1) + 是否保留 + 主题分类
  优势: 抗钓鱼站 / 软文 / 不相关转载, 支持细粒度主题过滤

LLM 不可用时降级为 关键词 + 长度 启发式。

调用示例:
    from src.scraper.ai_relevance import AIRelevanceFilter
    f = AIRelevanceFilter(router=router)
    keep, score, theme = f.filter(title, summary, url)
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger("scraper.ai_relevance")

# ============================================================
# 关键词词典 (LLM 降级兜底)
# ============================================================
_RELEVANT_KEYWORDS = {
    "宏观": ["GDP", "CPI", "PMI", "PPI", "通胀", "通缩", "增长", "经济", "GDP"],
    "政策": ["央行", "降准", "降息", "加息", "财政", "货币", "发改委", "国务院", "稳增长"],
    "金融": ["银行", "证券", "保险", "股市", "债市", "汇率", "人民币", "美元", "外汇"],
    "产业": ["新能源", "半导体", "人工智能", "数字经济", "房地产", "汽车", "钢铁",
            "光伏", "锂电", "医药", "生物", "芯片", "5G", "云计算"],
    "市场": ["A 股", "港股", "美股", "上证", "深证", "创业板", "科创板", "北交所"],
    "国际": ["美联储", "鲍威尔", "欧洲央行", "日银", "WTO", "IMF", "外贸", "出口"],
    "民生": ["就业", "工资", "社保", "养老", "教育", "医疗", "消费", "物价", "房价"],
    "风险": ["暴雷", "违约", "坏账", "信用", "评级", "破产", "重组", "退市"],
}

_ALL_KEYWORDS: List[str] = []
for _v in _RELEVANT_KEYWORDS.values():
    _ALL_KEYWORDS.extend(_v)


@dataclass
class RelevanceResult:
    score: float = 0.0          # 0~1, 越高越相关
    is_relevant: bool = False
    themes: List[str] = field(default_factory=list)   # 多标签
    primary_theme: str = ""
    reason: str = ""
    used_llm: bool = False
    latency_ms: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


_PROMPT = """你是一名中文财经新闻过滤专家, 评估一篇文章是否"宏观经济 / 金融政策"类新闻。

文章信息:
- 标题: {title}
- 摘要: {summary}
- URL: {url}

请输出 JSON:
{{
  "score": 0.0~1.0 之间的相关性分数 (0.7 以上建议保留),
  "is_relevant": true/false,
  "themes": ["主题标签 (macro/policy/finance/industry/market/global/livelihood/risk/other)"],
  "primary_theme": "最相关的主题",
  "reason": "一句话理由"
}}

只输出 JSON, 不输出其他内容。"""


class AIRelevanceFilter:
    """LLM 驱动的相关性过滤器."""

    # 阈值: score > 此值认为相关
    THRESHOLD = 0.6

    def __init__(self, router=None, threshold: float = 0.6):
        self.router = router
        self.threshold = threshold
        self._llm_calls = 0
        self._llm_hits = 0
        self._fallback_count = 0

    def _try_get_router(self):
        if self.router is not None:
            return self.router
        try:
            from src.ai.router import get_default_router
            return get_default_router()
        except Exception:
            return None

    def filter(self, title: str, summary: str = "",
               url: str = "") -> RelevanceResult:
        """评估文章相关性."""
        t0 = time.time()
        title = (title or "").strip()
        summary = (summary or "").strip()
        url = (url or "").strip()

        # 1. LLM 评估
        router = self._try_get_router()
        if router is not None and title:
            self._llm_calls += 1
            try:
                prompt = _PROMPT.format(title=title[:100],
                                        summary=summary[:300] or "(无摘要)",
                                        url=url or "(无URL)")
                raw, _, _, _ = router.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1, max_tokens=400, use_json_mode=True,
                )
                d = self._parse_json(raw)
                if d and "score" in d:
                    self._llm_hits += 1
                    score = float(d.get("score", 0.0))
                    return RelevanceResult(
                        score=score,
                        is_relevant=score >= self.threshold,
                        themes=[str(t)[:20] for t in d.get("themes", [])][:5],
                        primary_theme=str(d.get("primary_theme", ""))[:20] or "other",
                        reason=str(d.get("reason", ""))[:100],
                        used_llm=True,
                        latency_ms=round((time.time() - t0) * 1000, 1),
                    )
            except Exception as e:
                logger.debug(f"LLM 评估失败: {e}")

        # 2. 降级: 关键词 + 长度
        self._fallback_count += 1
        score, themes = self._keyword_score(title, summary)
        return RelevanceResult(
            score=score,
            is_relevant=score >= self.threshold,
            themes=themes,
            primary_theme=themes[0] if themes else "other",
            reason="keyword 兜底",
            used_llm=False,
            latency_ms=round((time.time() - t0) * 1000, 1),
        )

    def _keyword_score(self, title: str, summary: str) -> Tuple[float, List[str]]:
        text = title + " " + summary
        if not text.strip():
            return 0.0, []
        hits_per_theme: Dict[str, int] = {}
        total_hits = 0
        for theme, kws in _RELEVANT_KEYWORDS.items():
            cnt = sum(1 for kw in kws if kw in text)
            if cnt > 0:
                hits_per_theme[theme] = cnt
                total_hits += cnt
        if not hits_per_theme:
            return 0.1, []
        # 归一化: 命中数 / 6 截断到 [0.2, 0.95]
        score = min(0.95, 0.3 + 0.1 * total_hits)
        themes = sorted(hits_per_theme.keys(),
                        key=lambda t: -hits_per_theme[t])[:3]
        return score, themes

    def _parse_json(self, raw: str) -> Dict[str, Any]:
        import json
        s = (raw or "").strip()
        s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        start = s.find("{")
        end = s.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(s[start:end + 1])
            except Exception:
                pass
        return {}

    def stats(self) -> Dict[str, Any]:
        return {
            "llm_calls": self._llm_calls,
            "llm_hits": self._llm_hits,
            "fallback": self._fallback_count,
        }


def _self_test() -> None:
    rf = AIRelevanceFilter()
    cases = [
        ("央行宣布降准 0.5 个百分点", "释放长期资金 1 万亿", "https://example.com/1"),
        ("今日菜价小幅上涨 鸡蛋 5 元一斤", "市民买菜", "https://example.com/2"),
        ("美联储鲍威尔暗示 6 月不加息", "美股三大指数收涨", "https://example.com/3"),
    ]
    for t, s, u in cases:
        r = rf.filter(t, s, u)
        print(f"  {t[:30]:30s} -> score={r.score:.2f} theme={r.primary_theme:10s} relevant={r.is_relevant}")


if __name__ == "__main__":
    _self_test()
