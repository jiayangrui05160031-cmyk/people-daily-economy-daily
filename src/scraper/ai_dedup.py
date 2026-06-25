"""scraper.ai_dedup - LLM 驱动的语义去重 (v6 前沿升级)

传统 semantic_dedup.py 用 TF-IDF + 余弦相似度, 速度快但有边界:
  - 短文本 / 同义改写 / 翻译转载 都难以识别
  - 阈值难调, 高则漏, 低则重

本模块用 LLM 做更精准的"是否同一事件"判断:
  输入: 两篇文章的 (标题, 摘要, 来源)
  输出: is_duplicate (bool) + reason
  优势: 抗同义改写 ("降准 0.5%" vs "下调存款准备金率 50bp")
        抗多角度转载 ("央行降准" vs "银行股大涨")

LLM 不可用时降级为 TF-IDF 余弦。

调用示例:
    from src.scraper.ai_dedup import AIDeduper
    d = AIDeduper(router=router)
    is_dup, score, reason = d.is_duplicate(art_a, art_b)
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger("scraper.ai_dedup")


@dataclass
class DupResult:
    is_duplicate: bool = False
    confidence: float = 0.0     # 0~1
    reason: str = ""
    used_llm: bool = False
    latency_ms: float = 0.0


_PROMPT = """你是中文新闻去重专家, 判断两篇文章是否"报道同一事件"。

文章 A:
- 标题: {title_a}
- 摘要: {summary_a}
- 来源: {source_a}

文章 B:
- 标题: {title_b}
- 摘要: {summary_b}
- 来源: {source_b}

请输出 JSON:
{{
  "is_duplicate": true/false  (是否同一事件, 即使转载/翻译/不同角度, 也算 true),
  "confidence": 0.0~1.0  (判断置信度),
  "reason": "一句话理由"
}}

只输出 JSON, 不输出其他内容。"""


class AIDeduper:
    """LLM 驱动的语义去重器."""

    CONFIDENCE_THRESHOLD = 0.7  # 高于此值认为重复

    def __init__(self, router=None, threshold: float = 0.7):
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

    def is_duplicate(self, art_a: Dict[str, Any],
                     art_b: Dict[str, Any]) -> DupResult:
        """判断两篇文章是否同一事件."""
        t0 = time.time()
        router = self._try_get_router()
        if router is not None:
            self._llm_calls += 1
            try:
                prompt = _PROMPT.format(
                    title_a=str(art_a.get("title", ""))[:100],
                    summary_a=str(art_a.get("summary", art_a.get("content", "")))[:200],
                    source_a=str(art_a.get("source", ""))[:30],
                    title_b=str(art_b.get("title", ""))[:100],
                    summary_b=str(art_b.get("summary", art_b.get("content", "")))[:200],
                    source_b=str(art_b.get("source", ""))[:30],
                )
                raw, _, _, _ = router.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.05, max_tokens=200, use_json_mode=True,
                )
                d = self._parse_json(raw)
                if d and "is_duplicate" in d:
                    self._llm_hits += 1
                    is_dup = bool(d.get("is_duplicate", False))
                    conf = float(d.get("confidence", 0.0))
                    return DupResult(
                        is_duplicate=is_dup and conf >= self.threshold,
                        confidence=conf,
                        reason=str(d.get("reason", ""))[:100],
                        used_llm=True,
                        latency_ms=round((time.time() - t0) * 1000, 1),
                    )
            except Exception as e:
                logger.debug(f"LLM 去重失败: {e}")

        # 降级: TF-IDF 简单相似度
        self._fallback_count += 1
        sim = self._text_sim(art_a, art_b)
        is_dup = sim >= 0.7
        return DupResult(
            is_duplicate=is_dup,
            confidence=sim,
            reason=f"tfidf_sim={sim:.2f}",
            used_llm=False,
            latency_ms=round((time.time() - t0) * 1000, 1),
        )

    def _text_sim(self, a: Dict[str, Any], b: Dict[str, Any]) -> float:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            ta = (a.get("title", "") or "") + " " + (a.get("summary", a.get("content", "")) or "")[:200]
            tb = (b.get("title", "") or "") + " " + (b.get("summary", b.get("content", "")) or "")[:200]
            if not ta.strip() or not tb.strip():
                return 0.0
            vec = TfidfVectorizer(max_features=2000)
            X = vec.fit_transform([ta, tb])
            return float(cosine_similarity(X[0], X[1])[0][0])
        except Exception:
            # 字符级 jaccard
            sa = set((a.get("title", "") or "") + (a.get("content", "") or "")[:200])
            sb = set((b.get("title", "") or "") + (b.get("content", "") or "")[:200])
            if not sa or not sb:
                return 0.0
            return len(sa & sb) / len(sa | sb)

    def _parse_json(self, raw: str) -> Dict[str, Any]:
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

    def stats(self) -> Dict[str, int]:
        return {
            "llm_calls": self._llm_calls,
            "llm_hits": self._llm_hits,
            "fallback": self._fallback_count,
        }
