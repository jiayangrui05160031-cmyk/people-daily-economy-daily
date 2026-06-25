"""scraper.ai_extractor - LLM 驱动的智能内容提取 (v6 前沿升级)

传统 article_parser.py 用 BeautifulSoup + CSS 选择器, 脆弱:
  - 网站改版即失效
  - 不同模板需要维护 N 套选择器
  - 噪音段落 (分享 / 推荐 / 广告) 需要专门剔除

本模块用 LLM 替代这套手工规则:
  - 把 (HTML 截断 + URL + 提示) 喂给 LLM
  - LLM 一次性返回: 标题 / 正文 / 时间 / 来源 / 关键词 / 主题 / 情感
  - 降级策略: LLM 不可用时, 回到 BeautifulSoup (article_parser)

优势:
  + 抗网站改版 (LLM 关注语义不关注 DOM 结构)
  + 跨源统一 (人民网/经济网/新华网/财新 一套 prompt)
  + 一并产出 主题/情感/关键词 (省一次 LLM 调用)
  + 失败可降级, 永远不空

调用示例:
    from src.scraper.ai_extractor import AIExtractor
    ext = AIExtractor(router=router)
    result = ext.extract(html=raw_html, url="https://...")
    print(result.title, result.publish_time, result.content[:200])
    print(result.keywords, result.theme, result.sentiment)
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger("scraper.ai_extractor")

# ============================================================
# 数据类
# ============================================================
@dataclass
class ExtractedArticle:
    """LLM 提取结果."""
    title: str = ""
    content: str = ""
    summary: str = ""
    publish_time: str = ""
    source: str = ""
    reporter: str = ""
    keywords: List[str] = field(default_factory=list)
    theme: str = ""            # 主题分类 (政策/产业/市场/...)
    sentiment: str = "中性"     # 利好/中性/利空
    sentiment_score: float = 0.0
    industries: List[str] = field(default_factory=list)  # 涉及行业
    confidence: float = 0.0
    used_llm: bool = False
    fallback_reason: str = ""
    latency_ms: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# HTML 预处理: 移除噪音, 截断, 留关键骨架
# ============================================================
_NOISE_TAG_RE = re.compile(
    r"<(script|style|iframe|noscript|svg|form|button|input|select|textarea)[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
_NOISE_CLASS_RE = re.compile(
    r"<(div|section|aside|footer|header|nav)[^>]*class=[\"'][^\"']*(?:share|comment|recommend|ad|advert|banner|nav|footer|header|sidebar|popup|modal)[^\"']*[\"'][^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _clean_html(html: str, max_chars: int = 8000) -> str:
    """粗清洗: 去 script/style/分享/广告, 去标签, 留文本. 截断到 max_chars."""
    if not html:
        return ""
    s = _NOISE_TAG_RE.sub("", html)
    s = _NOISE_CLASS_RE.sub("", s)
    # 简单去标签
    s = _TAG_RE.sub(" ", s)
    # 去多余空白
    s = _WS_RE.sub(" ", s).strip()
    if len(s) > max_chars:
        s = s[:max_chars] + "..."
    return s


# ============================================================
# Prompt
# ============================================================
_EXTRACT_PROMPT = """你是一位资深中文财经编辑, 需要从一篇网页的纯文本中提取结构化字段。

请仔细阅读以下文本, 输出 JSON:
{{
  "title": "文章标题 (无标题则空字符串)",
  "content": "正文 (去除所有无关内容, 保留核心段落, 200-1500 字)",
  "summary": "100 字以内的摘要",
  "publish_time": "发布时间 (YYYY-MM-DD HH:MM 格式, 无法识别则空)",
  "source": "媒体来源 (如: 人民网, 经济日报, 新华网, 财新)",
  "reporter": "记者姓名 (无则空)",
  "keywords": ["5-8 个核心关键词"],
  "theme": "主题分类 (政策/产业/市场/国际/民生/金融/科技/能源/房地产/其他)",
  "sentiment": "利好/中性/利空",
  "sentiment_score": 0.0 到 1.0 的情感强度 (利好取正, 利空取负, 范围 -1.0~+1.0),
  "industries": ["涉及的行业 (新能源/半导体/金融/消费/...)"]
}}

网页文本 (URL: {url}):
\"\"\"
{text}
\"\"\"

只输出 JSON, 不输出其他内容。"""


# ============================================================
# 主类
# ============================================================
class AIExtractor:
    """LLM 驱动的文章提取器.

    用法:
        ext = AIExtractor(router)
        result = ext.extract(html, url)
    """

    def __init__(self, router=None, max_chars: int = 8000,
                 use_cache: bool = True):
        self.router = router
        self.max_chars = max_chars
        self.use_cache = use_cache
        self._cache: Dict[str, ExtractedArticle] = {}
        # 失败计数
        self._fail_count = 0
        self._success_count = 0

    def _try_get_router(self):
        if self.router is not None:
            return self.router
        try:
            from src.ai.router import get_default_router
            return get_default_router()
        except Exception as e:
            logger.debug(f"无法获取 router: {e}")
            return None

    def extract(self, html: str, url: str = "",
                hint_title: str = "") -> ExtractedArticle:
        """从 HTML 提取结构化字段."""
        t0 = time.time()
        if not html:
            return ExtractedArticle(fallback_reason="empty_html")

        cache_key = f"{len(html)}|{url[:100]}"
        if self.use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        text = _clean_html(html, self.max_chars)
        if len(text) < 30:
            return ExtractedArticle(
                fallback_reason="html_too_short",
                latency_ms=round((time.time() - t0) * 1000, 1),
            )

        router = self._try_get_router()
        if router is None:
            return self._fallback_extract(text, url, t0,
                reason="no_router")

        prompt = _EXTRACT_PROMPT.format(url=url or "(unknown)", text=text)
        try:
            raw, model, ptok, ctok = router.chat(
                messages=[{"role": "user", "content": prompt}],
                model=None,
                temperature=0.1,
                max_tokens=2000,
                use_json_mode=True,
            )
            d = self._parse_json(raw)
            if not d:
                return self._fallback_extract(text, url, t0,
                    reason="parse_fail")
            result = ExtractedArticle(
                title=str(d.get("title", "")).strip()[:200] or hint_title,
                content=str(d.get("content", "")).strip()[:5000],
                summary=str(d.get("summary", "")).strip()[:300],
                publish_time=str(d.get("publish_time", "")).strip()[:30],
                source=str(d.get("source", "")).strip()[:50],
                reporter=str(d.get("reporter", "")).strip()[:20],
                keywords=[str(k)[:30] for k in (d.get("keywords") or [])][:8],
                theme=str(d.get("theme", "")).strip()[:20] or "其他",
                sentiment=str(d.get("sentiment", "中性")).strip()[:5] or "中性",
                sentiment_score=float(d.get("sentiment_score", 0.0) or 0.0),
                industries=[str(i)[:15] for i in (d.get("industries") or [])][:8],
                confidence=0.85,
                used_llm=True,
                latency_ms=round((time.time() - t0) * 1000, 1),
            )
            self._success_count += 1
            if self.use_cache:
                self._cache[cache_key] = result
            logger.debug(
                f"AI 提取成功: {result.title[:30]} ({result.latency_ms}ms, "
                f"主题={result.theme}, 情感={result.sentiment})"
            )
            return result
        except Exception as e:
            self._fail_count += 1
            logger.debug(f"AI 提取失败 ({e}), 降级到 BeautifulSoup")
            return self._fallback_extract(text, url, t0,
                reason=f"llm_error:{str(e)[:50]}")

    def _parse_json(self, raw: str) -> Dict[str, Any]:
        """3 层容错: 原生 JSON -> 代码块 -> 大括号切片."""
        s = (raw or "").strip()
        # 去掉 <think>
        s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()
        # 代码块
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        # 大括号切片
        start = s.find("{")
        end = s.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(s[start:end + 1])
            except Exception:
                pass
        return {}

    def _fallback_extract(self, text: str, url: str, t0: float,
                           reason: str) -> ExtractedArticle:
        """降级: 用启发式提取 (无 LLM)."""
        # 简单启发式
        lines = [l.strip() for l in text.split("。") if len(l.strip()) > 10]
        content = "。".join(lines[:15])[:2000]
        # 时间
        m = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", text)
        if m:
            t = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
        else:
            m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", text)
            t = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}" if m else ""
        # 来源
        m = re.search(r"来源[：: ]+\s*([\u4e00-\u9fa5A-Za-z]{2,20})", text)
        src = m.group(1) if m else ""
        return ExtractedArticle(
            title=lines[0][:50] if lines else "",
            content=content,
            publish_time=t,
            source=src,
            theme="其他",
            sentiment="中性",
            confidence=0.3,
            used_llm=False,
            fallback_reason=reason,
            latency_ms=round((time.time() - t0) * 1000, 1),
        )

    def stats(self) -> Dict[str, Any]:
        return {
            "success": self._success_count,
            "fail": self._fail_count,
            "cache_size": len(self._cache),
        }


# ============================================================
# 自检
# ============================================================
def _self_test() -> None:
    sample = """
    <html>
    <head><title>央行宣布降准 0.5 个百分点 释放长期资金 1 万亿元</title></head>
    <body>
    <div id="content">
        <h1>央行宣布降准 0.5 个百分点 释放长期资金 1 万亿元</h1>
        <div class="meta">2026年06月12日 16:30  来源：人民日报  记者：王某某</div>
        <p>中国人民银行决定于 2026 年 6 月 15 日下调金融机构存款准备金率 0.5 个百分点,
        本次降准共计释放长期资金约 1 万亿元, 这是今年首次降准, 体现了政策面
        对实体经济的支持力度。</p>
        <p>业内人士分析, 此次降准有助于降低银行资金成本, 加大对小微企业、绿色发展等
        重点领域的金融支持, 对房地产市场和资本市场也形成利好。</p>
        <div class="share">分享到微博 分享到微信</div>
    </div>
    </body>
    </html>
    """
    ext = AIExtractor()
    r = ext.extract(sample, url="https://example.com/news/123")
    print(f"标题: {r.title}")
    print(f"时间: {r.publish_time}")
    print(f"来源: {r.source}")
    print(f"主题: {r.theme} | 情感: {r.sentiment} ({r.sentiment_score:+.2f})")
    print(f"关键词: {r.keywords}")
    print(f"行业: {r.industries}")
    print(f"用时: {r.latency_ms}ms | 降级原因: {r.fallback_reason}")


if __name__ == "__main__":
    _self_test()
