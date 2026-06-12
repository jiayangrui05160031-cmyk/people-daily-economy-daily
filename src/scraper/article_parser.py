"""article_parser — 详情页解析 ==============================================
从文章详情页提取:标题 / 正文段落 / 发布时间 / 来源 / 栏目。

经实爬验证,人民网财经频道当前模板关键节点:
  正文: #rm_txt_zw  (旧模板回退 #ozoom)
  时间: #newstime    (格式: 2026年06月11日20:10)
  来源: #laiyid      (常为空,回退 .channel 文本中的 "来源:" 段)
  标题: 正文上方的首个非空 h1/h2
  栏目: URL 中 c{数字} → 查映射表
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from bs4 import BeautifulSoup, Tag


# 正文容器(按优先级尝试)
CONTENT_SELECTORS = [
    "#rm_txt_zw",
    "#ozoom",
    ".article-content",
    ".article_content",
    ".text",
    "div.content",
    "div.show_text",
    ".TRS_Editor",
]

# 元数据专用选择器
TIME_SELECTORS = ["#newstime", ".article-time", ".pubtime", ".time"]
SOURCE_SELECTORS = ["#laiyid", ".source", ".article-source"]

# 干扰项容器(必须从正文中剔除)
NOISE_SELECTORS = [
    "script", "style", "iframe",
    "#bdshare", ".bdshare", ".share", ".share-wrap", "#btn-wrap",
    ".editor_pic",
    ".relevant", ".related", ".recommend",
    ".ad", ".adv", ".advertisement",
    ".article-edit",
]

# 发布时间正则(多种格式兼容)
PUBLISH_TIME_PATTERNS = [
    r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日\s*(\d{1,2}):(\d{2})",
    r"(\d{4})-(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{2})",
    r"(\d{4})/(\d{1,2})/(\d{1,2})\s+(\d{1,2}):(\d{2})",
]

# 来源 / 记者 / 编辑正则
SOURCE_RE = re.compile(r"来源[：:]\s*([^\s\n\r|]{2,30})")
REPORTER_RE = re.compile(r"记者[：: 　]*([^\s\n\r|]{2,8})")
EDITOR_RE = re.compile(r"编辑[：: 　]*([^\s\n\r|]{1,8})")


def _select_content(soup: BeautifulSoup) -> Optional[Tag]:
    """多组选择器依次尝试,首个命中返回。"""
    for selector in CONTENT_SELECTORS:
        node = soup.select_one(selector)
        if node:
            return node
    return None


def _strip_noise(content: Tag) -> None:
    """原地剔除干扰节点。"""
    for selector in NOISE_SELECTORS:
        for node in content.select(selector):
            try:
                node.decompose()
            except Exception:
                pass


def _extract_title(soup: BeautifulSoup, content: Optional[Tag]) -> str:
    """提取标题(h1 优先,优先取正文上方最近的一个非空标题)。

    真实模板里 h1 通常是页面大标题,但可能与正文标题不一致;
    稳妥做法:取 #rm_txt_zw 上方最近的 h1/h2/h3(排除 nav)。
    """
    if content is not None:
        # 找正文上方最近的标题
        for h in content.find_all_previous(["h1", "h2", "h3"], limit=10):
            text = h.get_text(strip=True)
            if text and len(text) >= 4 and not any(
                kw in text for kw in ["导航", "首页", "栏目", "频道", "登录"]
            ):
                return text

    # 回退
    h1 = soup.select_one("h1")
    if h1:
        t = h1.get_text(strip=True)
        if t and len(t) >= 4:
            return t
    og = soup.select_one('meta[property="og:title"]')
    if og and og.get("content"):
        return og["content"].strip()
    title = soup.select_one("title")
    if title:
        t = title.get_text(strip=True)
        return t.split("_")[0].split("-")[0].strip() if t else ""
    return ""


def _extract_content_paragraphs(content: Tag) -> List[str]:
    """提取正文段落列表(过滤空段、过短段、纯标点)。"""
    paragraphs: List[str] = []
    for p in content.find_all("p"):
        text = p.get_text(separator="", strip=True)
        if not text or len(text) < 4:
            continue
        if not re.search(r"[一-龥A-Za-z0-9]", text):
            continue
        # 跳过明显是版权/分享的段落
        if any(kw in text for kw in ["责编", "版权归", "未经授权", "扫码下载", "关注微信公众号"]):
            continue
        paragraphs.append(text)
    return paragraphs


def _parse_time_string(text: str) -> Optional[str]:
    """尝试从一段文本中解析发布时间,成功返回 YYYY-MM-DD HH:MM。"""
    for pattern in PUBLISH_TIME_PATTERNS:
        m = re.search(pattern, text)
        if m:
            year, month, day, hh, mm = m.groups()
            return f"{year}-{int(month):02d}-{int(day):02d} {int(hh):02d}:{int(mm):02d}"
    return None


def _extract_publish_time(soup: BeautifulSoup, content: Optional[Tag]) -> str:
    """提取发布时间,优先级:#newstime → .channel → 正文容器 → 全文。"""
    # 1) 专用时间节点
    for selector in TIME_SELECTORS:
        node = soup.select_one(selector)
        if node:
            t = _parse_time_string(node.get_text(strip=True))
            if t:
                return t

    # 2) .channel 节点(常含 "2026年06月11日20:10 | 来源:..." 拼接)
    channel_node = soup.select_one(".channel")
    if channel_node:
        t = _parse_time_string(channel_node.get_text(strip=True))
        if t:
            return t

    # 3) 正文容器
    if content is not None:
        t = _parse_time_string(content.get_text(" ", strip=True))
        if t:
            return t

    # 4) 全文兜底
    return _parse_time_string(soup.get_text(" ", strip=True)) or ""


def _extract_source(soup: BeautifulSoup, content: Optional[Tag]) -> str:
    """提取来源,优先级:#laiyid → .channel 文本正则 → 正文文本正则。"""
    # 1) 专用来源节点
    for selector in SOURCE_SELECTORS:
        node = soup.select_one(selector)
        if node:
            text = node.get_text(strip=True)
            if text and len(text) >= 2:
                return text

    # 2) .channel 节点中的 "来源:xxx"
    channel_node = soup.select_one(".channel")
    if channel_node:
        m = SOURCE_RE.search(channel_node.get_text(" ", strip=True))
        if m:
            return m.group(1)

    # 3) 全文正则(限正文区域)
    if content is not None:
        m = SOURCE_RE.search(content.get_text(" ", strip=True))
        if m:
            return m.group(1)

    return ""


def _extract_reporter_editor(content: Optional[Tag], soup: BeautifulSoup) -> Dict[str, str]:
    """从文末提取记者/编辑(常出现在正文之后或 .author 节点中)。"""
    # 优先看 .author
    author_node = soup.select_one(".author")
    candidates = []
    if author_node:
        candidates.append(author_node.get_text(" ", strip=True))
    if content is not None:
        candidates.append(content.get_text(" ", strip=True))
    candidates.append(soup.get_text(" ", strip=True))

    reporter = editor = ""
    for text in candidates:
        if not reporter:
            m = REPORTER_RE.search(text)
            if m:
                reporter = m.group(1)
        if not editor:
            m = EDITOR_RE.search(text)
            if m:
                editor = m.group(1)
        if reporter and editor:
            break

    return {"reporter": reporter, "editor": editor}


def _extract_channel(source_url: str) -> str:
    """从 URL 路径 c{数字} 推断栏目名。"""
    from src.config import CHANNEL_CODE_MAP
    m = re.search(r"/c(\d+)-", source_url)
    if not m:
        return ""
    code = m.group(1)
    return CHANNEL_CODE_MAP.get(code, f"其他(c{code})")


def parse_article_html(html: str, source_url: str = "") -> Dict:
    """解析详情页,返回结构化字段。

    Returns:
        dict with keys: title, content(list[str]), content_text,
                        publish_time, source, reporter, editor,
                        channel, url, word_count
    """
    soup = BeautifulSoup(html, "lxml")
    content = _select_content(soup)

    base = {
        "title": "",
        "content": [],
        "content_text": "",
        "publish_time": "",
        "source": "",
        "reporter": "",
        "editor": "",
        "channel": _extract_channel(source_url),
        "url": source_url,
        "word_count": 0,
    }

    if not content:
        base["title"] = _extract_title(soup, None)
        return base

    _strip_noise(content)
    base["title"] = _extract_title(soup, content)
    base["publish_time"] = _extract_publish_time(soup, content)
    base["source"] = _extract_source(soup, content)
    meta = _extract_reporter_editor(content, soup)
    base["reporter"] = meta["reporter"]
    base["editor"] = meta["editor"]

    paragraphs = _extract_content_paragraphs(content)
    base["content"] = paragraphs
    base["content_text"] = "\n".join(paragraphs)
    base["word_count"] = sum(len(p) for p in paragraphs)

    return base