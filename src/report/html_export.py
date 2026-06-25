"""src.report.html_export - self-contained HTML export

Convert Markdown report to styled self-contained HTML, open in browser to print/save as PDF.
No external libs (no wkhtmltopdf, no playwright, no reportlab), simple and reliable.

Features:
- Embedded GitHub Dark style CSS
- Dark/light adaptive (prefers-color-scheme)
- Code block highlight (offline minimal)
- Responsive tables
- Print styles @media print
- Anchor TOC sidebar
- 1 file = 1 report, double-click to open
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config import REPORT_DIR
from src.utils.logger import get_logger

logger = get_logger("report.html_export")


_CSS = r"""
:root {
  --bg: #ffffff;
  --fg: #1f2328;
  --border: #d0d7de;
  --code-bg: #f6f8fa;
  --link: #0969da;
  --table-stripe: #f6f8fa;
  --callout: #ddf4ff;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1117;
    --fg: #e6edf3;
    --border: #30363d;
    --code-bg: #161b22;
    --link: #58a6ff;
    --table-stripe: #161b22;
    --callout: #033d80;
  }
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--fg); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; line-height: 1.6; max-width: 980px; margin: 0 auto; padding: 24px 32px; }
h1, h2, h3, h4, h5, h6 { line-height: 1.25; margin-top: 24px; margin-bottom: 16px; font-weight: 600; }
h1 { font-size: 2em; border-bottom: 1px solid var(--border); padding-bottom: 0.3em; }
h2 { font-size: 1.5em; border-bottom: 1px solid var(--border); padding-bottom: 0.3em; }
h3 { font-size: 1.25em; }
h4 { font-size: 1em; }
a { color: var(--link); text-decoration: none; }
a:hover { text-decoration: underline; }
hr { border: 0; border-top: 1px solid var(--border); margin: 24px 0; }
blockquote { border-left: 4px solid var(--border); color: #6e7681; padding: 0 1em; margin: 0; }
code { background: var(--code-bg); padding: 0.2em 0.4em; border-radius: 6px; font-size: 85%; font-family: ui-monospace, "Cascadia Mono", Menlo, monospace; }
pre { background: var(--code-bg); padding: 16px; border-radius: 6px; overflow: auto; font-size: 85%; line-height: 1.45; }
pre code { background: transparent; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 16px 0; }
th, td { border: 1px solid var(--border); padding: 6px 13px; text-align: left; }
th { background: var(--table-stripe); font-weight: 600; }
tbody tr:nth-child(even) { background: var(--table-stripe); }
img { max-width: 100%; box-sizing: content-box; }
ul, ol { padding-left: 2em; }
li + li { margin-top: 4px; }
.toc { position: sticky; top: 16px; float: right; max-width: 280px; background: var(--code-bg); padding: 12px 16px; border-radius: 6px; margin: 16px 0 16px 16px; font-size: 0.9em; max-height: 80vh; overflow: auto; }
.toc ul { list-style: none; padding-left: 0; }
.toc a { color: var(--fg); }
.toc a:hover { color: var(--link); }
@media (max-width: 800px) { .toc { float: none; max-width: 100%; margin: 16px 0; } }
@media print {
  body { max-width: 100%; padding: 8px; }
  h2 { page-break-before: auto; page-break-after: avoid; }
  h3 { page-break-after: avoid; }
  table, pre, blockquote { page-break-inside: avoid; }
  .no-print, .toc { display: none; }
  a { color: var(--fg); text-decoration: none; }
}
"""


def _md_to_html(md):
    lines = md.split("\n")
    out = []
    in_code = False
    in_table = False
    in_list = False
    list_type = None
    in_quote = False
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code:
                out.append("</code></pre>"); in_code = False
            else:
                lang = stripped[3:].strip() or "text"
                out.append('<pre><code class="language-' + lang + '">'); in_code = True
            i += 1; continue
        if in_code:
            out.append(_esc(line)); i += 1; continue
        if re.match(r"^-{3,}$", stripped):
            if in_list: out.append("</" + list_type + ">"); in_list = False; list_type = None
            if in_quote: out.append("</blockquote>"); in_quote = False
            out.append("<hr>"); i += 1; continue
        m = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if m:
            if in_list: out.append("</" + list_type + ">"); in_list = False; list_type = None
            if in_quote: out.append("</blockquote>"); in_quote = False
            level = len(m.group(1))
            text = _inline(m.group(2))
            out.append('<h' + str(level) + ' id="' + _slug(text) + '">' + text + '</h' + str(level) + '>')
            i += 1; continue
        if stripped.startswith(">"):
            if not in_quote: out.append("<blockquote>"); in_quote = True
            out.append(_inline(stripped[1:].strip())); i += 1; continue
        if in_quote: out.append("</blockquote>"); in_quote = False
        if "|" in stripped and re.match(r"^\|", stripped):
            if not in_table:
                out.append("<table>")
                cells = [c.strip() for c in stripped.strip("|").split("|")]
                out.append("<thead><tr>")
                for c in cells: out.append("<th>" + _inline(c) + "</th>")
                out.append("</tr></thead>")
                if i + 1 < len(lines) and re.match(r"^\|[\s\-:|]+\|$", lines[i + 1].strip()):
                    i += 2
                else: i += 1
                out.append("<tbody>"); in_table = True
            else:
                cells = [c.strip() for c in stripped.strip("|").split("|")]
                out.append("<tr>")
                for c in cells: out.append("<td>" + _inline(c) + "</td>")
                out.append("</tr>"); i += 1
            continue
        if in_table: out.append("</tbody></table>"); in_table = False
        m_ul = re.match(r"^[-*+]\s+(.+)$", stripped)
        m_ol = re.match(r"^(\d+)\.\s+(.+)$", stripped)
        if m_ul:
            if not in_list or list_type != "ul":
                if in_list: out.append("</" + list_type + ">")
                out.append("<ul>"); in_list = True; list_type = "ul"
            out.append("<li>" + _inline(m_ul.group(1)) + "</li>"); i += 1; continue
        if m_ol:
            if not in_list or list_type != "ol":
                if in_list: out.append("</" + list_type + ">")
                out.append("<ol>"); in_list = True; list_type = "ol"
            out.append("<li>" + _inline(m_ol.group(2)) + "</li>"); i += 1; continue
        if in_list: out.append("</" + list_type + ">"); in_list = False; list_type = None
        if not stripped: i += 1; continue
        out.append("<p>" + _inline(stripped) + "</p>"); i += 1
    if in_code: out.append("</code></pre>")
    if in_list: out.append("</" + list_type + ">")
    if in_table: out.append("</tbody></table>")
    if in_quote: out.append("</blockquote>")
    return "\n".join(out)


def _inline(text):
    text = _esc(text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", '<a href="\2">\1</a>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", "<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", "<em>\1</em>", text)
    text = re.sub(r"`([^`]+)`", "<code>\1</code>", text)
    return text


def _esc(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _slug(text):
    s = re.sub(r"<[^>]+>", "", text)
    s = re.sub(r"[^\w\u4e00-\u9fff\s-]", "", s).strip()
    s = re.sub(r"\s+", "-", s)
    return s[:80] or "section"


def _build_toc(md):
    items = []
    for m in re.finditer(r"^(#{2,3})\s+(.+)$", md, re.MULTILINE):
        level = len(m.group(1))
        text = _inline(m.group(2).strip())
        slug = _slug(text)
        indent = "  " * (level - 2)
        items.append(indent + '<li><a href="#' + slug + '">' + text + '</a></li>')
    if not items: return ""
    return '<aside class="toc no-print"><strong>目录</strong><ul>' + "".join(items) + '</ul></aside>'


def export(md_path=None, html_path=None, md_text=None):
    if md_text is None:
        if not md_path: return None
        md_path = Path(md_path)
        if not md_path.exists(): return None
        md_text = md_path.read_text(encoding="utf-8")
    if html_path is None:
        if md_path: html_path = str(Path(md_path).with_suffix(".html"))
        else: html_path = str(REPORT_DIR / "report.html")
    body = _md_to_html(md_text)
    toc = _build_toc(md_text)
    full = (
        '<!DOCTYPE html>\n<html lang="zh-CN">\n<head>\n'
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        '<title>宏观经济智能分析报告</title>\n'
        '<style>' + _CSS + '</style>\n'
        '</head>\n<body>\n' + toc + '\n<article>\n' + body + '\n</article>\n</body>\n</html>'
    )
    Path(html_path).write_text(full, encoding="utf-8")
    logger.info("HTML 报告已生成: " + html_path + " (" + str(len(full)) + " 字符)")
    return Path(html_path)


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else None
    dst = sys.argv[2] if len(sys.argv) > 2 else None
    p = export(src, dst)
    print(p)
