"""report — 报告生成层 ==============================================
Jinja2 渲染 Markdown 报告 + JSON 归档。
"""

from .renderer import render, REPORT_DIR
from .archiver import archive_raw

__all__ = ["render", "REPORT_DIR", "archive_raw"]