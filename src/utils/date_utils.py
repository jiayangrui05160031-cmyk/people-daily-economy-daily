"""date_utils — 日期计算工具 ==============================================
支持"今天/昨天"(跨月跨年)、格式化、解析。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional


def parse_date(s: str) -> date:
    """解析 YYYY-MM-DD 字符串为 date 对象。"""
    return datetime.strptime(s, "%Y-%m-%d").date()


def format_date(d: date) -> str:
    """格式化为 YYYY-MM-DD。"""
    return d.strftime("%Y-%m-%d")


def yesterday(today: Optional[date] = None) -> date:
    """计算指定日期的前一天(默认今天)。跨月跨年自动处理。"""
    if today is None:
        today = date.today()
    return today - timedelta(days=1)


def today() -> date:
    """返回今天。"""
    return date.today()


def path_date_token(d: date) -> str:
    """生成 URL 路径中的日期段,如 /n1/2026/0611/。"""
    return f"/n1/{d.year}/{d.strftime('%m%d')}/"


def resolve_target_date(override: str = "") -> date:
    """根据配置或环境变量决定目标日期。

    优先级:override 参数 > 环境变量 REPORT_DATE > 今天(日报场景)。
    """
    if override:
        return parse_date(override)
    from src.config import REPORT_DATE_OVERRIDE
    if REPORT_DATE_OVERRIDE:
        return parse_date(REPORT_DATE_OVERRIDE)
    return date.today()