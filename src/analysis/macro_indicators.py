"""analysis.macro_indicators - 宏观指标集成 (CPI/PMI/PPI/GDP + LPR/M2/Shibor)

数据源分层:
  Layer 1: 东方财富 datacenter-web.eastmoney.com (免费, 真实数据)
             - RPT_ECONOMY_CPI  -> CPI 同/环比/累计
             - RPT_ECONOMY_PMI  -> 制造业/非制造业 PMI
             - RPT_ECONOMY_PPI  -> PPI 同/环比/累计
             - RPT_ECONOMY_GDP  -> GDP 累计值 + 同比
  Layer 2: 内置参考表 (LPR/M2/Shibor/10年国债, 标注"最近一期"或"参考值")
  Layer 3: 全部失败返回 None, 不阻塞主流程

核心能力:
- 拉取 + 标准化 + 趋势计算 (近 6 月方向)
- 与当日新闻主题词做关联解读 (例: 报道"通胀" -> CPI 同比)
- 输出 MacroReport 供报告/仪表盘/AI 使用

网络异常/超时 8s 内自动降级, 不影响主流程。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, List, Optional

import requests

from src.utils.date_utils import parse_date
from src.utils.logger import get_logger

logger = get_logger("analysis.macro_indicators")

EAST_BASE = "https://datacenter-web.eastmoney.com/api/data/v1/get"
TIMEOUT = 8

# 内置参考表 (LPR/M2/Shibor/10年国债, 标注"参考值")
# 数据更新于 2025-Q4 公开口径, 真实场景可对接 PBOC/统计局
_FALLBACK_TABLE: Dict[str, Dict[str, str]] = {
    "LPR_1Y":     {"value": "3.10", "mom": "持平", "yoy": "-30bp", "date": "2025-10-20"},
    "LPR_5Y":     {"value": "3.60", "mom": "持平", "yoy": "-25bp", "date": "2025-10-20"},
    "M2_YoY":     {"value": "8.4",  "mom": "+0.2pct", "yoy": "+0.6pct", "date": "2025-09"},
    "SHIBOR_ON":  {"value": "1.52", "mom": "+1bp", "yoy": "-15bp", "date": "2025-11"},
    "SHIBOR_3M":  {"value": "1.68", "mom": "持平", "yoy": "-32bp", "date": "2025-11"},
    "CN10Y_YIELD":{"value": "1.83", "mom": "+2bp", "yoy": "-25bp", "date": "2025-11"},
}

# 主题词 -> 宏观指标映射
_NEWS_THEME_TO_INDICATOR: Dict[str, str] = {
    "通胀": "CPI_YoY", "物价": "CPI_YoY", "cpi": "CPI_YoY",
    "通缩": "CPI_YoY", "ppi": "PPI_YoY", "工业品": "PPI_YoY",
    "制造业": "PMI_MAKE", "景气": "PMI_MAKE", "pmi": "PMI_MAKE",
    "服务业": "PMI_NMAKE", "非制造业": "PMI_NMAKE",
    "gdp": "GDP_YoY", "增长": "GDP_YoY", "经济": "GDP_YoY",
    "降息": "LPR_1Y", "降准": "LPR_1Y", "lpr": "LPR_1Y", "利率": "LPR_1Y",
    "房贷": "LPR_5Y", "信贷": "M2_YoY", "m2": "M2_YoY", "社融": "M2_YoY",
    "流动性": "SHIBOR_ON", "shibor": "SHIBOR_ON", "拆借": "SHIBOR_ON",
    "国债": "CN10Y_YIELD", "债券": "CN10Y_YIELD", "收益率": "CN10Y_YIELD",
}


@dataclass
class MacroIndicator:
    name: str
    code: str
    latest: str
    mom: str
    yoy: str
    trend: str  # ↑ / ↓ / →
    signal: str  # 中文短解读
    source: str  # eastmoney / fallback / demo
    series: List[Dict[str, str]] = field(default_factory=list)  # [{date, value}, ...]

    def as_dict(self):
        return asdict(self)


@dataclass
class NewsLinkage:
    macro_indicator: str
    value: str
    interpretation: str

    def as_dict(self):
        return asdict(self)


@dataclass
class MacroReport:
    date: str
    as_of: str
    source: str
    indicators: List[MacroIndicator]
    news_linkage: List[NewsLinkage] = field(default_factory=list)
    summary: str = ""

    def as_dict(self):
        d = asdict(self)
        d["indicators"] = [i.as_dict() for i in self.indicators]
        d["news_linkage"] = [n.as_dict() for n in self.news_linkage]
        return d


def _get_eastmoney(report_name: str, page_size: int = 6) -> Optional[List[Dict]]:
    """通用东方财富 datacenter 接口。"""
    url = (
        f"{EAST_BASE}?reportName={report_name}&columns=ALL&pageNumber=1"
        f"&pageSize={page_size}&sortColumns=REPORT_DATE&sortTypes=-1"
    )
    try:
        r = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return None
        d = r.json()
        if not d.get("success"):
            return None
        return (d.get("result") or {}).get("data") or []
    except Exception as e:
        logger.debug(f"eastmoney {report_name} fail: {e}")
        return None


def _fmt_trend(series: List[Dict], value_key: str) -> str:
    """根据最近 3 期方向给出 ↑/↓/→。"""
    vals = []
    for r in series[:3]:
        v = r.get(value_key)
        if v is None:
            return "→"
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            return "→"
    if len(vals) < 2:
        return "→"
    if vals[0] > vals[1] + 0.05:
        return "↑"
    if vals[0] < vals[1] - 0.05:
        return "↓"
    return "→"


def _cpi_indicator() -> Optional[MacroIndicator]:
    data = _get_eastmoney("RPT_ECONOMY_CPI", page_size=6)
    if not data:
        return None
    series = []
    for r in data:
        series.append({
            "date": (r.get("REPORT_DATE") or "")[:7],
            "value": r.get("NATIONAL_SAME") or "",
        })
    latest = data[0]
    return MacroIndicator(
        name="CPI (居民消费价格)",
        code="CPI_YoY",
        latest=str(latest.get("NATIONAL_SAME", "—")) + "%",
        mom=(str(latest.get("NATIONAL_SEQUENTIAL", "—")) + "%" if latest.get("NATIONAL_SEQUENTIAL") is not None else "—"),
        yoy=str(latest.get("NATIONAL_SAME", "—")) + "%",
        trend=_fmt_trend(data, "NATIONAL_SAME"),
        signal="温和通胀" if (latest.get("NATIONAL_SAME") or 0) < 3 else "需关注",
        source="eastmoney",
        series=series,
    )


def _pmi_indicator() -> Optional[MacroIndicator]:
    data = _get_eastmoney("RPT_ECONOMY_PMI", page_size=6)
    if not data:
        return None
    series = [{"date": (r.get("REPORT_DATE") or "")[:7], "value": r.get("MAKE_INDEX") or ""} for r in data]
    latest = data[0]
    make = latest.get("MAKE_INDEX")
    return MacroIndicator(
        name="制造业 PMI",
        code="PMI_MAKE",
        latest=str(make) if make is not None else "—",
        mom="—",
        yoy="—",
        trend=_fmt_trend(data, "MAKE_INDEX"),
        signal="扩张" if (make or 0) >= 50 else "收缩",
        source="eastmoney",
        series=series,
    )


def _ppi_indicator() -> Optional[MacroIndicator]:
    data = _get_eastmoney("RPT_ECONOMY_PPI", page_size=6)
    if not data:
        return None
    series = [{"date": (r.get("REPORT_DATE") or "")[:7], "value": r.get("BASE_SAME") or ""} for r in data]
    latest = data[0]
    return MacroIndicator(
        name="PPI (工业品出厂价格)",
        code="PPI_YoY",
        latest=str(latest.get("BASE_SAME", "—")) + "%",
        mom="—",
        yoy=str(latest.get("BASE_SAME", "—")) + "%",
        trend=_fmt_trend(data, "BASE_SAME"),
        signal="工业品价格上行" if (latest.get("BASE_SAME") or 0) > 0 else "工业通缩压力",
        source="eastmoney",
        series=series,
    )


def _gdp_indicator() -> Optional[MacroIndicator]:
    data = _get_eastmoney("RPT_ECONOMY_GDP", page_size=6)
    if not data:
        return None
    series = [{"date": (r.get("REPORT_DATE") or "")[:7], "value": r.get("SUM_SAME") or ""} for r in data]
    latest = data[0]
    return MacroIndicator(
        name="GDP (累计同比)",
        code="GDP_YoY",
        latest=str(latest.get("SUM_SAME", "—")) + "%",
        mom="—",
        yoy=str(latest.get("SUM_SAME", "—")) + "%",
        trend=_fmt_trend(data, "SUM_SAME"),
        signal="增长平稳" if (latest.get("SUM_SAME") or 0) >= 5 else "增长承压",
        source="eastmoney",
        series=series,
    )


def _fallback_indicator(code: str, name: str, signal: str) -> MacroIndicator:
    fb = _FALLBACK_TABLE.get(code, {"value": "—", "mom": "—", "yoy": "—", "date": "—"})
    return MacroIndicator(
        name=name, code=code,
        latest=fb["value"] + ("%" if code not in ("LPR_1Y", "LPR_5Y", "SHIBOR_ON", "SHIBOR_3M", "CN10Y_YIELD") else "%"),
        mom=fb["mom"], yoy=fb["yoy"],
        trend="→", signal=signal,
        source="fallback",
        series=[{"date": fb["date"], "value": fb["value"]}],
    )


def _link_news_to_indicators(theme_keywords, industries, indicators):
    """将当日主题词/产业 关联到宏观指标, 给出解读。"""
    code_to_ind = {i.code: i for i in indicators}
    linked = set()
    out = []
    # 主题词匹配
    if theme_keywords:
        for kw in theme_keywords[:8]:
            word = (kw.get("word") if isinstance(kw, dict) else str(kw)) or ""
            for w in [word, word.lower()]:
                if w in _NEWS_THEME_TO_INDICATOR:
                    code = _NEWS_THEME_TO_INDICATOR[w]
                    if code in code_to_ind and code not in linked:
                        ind = code_to_ind[code]
                        out.append(NewsLinkage(
                            macro_indicator=ind.name,
                            value=ind.latest,
                            interpretation=f"今日新闻涉及『{word}』, 对应 {ind.name} 当前 {ind.latest}, 趋势 {ind.trend}, {ind.signal}",
                        ))
                        linked.add(code)
                        break
    if not out and indicators:
        # 至少给一条解读
        ind = indicators[0]
        out.append(NewsLinkage(
            macro_indicator=ind.name,
            value=ind.latest,
            interpretation=f"宏观背景: {ind.name} 最新 {ind.latest}, 趋势 {ind.trend}, {ind.signal}",
        ))
    return out[:6]


def snapshot(target_date: str, theme_keywords=None, industries=None) -> Optional[MacroReport]:
    """拉取宏观指标快照。

    theme_keywords: [{word, score, explain}, ...] (来自 AI.theme_keywords.keywords)
    industries:     ["新能源", ...]
    """
    indicators: List[MacroIndicator] = []
    real_count = 0
    for fn, code, name, signal in [
        (_cpi_indicator,  "CPI_YoY",       "CPI (居民消费价格)",   "价格端观察"),
        (_pmi_indicator,  "PMI_MAKE",      "制造业 PMI",          "景气端观察"),
        (_ppi_indicator,  "PPI_YoY",       "PPI (工业品价格)",    "工业品价格"),
        (_gdp_indicator,  "GDP_YoY",       "GDP (累计同比)",      "增长端观察"),
    ]:
        try:
            ind = fn()
        except Exception as e:
            logger.debug(f"{code} fail: {e}")
            ind = None
        if ind:
            indicators.append(ind)
            real_count += 1
        else:
            indicators.append(_fallback_indicator(code, name, signal))
    # 利率/货币类: 全部用 fallback
    indicators.append(_fallback_indicator("LPR_1Y",      "LPR (1年期)",     "政策利率"))
    indicators.append(_fallback_indicator("LPR_5Y",      "LPR (5年期)",     "长期信贷"))
    indicators.append(_fallback_indicator("M2_YoY",      "M2 同比",         "货币总量"))
    indicators.append(_fallback_indicator("SHIBOR_ON",   "Shibor 隔夜",     "短端利率"))
    indicators.append(_fallback_indicator("CN10Y_YIELD", "10年国债收益率",  "无风险利率"))
    as_of = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    source = f"东方财富 ({real_count} 实时) + 内置参考 ({len(indicators) - real_count})"
    linkage = _link_news_to_indicators(theme_keywords or [], industries or [], indicators)
    summary = (
        f"宏观快照: 实时 {real_count} 项 (CPI/PMI/PPI/GDP), 内部参考 {len(indicators) - real_count} 项; "
        f"新闻匹配 {len(linkage)} 条。"
    )
    return MacroReport(
        date=target_date, as_of=as_of, source=source,
        indicators=indicators, news_linkage=linkage,
        summary=summary,
    )


if __name__ == "__main__":
    import json
    rep = snapshot("2026-06-12",
                   theme_keywords=[{"word": "通胀"}, {"word": "降息"}, {"word": "新能源"}],
                   industries=["新能源", "金融"])
    print(json.dumps(rep.as_dict(), ensure_ascii=False, indent=2))