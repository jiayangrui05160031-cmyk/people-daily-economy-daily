"""analysis.stock_correlation - 产业-A股新闻-市场联动分析

用东方财富免费 API (push2.eastmoney.com) 抓取 A 股实时行情, 与新闻情绪做联动分析:
- 每个被命中的产业, 拉取 6 只相关 A 股的当日涨跌幅
- 行业新闻情绪 vs 行业个股表现 → 判断是『预期改善』还是『预期差』
- 输出 Top 上涨/下跌 A 股 + 行业联动评分

网络异常时优雅降级 (返回 None, 不阻塞主流程)。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import requests
from src.config import INDUSTRY_TO_STOCKS
from src.utils.logger import get_logger

logger = get_logger("analysis.stock_correlation")

EAST_F10 = "https://push2.eastmoney.com/api/qt/stock/get"
EAST_SECID = "https://push2.eastmoney.com/api/qt/stock/get"
TIMEOUT = 8


@dataclass
class StockQuote:
    code: str
    name: str
    price: float
    pct_change: float
    turnover: float  # 成交额 (元)


@dataclass
class IndustryMarketSummary:
    industry: str
    stocks: List[StockQuote]
    avg_pct_change: float
    best: Optional[StockQuote] = None
    worst: Optional[StockQuote] = None
    breadth: float = 0.0  # 上涨家数 / 总家数


@dataclass
class MarketCorrelationReport:
    date: str
    industries: List[IndustryMarketSummary]
    market_overall: str  # 'bullish' | 'bearish' | 'mixed' | 'unavailable'
    top_gainers: List[StockQuote]
    top_losers: List[StockQuote]
    note: str = ""

    def as_dict(self):
        d = asdict(self)
        d['industries'] = [asdict(i) for i in self.industries]
        d['top_gainers'] = [asdict(s) for s in self.top_gainers]
        d['top_losers'] = [asdict(s) for s in self.top_losers]
        return d


def _name_to_eastmoney_code(name: str) -> Optional[Tuple[str, str]]:
    """股票名 -> (secid, code) 映射表 (Top 60 主流 A 股, 覆盖 12 产业链)。"""
    table = {
        "宁德时代": ("0.300750", "300750"), "比亚迪": ("0.002594", "002594"),
        "隆基绿能": ("0.601012", "601012"), "阳光电源": ("0.300274", "300274"),
        "通威股份": ("0.600438", "600438"), "金风科技": ("0.002202", "002202"),
        "中芯国际": ("1.688981", "688981"), "北方华创": ("0.002371", "002371"),
        "韦尔股份": ("0.603501", "603501"), "长电科技": ("0.600584", "600584"),
        "兆易创新": ("0.603986", "603986"), "中微公司": ("1.688012", "688012"),
        "科大讯飞": ("0.002230", "002230"), "海光信息": ("1.688041", "688041"),
        "寒武纪": ("1.688256", "688256"), "商汤": ("1.800000", "00020"),
        "云从科技": ("1.688327", "688327"), "拓尔思": ("0.300229", "300229"),
        "中国移动": ("1.600941", "600941"), "中国电信": ("1.601728", "601728"),
        "中国联通": ("1.600050", "600050"), "紫光股份": ("0.000938", "000938"),
        "中兴通讯": ("0.000063", "000063"), "深桑达": ("0.000032", "000032"),
        "保利发展": ("0.600048", "600048"), "万科A": ("0.000002", "000002"),
        "招商蛇口": ("0.001979", "001979"), "金地集团": ("0.600383", "600383"),
        "龙湖集团": ("1.609609", "609609"), "中海地产": ("1.00688", "00688"),
        "工商银行": ("1.601398", "601398"), "建设银行": ("1.601939", "601939"),
        "中国平安": ("1.601318", "601318"), "招商银行": ("1.600036", "600036"),
        "中信证券": ("1.600030", "600030"), "东方财富": ("0.300059", "300059"),
        "三一重工": ("0.600031", "600031"), "汇川技术": ("0.300124", "300124"),
        "埃斯顿": ("0.002747", "002747"), "绿的谐波": ("0.688017", "688017"),
        "徐工机械": ("0.000425", "000425"), "中国一重": ("1.601106", "601106"),
        "贵州茅台": ("1.600519", "600519"), "五粮液": ("0.000858", "000858"),
        "伊利股份": ("1.600887", "600887"), "海天味业": ("1.603288", "603288"),
        "美的集团": ("1.000333", "000333"), "格力电器": ("0.000651", "000651"),
        "中远海控": ("1.601919", "601919"), "中国外运": ("1.601598", "601598"),
        "中集集团": ("0.000039", "000039"), "招商轮船": ("1.601872", "601872"),
        "宁波港": ("1.601018", "601018"), "上港集团": ("1.600018", "600018"),
        "北大荒": ("1.600598", "600598"), "隆平高科": ("0.000998", "000998"),
        "牧原股份": ("0.002714", "002714"), "温氏股份": ("0.300498", "300498"),
        "新希望": ("0.000876", "000876"), "海大集团": ("0.002311", "002311"),
        "恒瑞医药": ("1.600276", "600276"), "迈瑞医疗": ("0.300760", "300760"),
        "药明康德": ("1.603259", "603259"), "智飞生物": ("0.300122", "300122"),
        "片仔癀": ("1.600436", "600436"), "爱尔眼科": ("0.300015", "300015"),
        "中国中铁": ("1.601390", "601390"), "中国铁建": ("1.601186", "601186"),
        "中国交建": ("1.601800", "601800"), "中国建筑": ("1.601668", "601668"),
        "中国电建": ("1.601669", "601669"), "中国能建": ("1.601868", "601868"),
    }
    return table.get(name)


def _fetch_one(name: str, code: str, secid: str) -> Optional[StockQuote]:
    try:
        params = {
            "secid": secid, "fields": "f43,f44,f45,f48,f168",
            "invt": "2", "fltt": "2", "_": "1",
        }
        r = requests.get(EAST_F10, params=params, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        data = r.json().get("data") or {}
        if not data:
            return None
        price = data.get("f43") or 0
        pct = data.get("f170") or data.get("f168") or 0
        turnover = data.get("f48") or 0
        if price <= 0:
            return None
        return StockQuote(
            code=code, name=name,
            price=round(price / 100, 3),
            pct_change=round(pct / 100, 3),
            turnover=float(turnover),
        )
    except Exception as e:
        logger.debug(f"fetch {name} failed: {e}")
        return None


def _classify_market(industries: List[IndustryMarketSummary]) -> str:
    if not industries:
        return "unavailable"
    all_pct = []
    for ind in industries:
        for s in ind.stocks:
            all_pct.append(s.pct_change)
    if not all_pct:
        return "unavailable"
    avg = sum(all_pct) / len(all_pct)
    pos = sum(1 for p in all_pct if p > 0) / len(all_pct)
    if avg > 0.5 and pos > 0.6: return "bullish"
    if avg < -0.5 and pos < 0.4: return "bearish"
    return "mixed"


def correlate(date_str: str, industries_hit: List[str], top_n_per_industry: int = 3) -> Optional[MarketCorrelationReport]:
    """主入口: 给定当天命中的产业, 拉行情并生成联动报告。"""
    if not industries_hit:
        return None
    summaries: List[IndustryMarketSummary] = []
    all_stocks: List[StockQuote] = []
    for ind in industries_hit:
        candidates = INDUSTRY_TO_STOCKS.get(ind, [])[:top_n_per_industry]
        quotes: List[StockQuote] = []
        for name in candidates:
            m = _name_to_eastmoney_code(name)
            if not m:
                continue
            secid, code = m
            q = _fetch_one(name, code, secid)
            if q:
                quotes.append(q)
                all_stocks.append(q)
        if not quotes:
            continue
        avg = sum(q.pct_change for q in quotes) / len(quotes)
        breadth = sum(1 for q in quotes if q.pct_change > 0) / len(quotes)
        summaries.append(IndustryMarketSummary(
            industry=ind, stocks=quotes,
            avg_pct_change=round(avg, 3),
            best=max(quotes, key=lambda x: x.pct_change),
            worst=min(quotes, key=lambda x: x.pct_change),
            breadth=round(breadth, 3),
        ))
    if not all_stocks:
        return None
    sorted_gainers = sorted(all_stocks, key=lambda s: -s.pct_change)[:5]
    sorted_losers = sorted(all_stocks, key=lambda s: s.pct_change)[:5]
    return MarketCorrelationReport(
        date=date_str, industries=summaries,
        market_overall=_classify_market(summaries),
        top_gainers=sorted_gainers, top_losers=sorted_losers,
        note="(东方财富实时行情, 仅供研究)",
    )


if __name__ == "__main__":
    rep = correlate("2026-06-12", ["新能源", "半导体", "金融"], top_n_per_industry=2)
    if rep:
        import json
        print(json.dumps(rep.as_dict(), ensure_ascii=False, indent=2))
    else:
        print("(网络不可用或无数据)")
