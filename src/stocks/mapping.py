"""stocks.mapping - 产业-A股概念板块映射"""
from __future__ import annotations

from typing import Dict, List, Tuple

from src.config import INDUSTRY_KEYWORDS, INDUSTRY_TO_STOCKS
from src.utils.logger import get_logger

logger = get_logger("stocks.mapping")


def related_stocks(industry, top_k=6):
    return INDUSTRY_TO_STOCKS.get(industry, [])[:top_k]


def map_industries_to_stocks(industries):
    return {ind: related_stocks(ind) for ind in industries if ind in INDUSTRY_TO_STOCKS}


def list_all_pairs():
    return list(INDUSTRY_TO_STOCKS.items())


def policy_chain(industry):
    return {
        "industry": industry,
        "keywords": INDUSTRY_KEYWORDS.get(industry, []),
        "stocks": related_stocks(industry),
    }


if __name__ == "__main__":
    print("新能源 stocks:", related_stocks("新能源"))
    print("半导体 chain:", policy_chain("半导体"))
    print("mapped industries:", list_all_pairs()[:3])
