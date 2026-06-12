"""main.py — 一键 CLI 入口 ==============================================
串联:爬取(多源) → NLP → AI → 报告生成 → JSON 归档。

用法:
    python main.py                          # 全流程,今天日期
    python main.py --yesterday              # 抓昨天(兼容旧语义)
    python main.py --date 2026-06-12        # 指定日期
    python main.py --days 7                 # 自定义回溯天数(默认 2)
    python main.py --skip-scrape            # 跳过爬取(用已有 JSON)
    python main.py --skip-ai                # 跳过 AI(降级为纯 NLP 报告)
    python main.py --debug                  # 调试日志
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

# 把项目根加入 sys.path(便于 python main.py 直接运行)
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai.analyzer import analyze_all
from src.nlp.stats import analyze as nlp_analyze
from src.report.archiver import archive_raw, load_raw
from src.report.renderer import render
from src.scraper.pipeline import Article
from src.scraper.multi_source import fetch_multi_source_articles
from src.utils.date_utils import resolve_target_date, yesterday
from src.utils.logger import get_logger

logger = get_logger("main")


# ============================================================
# CLI 装饰
# ============================================================
def banner(text: str, width: int = 60) -> None:
    print()
    print("=" * width)
    print(f"  {text}")
    print("=" * width)


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def warn(msg: str) -> None:
    print(f"[!] {msg}")


# ============================================================
# 4 个 step
# ============================================================
def step1_scrape(target_date: str, lookback_days: int = 2) -> List[Article]:
    """多源爬取(人民网 + 中国经济网)。"""
    banner(f"STEP 1/4 — 多源爬取 {target_date} (向前 {lookback_days} 天)")
    t0 = time.time()
    articles = fetch_multi_source_articles(
        target_date=target_date, lookback_days=lookback_days,
    )
    # 统计来源
    src_count = {"people": 0, "ce": 0, "?": 0}
    for a in articles:
        if "people.com.cn" in a.url:
            src_count["people"] += 1
        elif "ce.cn" in a.url:
            src_count["ce"] += 1
        else:
            src_count["?"] += 1
    src_info = ", ".join(f"{k}={v}" for k, v in src_count.items() if v > 0)
    ok(f"抓取完成: {len(articles)} 篇 ({src_info}),用时 {time.time() - t0:.1f}s")
    return articles


def step2_nlp(articles: List[Article]):
    """NLP 分析。"""
    banner("STEP 2/4 — NLP 分析(分词 / 关键词 / 词频)")
    t0 = time.time()
    stats = nlp_analyze(articles)
    ok(f"NLP 完成: {stats.total_words} 词, {len(stats.keywords)} 关键词, "
       f"{len(stats.industry_hits)} 产业命中,用时 {time.time() - t0:.1f}s")
    if stats.keywords:
        print("    Top 5 关键词:", ", ".join(w for w, _ in stats.keywords[:5]))
    if stats.industry_hits:
        print("    命中产业:", ", ".join(stats.industry_hits.keys()))
    return stats


def step3_ai(articles: List[Article], nlp_stats):
    """AI 分析(失败时降级为纯 NLP)。"""
    banner("STEP 3/4 — AI 分析(6 个任务)")
    t0 = time.time()
    result = analyze_all(articles, nlp_stats=nlp_stats)
    ok(f"AI 完成: 主题词 {len(result.theme_keywords)}, 政策 {len(result.policies)}, "
       f"产业 {len(result.industries)}, 判断 {len(result.outlooks)}, "
       f"用时 {time.time() - t0:.1f}s")
    if result.policy_direction:
        print(f"    政策风向: {result.policy_direction.get('direction', '?')}")
    return result


def step4_report(articles: List[Article], nlp_stats, ai_result, target_date: str) -> Path:
    """生成 Markdown 报告 + JSON 归档。"""
    banner("STEP 4/4 — 报告生成")
    t0 = time.time()

    # 1. JSON 归档
    archive_path = archive_raw(articles, target_date)
    ok(f"JSON 归档: {archive_path}")

    # 2. Markdown 报告
    report_path = render(articles, nlp_stats, ai_result, target_date)
    ok(f"报告生成: {report_path}")
    ok(f"全部完成,用时 {time.time() - t0:.1f}s")

    return report_path


# ============================================================
# 主流程
# ============================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="每日《人民日报》+《中国经济网》经济新闻热点报告生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--date", type=str, default="",
                        help="报告日期 YYYY-MM-DD,默认今天(推荐日报场景)")
    parser.add_argument("--days", type=int, default=2,
                        help="向前回溯天数(含当天),默认 2 天")
    parser.add_argument("--yesterday", action="store_true",
                        help="抓『昨天』而非『今天』(兼容旧语义)")
    parser.add_argument("--skip-scrape", action="store_true",
                        help="跳过爬取,使用 data/raw/{date}.json")
    parser.add_argument("--skip-ai", action="store_true",
                        help="跳过 AI,降级为纯 NLP 报告")
    parser.add_argument("--debug", action="store_true", help="调试日志")
    args = parser.parse_args()

    if args.debug:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    # 计算目标日期: --date > --yesterday > 今天
    if args.date:
        target_date = resolve_target_date(args.date).isoformat()
    elif args.yesterday:
        target_date = yesterday().isoformat()
    else:
        target_date = resolve_target_date("").isoformat()  # 默认今天
    banner(f"📰 人民日报 + 中国经济网 · 经济新闻日报 · {target_date}")

    # --- 1. 爬取或加载 ---
    if args.skip_scrape:
        warn(f"--skip-scrape 已启用,从归档加载 {target_date}")
        try:
            articles = load_raw(target_date)
            ok(f"加载完成: {len(articles)} 篇")
        except FileNotFoundError as e:
            print(f"[X] {e}")
            return 1
    else:
        articles = step1_scrape(target_date, lookback_days=args.days)
        if not articles:
            warn("未抓取到任何文章(可能是节假日或网络问题),退出")
            return 0

    # --- 2. NLP ---
    nlp_stats = step2_nlp(articles)

    # --- 3. AI ---
    if args.skip_ai:
        warn("--skip-ai 已启用,跳过 AI 分析")
        from src.ai.analyzer import AnalysisResult
        ai_result = AnalysisResult()
    else:
        ai_result = step3_ai(articles, nlp_stats)

    # --- 4. 报告 ---
    report_path = step4_report(articles, nlp_stats, ai_result, target_date)

    # 总结
    banner("✅ 全部完成")
    print(f"  📅 日期: {target_date}")
    print(f"  📰 文章数: {len(articles)}")
    print(f"  📝 报告路径: {report_path}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())