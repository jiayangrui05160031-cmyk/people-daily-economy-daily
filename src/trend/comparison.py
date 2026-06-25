"""trend.comparison - 时序对比引擎

- day_over_day: 当日 vs 前一交易日
- compare: 主入口,产出量化指标对比 + 关键词/产业差集
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import timedelta
from typing import Dict, List, Optional

from src.storage import repository as repo
from src.utils.date_utils import parse_date, previous_business_day
from src.utils.logger import get_logger

logger = get_logger("trend.comparison")


@dataclass
class MetricDiff:
    metric: str
    today: float
    prev: float
    abs_diff: float
    pct_diff: float
    direction: str
    label: str


@dataclass
class ComparisonResult:
    today_date: str
    prev_date: str
    same_weekday_date: str
    article_count_diff: MetricDiff
    sentiment_diff: Optional[MetricDiff] = None
    policy_diff: Optional[MetricDiff] = None
    entropy_diff: Optional[MetricDiff] = None
    top_share_diff: Optional[MetricDiff] = None
    keyword_overlap_pct: float = 0.0
    new_keywords: List[str] = None
    vanished_keywords: List[str] = None
    industry_new: List[str] = None
    industry_gone: List[str] = None
    note: str = ""

    def as_dict(self):
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, MetricDiff):
                d[k] = asdict(v)
        return d


def _diff(metric, today, prev, label, scale=1.0):
    abs_d = today - prev
    pct = (abs_d / prev * 100.0) if prev not in (0, 0.0) else 0.0
    if abs_d > 0.02 * scale:
        direction = "up"
    elif abs_d < -0.02 * scale:
        direction = "down"
    else:
        direction = "flat"
    return MetricDiff(metric=metric, today=today, prev=prev,
                      abs_diff=round(abs_d, 4), pct_diff=round(pct, 2),
                      direction=direction, label=label)


def _keyword_sets(date_str):
    rows = repo.get_keywords(date_str, theme_only=True)
    if not rows:
        rows = repo.get_keywords(date_str)
    return {r.keyword for r in rows}


def _industry_sets(date_str):
    rows = repo.get_industries(date_str)
    return {r.industry for r in rows}


def compare(today_date, prev_date="", week_date=""):
    today = repo.get_metric(today_date)
    if not today:
        return ComparisonResult(
            today_date=today_date, prev_date=prev_date or "",
            same_weekday_date=week_date or "",
            article_count_diff=MetricDiff("article_count", 0, 0, 0, 0, "flat", "文章数"),
            note="no_today_metric",
        )
    if not prev_date:
        try:
            prev_dt = previous_business_day(parse_date(today_date))
            prev_date = prev_dt.isoformat()
        except Exception:
            prev_date = (parse_date(today_date) - timedelta(days=1)).isoformat()
    prev = repo.get_metric(prev_date)

    today_count = today.article_count or 0
    prev_count = (prev.article_count if prev else 0) or 0

    res = ComparisonResult(
        today_date=today_date, prev_date=prev_date,
        same_weekday_date=week_date or "",
        article_count_diff=_diff("article_count", today_count, prev_count, "文章数", scale=10),
    )

    if prev:
        res.sentiment_diff = _diff("sentiment_index", today.sentiment_index, prev.sentiment_index, "情绪指数", scale=10)
        res.policy_diff = _diff("policy_stance_score", today.policy_stance_score, prev.policy_stance_score, "政策倾向", scale=0.2)
        res.entropy_diff = _diff("attention_entropy", today.attention_entropy, prev.attention_entropy, "注意力熵", scale=0.1)
        res.top_share_diff = _diff("attention_top_share", today.attention_top_share, prev.attention_top_share, "头部集中度", scale=0.1)

    today_kws = _keyword_sets(today_date)
    prev_kws = _keyword_sets(prev_date)
    if prev_kws:
        inter = today_kws & prev_kws
        union = today_kws | prev_kws
        res.keyword_overlap_pct = round(len(inter) / len(union) * 100, 2) if union else 0.0
    res.new_keywords = sorted(today_kws - prev_kws)[:20]
    res.vanished_keywords = sorted(prev_kws - today_kws)[:20]

    today_ind = _industry_sets(today_date)
    prev_ind = _industry_sets(prev_date)
    res.industry_new = sorted(today_ind - prev_ind)
    res.industry_gone = sorted(prev_ind - today_ind)

    if week_date:
        week_metric = repo.get_metric(week_date)
        if week_metric:
            res.note = (f"week-over-week article_count: {today_count} vs {week_metric.article_count}; "
                        f"sentiment: {today.sentiment_index:.1f} vs {week_metric.sentiment_index:.1f}")

    return res


if __name__ == "__main__":
    from src.storage import db
    db.get_conn()
    from datetime import date
    today = date.today().isoformat()
    repo.upsert_metric(repo.DailyMetric(date=today, article_count=30, total_words=5000,
                                        sentiment_index=55, policy_stance_score=0.5,
                                        attention_entropy=0.8, attention_top_share=0.6))
    repo.upsert_keywords(today, [repo.KeywordRow(date=today, keyword="降准", freq=10, is_theme=1),
                                  repo.KeywordRow(date=today, keyword="新能源", freq=8, is_theme=1)])
    repo.upsert_industries(today, [repo.IndustryRow(date=today, industry="新能源", hit_count=8)])
    prev = (date.today() - timedelta(days=1)).isoformat()
    repo.upsert_metric(repo.DailyMetric(date=prev, article_count=25, total_words=4200,
                                        sentiment_index=50, policy_stance_score=0.0,
                                        attention_entropy=0.75, attention_top_share=0.55))
    repo.upsert_keywords(prev, [repo.KeywordRow(date=prev, keyword="降准", freq=8, is_theme=1),
                                repo.KeywordRow(date=prev, keyword="楼市", freq=5, is_theme=1)])
    repo.upsert_industries(prev, [repo.IndustryRow(date=prev, industry="房地产", hit_count=5)])

    cmp_res = compare(today, prev_date=prev)
    import json
    print(json.dumps(cmp_res.as_dict(), ensure_ascii=False, indent=2))
