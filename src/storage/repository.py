"""storage.repository - 时序 DB 读写仓库

每个表一组函数:upsert_* / get_* / list_*。
调用方传 dataclass 或 dict,内部统一打包成 SQL。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date as _date
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.storage import db
from src.utils.logger import get_logger

logger = get_logger("storage.repo")


# ============================================================
# 数据类 (轻量 DTO)
# ============================================================
@dataclass
class ArticleRow:
    date: str
    article_id: str
    title: str
    url: str = ""
    source: str = ""
    publish_time: str = ""
    channel: str = ""
    word_count: int = 0
    content_text: str = ""


@dataclass
class DailyMetric:
    date: str
    article_count: int = 0
    total_words: int = 0
    unique_keywords: int = 0
    policy_stance_score: float = 0.0
    sentiment_index: float = 50.0
    attention_entropy: float = 0.0
    attention_top_share: float = 0.0
    industry_count: int = 0
    policy_count: int = 0
    event_count: int = 0
    raw_json: str = ""


@dataclass
class KeywordRow:
    date: str
    keyword: str
    freq: int = 0
    tfidf: float = 0.0
    is_theme: int = 0
    theme_score: float = 0.0


@dataclass
class IndustryRow:
    date: str
    industry: str
    hit_count: int = 0
    article_count: int = 0
    heat: str = ""
    stance: str = ""


@dataclass
class EntityRow:
    date: str
    entity_type: str
    entity_value: str
    freq: int = 1


@dataclass
class AIReportRow:
    date: str
    provider: str = ""
    model: str = ""
    payload_json: str = ""
    self_eval_consistency: float = 0.0
    self_eval_groundedness: float = 0.0
    self_eval_completeness: float = 0.0
    self_eval_overall: float = 0.0
    created_at: str = ""


# ============================================================
# Articles
# ============================================================
def upsert_articles(date, articles):
    rows = [
        (a.date, a.article_id, a.title, a.url, a.source, a.publish_time,
         a.channel, a.word_count, a.content_text)
        for a in articles
    ]
    if not rows:
        return 0
    with db.tx() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO article "
            "(date, article_id, title, url, source, publish_time, channel, word_count, content_text) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            rows,
        )
    return len(rows)


def get_articles(date):
    conn = db.get_conn()
    cur = conn.execute(
        "SELECT * FROM article WHERE date=? ORDER BY publish_time DESC", (date,)
    )
    return [ArticleRow(**dict(r)) for r in cur.fetchall()]


def count_articles(date):
    conn = db.get_conn()
    return conn.execute(
        "SELECT COUNT(*) FROM article WHERE date=?", (date,)
    ).fetchone()[0]


# ============================================================
# Daily metric
# ============================================================
def upsert_metric(m):
    with db.tx() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO daily_metric "
            "(date, article_count, total_words, unique_keywords, policy_stance_score, "
            " sentiment_index, attention_entropy, attention_top_share, industry_count, "
            " policy_count, event_count, raw_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (m.date, m.article_count, m.total_words, m.unique_keywords,
             m.policy_stance_score, m.sentiment_index, m.attention_entropy,
             m.attention_top_share, m.industry_count, m.policy_count,
             m.event_count, m.raw_json),
        )


def get_metric(date):
    conn = db.get_conn()
    r = conn.execute("SELECT * FROM daily_metric WHERE date=?", (date,)).fetchone()
    return DailyMetric(**dict(r)) if r else None


def list_metrics(start_date="", end_date=""):
    conn = db.get_conn()
    sql = "SELECT * FROM daily_metric"
    args = []
    if start_date and end_date:
        sql += " WHERE date BETWEEN ? AND ?"
        args = [start_date, end_date]
    elif start_date:
        sql += " WHERE date >= ?"
        args = [start_date]
    sql += " ORDER BY date ASC"
    return [DailyMetric(**dict(r)) for r in conn.execute(sql, args).fetchall()]


def latest_dates(n=30):
    conn = db.get_conn()
    cur = conn.execute(
        "SELECT DISTINCT date FROM daily_metric ORDER BY date DESC LIMIT ?", (n,)
    )
    return [r[0] for r in cur.fetchall()]


# ============================================================
# Keyword
# ============================================================
def upsert_keywords(date, kws):
    rows = [(k.date, k.keyword, k.freq, k.tfidf, k.is_theme, k.theme_score) for k in kws]
    if not rows:
        return 0
    with db.tx() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO keyword_daily "
            "(date, keyword, freq, tfidf, is_theme, theme_score) VALUES (?,?,?,?,?,?)",
            rows,
        )
    return len(rows)


def get_keywords(date, theme_only=False):
    conn = db.get_conn()
    sql = "SELECT * FROM keyword_daily WHERE date=?"
    args = [date]
    if theme_only:
        sql += " AND is_theme=1"
    sql += " ORDER BY theme_score DESC, freq DESC"
    return [KeywordRow(**dict(r)) for r in conn.execute(sql, args).fetchall()]


def get_keyword_series(keyword, days=30):
    conn = db.get_conn()
    cur = conn.execute(
        "SELECT date, freq FROM keyword_daily WHERE keyword=? "
        "ORDER BY date DESC LIMIT ?",
        (keyword, days),
    )
    return [(r[0], r[1]) for r in cur.fetchall()][::-1]


def get_top_keywords_window(start_date, end_date, top_k=50):
    conn = db.get_conn()
    cur = conn.execute(
        "SELECT keyword, SUM(freq) AS total FROM keyword_daily "
        "WHERE date BETWEEN ? AND ? GROUP BY keyword ORDER BY total DESC LIMIT ?",
        (start_date, end_date, top_k),
    )
    return [(r[0], r[1]) for r in cur.fetchall()]


# ============================================================
# Industry
# ============================================================
def upsert_industries(date, items):
    rows = [(i.date, i.industry, i.hit_count, i.article_count, i.heat, i.stance) for i in items]
    if not rows:
        return 0
    with db.tx() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO industry_daily "
            "(date, industry, hit_count, article_count, heat, stance) VALUES (?,?,?,?,?,?)",
            rows,
        )
    return len(rows)


def get_industries(date):
    conn = db.get_conn()
    cur = conn.execute(
        "SELECT * FROM industry_daily WHERE date=? ORDER BY hit_count DESC", (date,)
    )
    return [IndustryRow(**dict(r)) for r in cur.fetchall()]


def get_industry_series(industry, days=30):
    conn = db.get_conn()
    cur = conn.execute(
        "SELECT date, hit_count FROM industry_daily WHERE industry=? "
        "ORDER BY date DESC LIMIT ?",
        (industry, days),
    )
    return [(r[0], r[1]) for r in cur.fetchall()][::-1]


# ============================================================
# Entity
# ============================================================
def upsert_entities(date, items):
    rows = [(e.date, e.entity_type, e.entity_value, e.freq) for e in items]
    if not rows:
        return 0
    with db.tx() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO entity_daily "
            "(date, entity_type, entity_value, freq) VALUES (?,?,?,?)",
            rows,
        )
    return len(rows)


def get_entities(date):
    conn = db.get_conn()
    cur = conn.execute(
        "SELECT * FROM entity_daily WHERE date=? ORDER BY freq DESC", (date,)
    )
    return [EntityRow(**dict(r)) for r in cur.fetchall()]


def get_entity_series(entity_value, days=30):
    conn = db.get_conn()
    cur = conn.execute(
        "SELECT date, SUM(freq) AS f FROM entity_daily "
        "WHERE entity_value=? AND date >= date('now', ?) "
        "GROUP BY date ORDER BY date ASC",
        (entity_value, f"-{days} days"),
    )
    return [(r[0], r[1]) for r in cur.fetchall()]


# ============================================================
# AI Report
# ============================================================
def upsert_ai_report(r):
    with db.tx() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO ai_report "
            "(date, provider, model, payload_json, self_eval_consistency, "
            " self_eval_groundedness, self_eval_completeness, self_eval_overall, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (r.date, r.provider, r.model, r.payload_json,
             r.self_eval_consistency, r.self_eval_groundedness,
             r.self_eval_completeness, r.self_eval_overall, r.created_at),
        )


def get_ai_report(date):
    conn = db.get_conn()
    r = conn.execute("SELECT * FROM ai_report WHERE date=?", (date,)).fetchone()
    return AIReportRow(**dict(r)) if r else None


# ============================================================
# Source health
# ============================================================
def record_source_health(date, source, success, latency_ms=0):
    with db.tx() as conn:
        existing = conn.execute(
            "SELECT success_count, fail_count, latency_ms FROM source_health WHERE date=? AND source=?",
            (date, source),
        ).fetchone()
        if existing:
            sc, fc, lm = existing
            if success:
                sc += 1
                lm = (lm + latency_ms) // 2 if lm else latency_ms
            else:
                fc += 1
            conn.execute(
                "UPDATE source_health SET success_count=?, fail_count=?, latency_ms=? "
                "WHERE date=? AND source=?",
                (sc, fc, lm, date, source),
            )
        else:
            conn.execute(
                "INSERT INTO source_health (date, source, success_count, fail_count, latency_ms) "
                "VALUES (?,?,?,?,?)",
                (date, source, 1 if success else 0, 0 if success else 1, latency_ms),
            )


# ============================================================
# 自检
# ============================================================
if __name__ == "__main__":
    today = _date.today().isoformat()
    upsert_articles(today, [
        ArticleRow(date=today, article_id="t1", title="测试", url="http://x", content_text="abc"),
    ])
    upsert_metric(DailyMetric(date=today, article_count=1, total_words=10))
    upsert_keywords(today, [KeywordRow(date=today, keyword="测试", freq=1, is_theme=1)])
    upsert_industries(today, [IndustryRow(date=today, industry="新能源", hit_count=1)])
    upsert_entities(today, [EntityRow(date=today, entity_type="央行/货币当局", entity_value="央行")])
    upsert_ai_report(AIReportRow(date=today, provider="test", model="test", payload_json="{}"))

    print("articles:", count_articles(today))
    print("metric:", get_metric(today))
    print("kws:", len(get_keywords(today)))
    print("industries:", get_industries(today))
    print("entities:", get_entities(today))
    print("ai:", get_ai_report(today))
    print("latest dates:", latest_dates(5))
