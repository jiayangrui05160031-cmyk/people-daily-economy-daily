"""storage.db - SQLite time-series DB layer (autocommit + WAL)"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from src.config import TIMESERIES_DB_PATH
from src.utils.logger import get_logger

logger = get_logger("storage.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS article (
    date TEXT NOT NULL,
    article_id TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT,
    source TEXT,
    publish_time TEXT,
    channel TEXT,
    word_count INTEGER DEFAULT 0,
    content_text TEXT,
    PRIMARY KEY (date, article_id)
);
CREATE INDEX IF NOT EXISTS idx_article_date ON article(date);
CREATE INDEX IF NOT EXISTS idx_article_source ON article(source);

CREATE TABLE IF NOT EXISTS daily_metric (
    date TEXT PRIMARY KEY,
    article_count INTEGER,
    total_words INTEGER,
    unique_keywords INTEGER,
    policy_stance_score REAL,
    sentiment_index REAL,
    attention_entropy REAL,
    attention_top_share REAL,
    industry_count INTEGER,
    policy_count INTEGER,
    event_count INTEGER,
    raw_json TEXT
);

CREATE TABLE IF NOT EXISTS keyword_daily (
    date TEXT NOT NULL,
    keyword TEXT NOT NULL,
    freq INTEGER DEFAULT 0,
    tfidf REAL DEFAULT 0.0,
    is_theme INTEGER DEFAULT 0,
    theme_score REAL DEFAULT 0.0,
    PRIMARY KEY (date, keyword)
);
CREATE INDEX IF NOT EXISTS idx_kw_date ON keyword_daily(date);
CREATE INDEX IF NOT EXISTS idx_kw_keyword ON keyword_daily(keyword);

CREATE TABLE IF NOT EXISTS industry_daily (
    date TEXT NOT NULL,
    industry TEXT NOT NULL,
    hit_count INTEGER DEFAULT 0,
    article_count INTEGER DEFAULT 0,
    heat TEXT,
    stance TEXT,
    PRIMARY KEY (date, industry)
);
CREATE INDEX IF NOT EXISTS idx_ind_date ON industry_daily(date);
CREATE INDEX IF NOT EXISTS idx_ind_industry ON industry_daily(industry);

CREATE TABLE IF NOT EXISTS entity_daily (
    date TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_value TEXT NOT NULL,
    freq INTEGER DEFAULT 1,
    PRIMARY KEY (date, entity_type, entity_value)
);
CREATE INDEX IF NOT EXISTS idx_ent_date ON entity_daily(date);
CREATE INDEX IF NOT EXISTS idx_ent_value ON entity_daily(entity_value);

CREATE TABLE IF NOT EXISTS ai_report (
    date TEXT PRIMARY KEY,
    provider TEXT,
    model TEXT,
    payload_json TEXT,
    self_eval_consistency REAL,
    self_eval_groundedness REAL,
    self_eval_completeness REAL,
    self_eval_overall REAL,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS source_health (
    date TEXT NOT NULL,
    source TEXT NOT NULL,
    success_count INTEGER DEFAULT 0,
    fail_count INTEGER DEFAULT 0,
    latency_ms INTEGER DEFAULT 0,
    PRIMARY KEY (date, source)
);
"""

_LOCK = threading.Lock()
_CONN = None  # type: ignore[var-annotated]


def _connect():
    conn = sqlite3.connect(str(TIMESERIES_DB_PATH), detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def get_conn():
    global _CONN
    if _CONN is None:
        with _LOCK:
            if _CONN is None:
                _CONN = _connect()
    return _CONN


@contextmanager
def tx():
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def close():
    global _CONN
    if _CONN is not None:
        _CONN.close()
        _CONN = None


def reset_for_tests():
    close()
    if TIMESERIES_DB_PATH.exists():
        TIMESERIES_DB_PATH.unlink()
    get_conn()


def to_json(v):
    return json.dumps(v, ensure_ascii=False, default=str)


if __name__ == "__main__":
    conn = get_conn()
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    print("DB OK, tables:", [r[0] for r in rows])
    print("DB path:", TIMESERIES_DB_PATH)
