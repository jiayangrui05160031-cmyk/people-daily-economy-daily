"""cache.py - SQLite WAL 模式响应缓存 =========================================
- 按 (date, articles_hash, task_name) 缓存 LLM 原始响应
- 重复运行零成本(同一天相同文章不再调 API)
- WAL 模式 + 索引,支持高并发读 + 单写
- 自动 30 天过期
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from src.utils.logger import get_logger

logger = get_logger("ai.cache")

# 缓存目录
CACHE_DIR: Path = Path(__file__).resolve().parent.parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DB_PATH: Path = CACHE_DIR / "llm_cache.sqlite3"

# 默认过期 30 天
DEFAULT_TTL_DAYS = 30


def _articles_hash(articles: List[Any]) -> str:
    """计算文章列表的稳定哈希(用于缓存 key)。"""
    # 用 (title, url, word_count) 做指纹,避免大文本
    items = sorted(
        f"{a.title}|{a.url}|{a.word_count}" for a in articles
        if hasattr(a, "title")
    )
    raw = "\n".join(items).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


class LLMCache:
    """线程安全的 SQLite 缓存(每次调用独立 connection,WAL 模式并发 OK)。"""

    def __init__(self, db_path: Path = CACHE_DB_PATH, ttl_days: int = DEFAULT_TTL_DAYS):
        self.db_path = db_path
        self.ttl_days = ttl_days
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(str(self.db_path), timeout=10, isolation_level=None)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        c.execute("PRAGMA temp_store=MEMORY")
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self) -> None:
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS llm_responses (
                    cache_key TEXT PRIMARY KEY,
                    date TEXT NOT NULL,
                    task_name TEXT NOT NULL,
                    articles_hash TEXT NOT NULL,
                    prompt_tokens INTEGER DEFAULT 0,
                    completion_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    cost_cny REAL DEFAULT 0,
                    model TEXT,
                    raw_content TEXT NOT NULL,
                    parsed_json TEXT,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_date ON llm_responses(date)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_expires ON llm_responses(expires_at)")

    @staticmethod
    def make_key(date: str, task_name: str, articles_hash: str, model: str = "") -> str:
        raw = f"{date}|{task_name}|{articles_hash}|{model}".encode()
        return hashlib.sha256(raw).hexdigest()[:24]

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM llm_responses WHERE cache_key=? AND expires_at > ?",
                (cache_key, time.time()),
            ).fetchone()
            if not row:
                return None
            return dict(row)

    def put(
        self,
        cache_key: str,
        date: str,
        task_name: str,
        articles_hash: str,
        raw_content: str,
        parsed_json: Optional[Dict[str, Any]] = None,
        model: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_cny: float = 0.0,
    ) -> None:
        now = time.time()
        with self._conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO llm_responses
                   (cache_key, date, task_name, articles_hash, model,
                    raw_content, parsed_json,
                    prompt_tokens, completion_tokens, total_tokens, cost_cny,
                    created_at, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    cache_key, date, task_name, articles_hash, model,
                    raw_content,
                    json.dumps(parsed_json, ensure_ascii=False) if parsed_json else None,
                    prompt_tokens, completion_tokens, prompt_tokens + completion_tokens,
                    cost_cny, now, now + self.ttl_days * 86400,
                ),
            )

    def stats(self, date: Optional[str] = None) -> Dict[str, Any]:
        """统计缓存命中情况(用于 trace)。"""
        with self._conn() as c:
            if date:
                rows = c.execute(
                    "SELECT task_name, COUNT(*) as n, SUM(total_tokens) as tok, SUM(cost_cny) as cost "
                    "FROM llm_responses WHERE date=? GROUP BY task_name",
                    (date,),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT task_name, COUNT(*) as n, SUM(total_tokens) as tok, SUM(cost_cny) as cost "
                    "FROM llm_responses GROUP BY task_name"
                ).fetchall()
            return {r["task_name"]: dict(r) for r in rows}

    def clear_expired(self) -> int:
        with self._conn() as c:
            cur = c.execute("DELETE FROM llm_responses WHERE expires_at < ?", (time.time(),))
            return cur.rowcount

    def clear_all(self) -> int:
        with self._conn() as c:
            cur = c.execute("DELETE FROM llm_responses")
            return cur.rowcount


# 全局单例
_default_cache: Optional[LLMCache] = None


def get_default_cache() -> LLMCache:
    global _default_cache
    if _default_cache is None:
        _default_cache = LLMCache()
    return _default_cache


if __name__ == "__main__":
    # 自检
    c = LLMCache()
    arts_hash = _articles_hash([type("X", (), {"title": "a", "url": "u", "word_count": 1})()])
    k = c.make_key("2026-06-12", "test_task", arts_hash, "MiniMax-M3")
    print(f"key: {k}")

    # miss
    assert c.get(k) is None, "should miss first"

    # put
    c.put(k, "2026-06-12", "test_task", arts_hash, "raw response", {"x": 1}, "MiniMax-M3",
          prompt_tokens=10, completion_tokens=20, cost_cny=0.001)

    # hit
    row = c.get(k)
    assert row is not None
    assert row["parsed_json"] is not None
    print(f'hit: parsed_json={row["parsed_json"][:50]!r}, tokens={row["total_tokens"]}')

    # stats
    print(f'stats: {c.stats("2026-06-12")}')

    # 过期清理(模拟)
    with c._conn() as conn:
        conn.execute("UPDATE llm_responses SET expires_at = 0")
    deleted = c.clear_expired()
    print(f"cleared {deleted} expired")

    c.clear_all()
    print("All cache self-tests passed")

