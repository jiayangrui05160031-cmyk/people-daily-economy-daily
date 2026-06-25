"""trend.emerging - 热点涌现与衰退检测

通过对比近 N 天 vs 之前 M 天的词频,识别:
- emerging: 新出现或频次大幅上升
- declining: 频次大幅下降
- persistent: 持续高频
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import timedelta
from typing import List, Tuple

from src.config import TREND_EMERGING_RATIO, TREND_DECLINING_RATIO
from src.storage import repository as repo
from src.storage.db import get_conn
from src.utils.date_utils import parse_date
from src.utils.logger import get_logger

logger = get_logger("trend.emerging")


@dataclass
class TrendKeyword:
    keyword: str
    recent_freq: int
    prev_freq: int
    ratio: float
    trend: str
    momentum: float
    dates: List[Tuple[str, int]]


@dataclass
class TrendReport:
    window_recent: List[str]
    window_prev: List[str]
    emerging: List[TrendKeyword]
    declining: List[TrendKeyword]
    persistent: List[TrendKeyword]

    def as_dict(self):
        return {
            "window_recent": self.window_recent,
            "window_prev": self.window_prev,
            "emerging": [asdict(k) for k in self.emerging[:30]],
            "declining": [asdict(k) for k in self.declining[:30]],
            "persistent": [asdict(k) for k in self.persistent[:30]],
        }


def _series(keyword, dates):
    if not dates:
        return []
    conn = get_conn()
    qmarks = ",".join("?" * len(dates))
    cur = conn.execute(
        f"SELECT date, freq FROM keyword_daily WHERE keyword=? AND date IN ({qmarks}) ORDER BY date ASC",
        [keyword] + dates,
    )
    return [(r[0], r[1]) for r in cur.fetchall()]


def _window_freq(keyword, dates):
    if not dates:
        return 0
    conn = get_conn()
    qmarks = ",".join("?" * len(dates))
    return conn.execute(
        f"SELECT COALESCE(SUM(freq),0) FROM keyword_daily WHERE keyword=? AND date IN ({qmarks})",
        [keyword] + dates,
    ).fetchone()[0]


def detect(today_date, recent_days=3, prev_days=7, min_total=3, top_k=50):
    today = parse_date(today_date)
    recent = [(today - timedelta(days=i)).isoformat() for i in range(recent_days)][::-1]
    prev = [(today - timedelta(days=i)).isoformat() for i in range(recent_days, recent_days + prev_days)][::-1]

    conn = get_conn()
    qmarks = ",".join("?" * len(recent))
    top_rows = conn.execute(
        f"SELECT keyword, SUM(freq) AS total FROM keyword_daily WHERE date IN ({qmarks}) "
        f"GROUP BY keyword ORDER BY total DESC LIMIT ?",
        recent + [top_k],
    ).fetchall()
    candidates = [r[0] for r in top_rows]

    emerging, declining, persistent = [], [], []
    for kw in candidates:
        r_freq = _window_freq(kw, recent)
        p_freq = _window_freq(kw, prev)
        total = r_freq + p_freq
        if total < min_total:
            continue
        ratio = (r_freq / max(p_freq, 1)) if p_freq > 0 else (TREND_EMERGING_RATIO if r_freq > 0 else 0.0)
        momentum = (r_freq - p_freq) / max(p_freq, 1)
        series = _series(kw, recent + prev)

        if p_freq == 0 and r_freq > 0:
            trend = "emerging"
        elif ratio >= TREND_EMERGING_RATIO:
            trend = "emerging"
        elif p_freq > 0 and r_freq / p_freq <= TREND_DECLINING_RATIO:
            trend = "declining"
        elif r_freq >= 5 and p_freq >= 5 and 0.5 <= ratio <= 2.0:
            trend = "persistent"
        else:
            trend = "stable"

        tk = TrendKeyword(keyword=kw, recent_freq=r_freq, prev_freq=p_freq,
                          ratio=round(ratio, 3), trend=trend,
                          momentum=round(momentum, 3), dates=series)
        if trend == "emerging":
            emerging.append(tk)
        elif trend == "declining":
            declining.append(tk)
        elif trend == "persistent":
            persistent.append(tk)

    emerging.sort(key=lambda x: x.ratio, reverse=True)
    declining.sort(key=lambda x: x.ratio)
    persistent.sort(key=lambda x: x.recent_freq, reverse=True)

    return TrendReport(window_recent=recent, window_prev=prev,
                       emerging=emerging, declining=declining, persistent=persistent)


if __name__ == "__main__":
    from src.storage import db
    db.get_conn()
    from datetime import date
    today = date.today()
    for i in range(10):
        d = (today - timedelta(days=i)).isoformat()
        repo.upsert_keywords(d, [
            repo.KeywordRow(date=d, keyword="降准", freq=max(1, 8 - i)),
            repo.KeywordRow(date=d, keyword="新能源", freq=max(1, 6 + i)),
            repo.KeywordRow(date=d, keyword="楼市", freq=max(1, 5 - i // 2)),
        ])
    tr = detect(today.isoformat(), recent_days=3, prev_days=7)
    print("emerging:", [t.keyword for t in tr.emerging])
    print("declining:", [t.keyword for t in tr.declining])
    print("persistent:", [t.keyword for t in tr.persistent])
