"""scraper.anti_ban - 反爬策略 (UA 池 + 抖动 + 退避)"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Dict, Optional

from src.config import USER_AGENT_POOL
from src.utils.logger import get_logger

logger = get_logger("scraper.anti_ban")


class AntiBanPolicy:
    def __init__(self, ua_pool=None, base_delay=1.0, jitter=0.5, max_retries=3):
        self.ua_pool = ua_pool or USER_AGENT_POOL
        self.base_delay = base_delay
        self.jitter = jitter
        self.max_retries = max_retries
        self._ua_idx = 0
        self._last_request_at = 0.0

    def next_ua(self):
        ua = self.ua_pool[self._ua_idx % len(self.ua_pool)]
        self._ua_idx += 1
        return ua

    def jittered_delay(self):
        return self.base_delay + random.uniform(0, self.jitter)

    def build_headers(self, referer=""):
        return {
            "User-Agent": self.next_ua(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Referer": referer or "https://www.baidu.com/",
            "Cache-Control": "no-cache",
        }

    def wait(self):
        delay = self.jittered_delay()
        elapsed = time.time() - self._last_request_at
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_request_at = time.time()


def exponential_backoff(attempt, base=1.0, cap=16.0):
    return min(cap, base * (2 ** attempt) + random.uniform(0, 0.5))


if __name__ == "__main__":
    p = AntiBanPolicy()
    print("UA:", p.next_ua())
    print("headers:", p.build_headers())
    print("delay:", p.jittered_delay())
