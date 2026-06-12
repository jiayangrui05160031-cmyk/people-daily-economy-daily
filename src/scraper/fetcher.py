"""fetcher — HTTP 请求会话 ==============================================
封装 requests.Session,处理 UA / Referer / 限速 / 失败重试 / 编码。
"""

from __future__ import annotations

import time
from typing import Optional

import requests

from src.config import (
    REFERER,
    REQUEST_INTERVAL_SEC,
    REQUEST_TIMEOUT_SEC,
    USER_AGENT,
)
from src.utils.logger import get_logger

logger = get_logger("scraper.fetcher")


class Fetcher:
    """轻量 HTTP 客户端,自带限速、重试、自动编码。"""

    def __init__(
        self,
        user_agent: str = USER_AGENT,
        referer: str = REFERER,
        interval: float = REQUEST_INTERVAL_SEC,
        max_retries: int = 3,
    ) -> None:
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Referer": referer,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate",
        })
        self.interval = interval
        self.max_retries = max_retries
        self._last_request_at: float = 0.0

    def _throttle(self) -> None:
        """确保两次请求间隔 >= self.interval。"""
        now = time.time()
        wait = self.interval - (now - self._last_request_at)
        if wait > 0:
            time.sleep(wait)

    def get(self, url: str, timeout: int = REQUEST_TIMEOUT_SEC) -> str:
        """GET 请求,返回 HTML 文本(自动编码)。失败按指数退避重试。"""
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                resp = self.session.get(url, timeout=timeout)
                self._last_request_at = time.time()
                resp.raise_for_status()
                # 自动判定编码(GBK/UTF-8)
                resp.encoding = resp.apparent_encoding or "utf-8"
                logger.debug(f"GET {url} → {resp.status_code} ({len(resp.text)} bytes)")
                return resp.text
            except Exception as e:
                last_err = e
                wait = 2 ** attempt
                logger.warning(f"GET {url} 第 {attempt}/{self.max_retries} 次失败: {e},等待 {wait}s")
                time.sleep(wait)
                self._last_request_at = time.time()
        raise RuntimeError(f"GET {url} 失败 {self.max_retries} 次: {last_err}")

    def close(self) -> None:
        self.session.close()