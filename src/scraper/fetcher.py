"""fetcher — HTTP fetcher with anti-ban + retry

升级点:
- AntiBanPolicy 注入 (UA 池 / 抖动)
- 指数退避重试
- 抓取耗时统计
"""
from __future__ import annotations

import time
from typing import Optional

import requests

from src.config import REQUEST_INTERVAL_SEC, REQUEST_TIMEOUT_SEC
from src.scraper.anti_ban import AntiBanPolicy, exponential_backoff
from src.utils.logger import get_logger

logger = get_logger("scraper.fetcher")


class Fetcher:
    def __init__(
        self,
        base_delay: float = REQUEST_INTERVAL_SEC,
        jitter: float = 0.6,
        max_retries: int = 3,
        timeout: int = REQUEST_TIMEOUT_SEC,
        policy: Optional[AntiBanPolicy] = None,
    ):
        self.policy = policy or AntiBanPolicy(base_delay=base_delay, jitter=jitter)
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()

    def get(self, url: str, referer: str = "", skip_wait: bool = False) -> str:
        """抓取 URL,带反爬 + 重试。"""
        last_err = None
        for attempt in range(self.max_retries):
            try:
                if not skip_wait:
                    self.policy.wait()
                headers = self.policy.build_headers(referer=referer)
                t0 = time.time()
                resp = self.session.get(url, headers=headers, timeout=self.timeout)
                latency = (time.time() - t0) * 1000
                if resp.status_code == 200:
                    resp.encoding = resp.apparent_encoding or "utf-8"
                    logger.debug(f"GET {url} ok {latency:.0f}ms ({len(resp.text)} chars)")
                    return resp.text
                if resp.status_code in (403, 429):
                    wait = exponential_backoff(attempt)
                    logger.warning(f"GET {url} status={resp.status_code}, retry in {wait:.1f}s")
                    time.sleep(wait)
                    last_err = RuntimeError(f"status {resp.status_code}")
                    continue
                resp.raise_for_status()
                return resp.text
            except requests.RequestException as e:
                last_err = e
                wait = exponential_backoff(attempt)
                logger.warning(f"GET {url} attempt {attempt + 1} failed: {e}; retry in {wait:.1f}s")
                time.sleep(wait)
        raise RuntimeError(f"failed to fetch {url} after {self.max_retries} retries: {last_err}")

    def close(self):
        try:
            self.session.close()
        except Exception:
            pass


if __name__ == "__main__":
    f = Fetcher()
    try:
        text = f.get("https://www.baidu.com/", skip_wait=True)
        print(f"ok: {len(text)} chars")
    finally:
        f.close()
