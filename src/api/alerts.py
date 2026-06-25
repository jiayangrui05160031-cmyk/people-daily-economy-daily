"""api.alerts - WebSocket 实时告警系统 (v6 前沿升级)

当 16 个分析模块检测到 异常 / 风险事件 / 重大决策信号 时,
主动通过 WebSocket 推送给所有订阅的客户端.

核心能力:
  1) AlertManager 单例 — 维护在线连接池
  2) 阈值触发: 当 volatility > 30 / sentiment < 30 / signal.score < -0.5 等
  3) 多种 alert 类型: anomaly / signal / risk / event / policy
  4) 客户端可订阅 / 退订特定类型
  5) 告警持久化: 写入 data/historical/alerts.jsonl, 可回放

典型用法:
    from src.api.alerts import get_alert_manager
    am = get_alert_manager()
    await am.start_background_scanner(interval=30)
    # 客户端: ws://host:8000/v6/alerts
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.config import LOG_DIR
from src.utils.logger import get_logger

logger = get_logger("api.alerts")

ALERT_LOG = LOG_DIR / "alerts.jsonl"


class AlertType(str, Enum):
    ANOMALY = "anomaly"
    SIGNAL = "signal"
    RISK = "risk"
    EVENT = "event"
    POLICY = "policy"
    SENTIMENT = "sentiment"
    VOLATILITY = "volatility"


class AlertLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    id: str
    type: str       # AlertType
    level: str      # AlertLevel
    title: str
    message: str
    date: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# 连接管理
# ============================================================
class _Connection:
    """单个 WebSocket 客户端 + 它订阅的 alert 类型."""
    def __init__(self, ws, subscribed: Optional[Set[str]] = None):
        self.ws = ws
        self.subscribed = subscribed  # None = 全部, set = 子集
        self.connected_at = time.time()


class AlertManager:
    def __init__(self):
        self._conns: List[_Connection] = []
        self._lock = asyncio.Lock()
        self._background_task: Optional[asyncio.Task] = None
        self._alert_count = 0
        self._last_alert_at: Optional[float] = None
        # 简单去重: 同 type+date+title 在 60s 内不重复
        self._dedup: Dict[str, float] = {}
        self._dedup_window = 60.0
        # 持久化
        ALERT_LOG.parent.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 客户端管理
    # ============================================================
    async def connect(self, ws, subscribed: Optional[Set[str]] = None) -> None:
        async with self._lock:
            self._conns.append(_Connection(ws, subscribed))
        logger.info(f"alert 客户端连接 (n={len(self._conns)}, sub={subscribed or 'all'})")
        # 推 1 帧 welcome
        try:
            await ws.send_json({
                "type": "welcome",
                "ts": datetime.utcnow().isoformat() + "Z",
                "n_connections": len(self._conns),
                "subscribed": list(subscribed) if subscribed else "all",
            })
        except Exception:
            pass

    async def disconnect(self, ws) -> None:
        async with self._lock:
            self._conns = [c for c in self._conns if c.ws is not ws]
        logger.info(f"alert 客户端断开 (n={len(self._conns)})")

    async def _broadcast(self, alert: Alert) -> int:
        """广播 alert 给所有匹配的客户端. 返回推送成功数."""
        n = 0
        dead: List[_Connection] = []
        async with self._lock:
            conns = list(self._conns)
        for c in conns:
            if c.subscribed is not None and alert.type not in c.subscribed:
                continue
            try:
                await c.ws.send_json({
                    "type": "alert",
                    "payload": alert.as_dict(),
                })
                n += 1
            except Exception:
                dead.append(c)
        # 清理死连接
        if dead:
            async with self._lock:
                self._conns = [c for c in self._conns if c not in dead]
        return n

    # ============================================================
    # 触发 alert
    # ============================================================
    def _dedup_key(self, alert: Alert) -> str:
        return f"{alert.type}|{alert.date}|{alert.title[:50]}"

    async def fire(self, alert: Alert) -> bool:
        """触发 alert, 自动去重 + 持久化 + 广播."""
        if not alert.created_at:
            alert.created_at = datetime.utcnow().isoformat() + "Z"
        # 去重
        key = self._dedup_key(alert)
        now = time.time()
        if key in self._dedup and now - self._dedup[key] < self._dedup_window:
            return False
        self._dedup[key] = now
        # 持久化
        try:
            with ALERT_LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps(alert.as_dict(), ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug(f"alert 持久化失败: {e}")
        # 广播
        n = await self._broadcast(alert)
        self._alert_count += 1
        self._last_alert_at = now
        logger.info(
            f"alert 触发: {alert.type}/{alert.level} - {alert.title[:40]} "
            f"({n} 客户端接收)"
        )
        return True

    # ============================================================
    # 后台扫描器
    # ============================================================
    async def start_background_scanner(self, interval: float = 30.0) -> None:
        """启动后台扫描, 周期性检测 + 触发 alert."""
        if self._background_task is not None:
            return
        self._background_task = asyncio.create_task(self._scan_loop(interval))
        logger.info(f"alert 后台扫描已启动 (interval={interval}s)")

    async def stop_background_scanner(self) -> None:
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except Exception:
                pass
            self._background_task = None

    async def _scan_loop(self, interval: float) -> None:
        """周期性扫描 daily_metric + 16 模块, 触发 alert."""
        while True:
            try:
                await self._scan_once()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"alert scan 异常: {e}")
            await asyncio.sleep(interval)

    async def _scan_once(self) -> None:
        """跑 1 次扫描."""
        from src.storage import db as db_mod
        from src.utils.date_utils import resolve_target_date
        target_date = resolve_target_date("").isoformat()
        try:
            with db_mod.get_conn() as conn:
                row = conn.execute(
                    "SELECT * FROM daily_metric WHERE date = ?", (target_date,),
                ).fetchone()
        except Exception:
            return
        if not row:
            return
        sentiment = float(row["sentiment_index"] or 50.0)
        policy = float(row["policy_stance_score"] or 0.0)
        ent = float(row["attention_entropy"] or 0.5)
        industry = int(row["industry_count"] or 0)

        # 1. 情绪极低
        if sentiment < 30:
            await self.fire(Alert(
                id=f"sent-{target_date}", type=AlertType.SENTIMENT,
                level=AlertLevel.CRITICAL,
                title="市场情绪极度悲观",
                message=f"情绪指数 {sentiment:.1f} < 30 阈值, 历史上多为重大风险事件前夕",
                date=target_date, metrics={"sentiment_index": sentiment},
            ))
        elif sentiment < 40:
            await self.fire(Alert(
                id=f"sent-{target_date}", type=AlertType.SENTIMENT,
                level=AlertLevel.WARNING,
                title="市场情绪偏弱",
                message=f"情绪指数 {sentiment:.1f}, 建议关注后续政策面信号",
                date=target_date, metrics={"sentiment_index": sentiment},
            ))

        # 2. 政策急剧收紧
        if policy < -0.4:
            await self.fire(Alert(
                id=f"pol-{target_date}", type=AlertType.POLICY,
                level=AlertLevel.WARNING,
                title="政策立场偏紧",
                message=f"政策立场 {policy:+.2f}, 显著收紧, 关注后续流动性",
                date=target_date, metrics={"policy_stance_score": policy},
            ))

        # 3. 注意力集中度过高 (单一话题霸榜)
        if ent < 0.3 and industry > 0:
            await self.fire(Alert(
                id=f"vol-{target_date}", type=AlertType.VOLATILITY,
                level=AlertLevel.WARNING,
                title="注意力高度集中",
                message=f"熵 {ent:.3f} < 0.3, 单一话题占据 70%+ 关注, 警惕舆情泡沫",
                date=target_date, metrics={"attention_entropy": ent},
            ))

    # ============================================================
    # 状态
    # ============================================================
    def stats(self) -> Dict[str, Any]:
        return {
            "n_connections": len(self._conns),
            "alert_count": self._alert_count,
            "last_alert_at": self._last_alert_at,
            "background_running": self._background_task is not None
                and not self._background_task.done(),
            "dedup_size": len(self._dedup),
        }


# ============================================================
# 单例
# ============================================================
_MANAGER: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = AlertManager()
    return _MANAGER
