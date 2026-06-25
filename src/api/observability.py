"""api.observability - 可观测性模块 (Prometheus metrics + 结构化日志 + 请求追踪)

核心能力:
  1) Prometheus 指标 — Counter / Histogram / Gauge 全套
  2) 端点请求埋点中间件 (PROMETHEUS-aware)
  3) 结构化 JSON 日志 (可对接 ELK / Loki)
  4) 链路追踪 ID (X-Request-ID 自动注入)

不依赖 prometheus_client 时降级为 in-memory dict (开发模式仍能跑)。
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, Info,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
    )
    HAS_PROM = True
except ImportError:
    HAS_PROM = False

from src.config import LOG_DIR
from src.utils.logger import get_logger

logger = get_logger("api.observability")

REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="-")


# ============================================================
# 1) Prometheus 指标注册表
# ============================================================
if HAS_PROM:
    REGISTRY = CollectorRegistry()

    # 业务指标
    API_REQUESTS_TOTAL = Counter(
        "macro_api_requests_total",
        "Total HTTP requests by endpoint / method / status",
        ["endpoint", "method", "status"],
        registry=REGISTRY,
    )
    API_REQUEST_DURATION = Histogram(
        "macro_api_request_duration_seconds",
        "Request latency in seconds",
        ["endpoint", "method"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
        registry=REGISTRY,
    )
    API_ACTIVE_REQUESTS = Gauge(
        "macro_api_active_requests",
        "Currently in-flight requests",
        ["endpoint"],
        registry=REGISTRY,
    )

    # 业务域指标
    MODULE_RUNS_TOTAL = Counter(
        "macro_module_runs_total",
        "Analysis module invocations",
        ["module", "status"],
        registry=REGISTRY,
    )
    MODULE_RUN_DURATION = Histogram(
        "macro_module_run_duration_seconds",
        "Module run duration in seconds",
        ["module"],
        buckets=(0.01, 0.1, 0.5, 1.0, 5.0, 30.0, 60.0, 300.0),
        registry=REGISTRY,
    )
    LLM_CALLS_TOTAL = Counter(
        "macro_llm_calls_total",
        "LLM API calls by provider / model / status",
        ["provider", "model", "status"],
        registry=REGISTRY,
    )
    LLM_TOKENS_TOTAL = Counter(
        "macro_llm_tokens_total",
        "LLM tokens consumed",
        ["provider", "model", "direction"],  # direction: prompt / completion
        registry=REGISTRY,
    )

    # 系统指标
    DB_QUERIES_TOTAL = Counter(
        "macro_db_queries_total",
        "DB queries by table",
        ["table"],
        registry=REGISTRY,
    )
    CACHE_HITS_TOTAL = Counter(
        "macro_cache_hits_total",
        "LLM cache hits / misses",
        ["result"],  # hit / miss
        registry=REGISTRY,
    )
    APP_INFO = Info(
        "macro_app",
        "Application metadata",
        registry=REGISTRY,
    )
    APP_INFO.info({
        "name": "macro-intelligence",
        "version": "v6",
        "modules": "16",
    })


# ============================================================
# 2) 内存降级 (无 prometheus_client 也能跑)
# ============================================================
@dataclass
class InMemoryMetric:
    """轻量内存指标, 用于开发或无 prom 依赖时."""
    counters: Dict[str, float] = field(default_factory=dict)
    histograms: Dict[str, List[float]] = field(default_factory=dict)
    gauges: Dict[str, float] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)

    def inc(self, name: str, labels: Dict[str, str], value: float = 1.0) -> None:
        key = f"{name}|" + ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        self.counters[key] = self.counters.get(key, 0.0) + value

    def observe(self, name: str, labels: Dict[str, str], value: float) -> None:
        key = f"{name}|" + ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        self.histograms.setdefault(key, []).append(value)

    def set_gauge(self, name: str, labels: Dict[str, str], value: float) -> None:
        key = f"{name}|" + ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        self.gauges[key] = value

    def snapshot(self) -> Dict[str, Any]:
        # 计算每个 histogram 的 p50 / p95
        hist_stats = {}
        for key, vals in self.histograms.items():
            if not vals:
                continue
            s = sorted(vals)
            n = len(s)
            p50 = s[int(n * 0.5)]
            p95 = s[min(int(n * 0.95), n - 1)]
            p99 = s[min(int(n * 0.99), n - 1)]
            hist_stats[key] = {
                "count": n, "p50": round(p50, 4),
                "p95": round(p95, 4), "p99": round(p99, 4),
                "max": round(max(s), 4), "min": round(min(s), 4),
            }
        return {
            "uptime_seconds": round(time.time() - self.started_at, 2),
            "counters": dict(self.counters),
            "histograms": hist_stats,
            "gauges": dict(self.gauges),
        }


INMEM = InMemoryMetric()


# ============================================================
# 3) 指标统一接口 (Prom 优先, 否则内存)
# ============================================================
def inc_counter(name: str, labels: Optional[Dict[str, str]] = None, value: float = 1.0) -> None:
    labels = labels or {}
    if HAS_PROM:
        try:
            metric = globals().get(name)
            if metric is not None and hasattr(metric, "labels"):
                metric.labels(**labels).inc(value)
                return
        except Exception:
            pass
    INMEM.inc(name, labels, value)


def observe_histogram(name: str, labels: Optional[Dict[str, str]] = None, value: float = 0.0) -> None:
    labels = labels or {}
    if HAS_PROM:
        try:
            metric = globals().get(name)
            if metric is not None and hasattr(metric, "labels"):
                metric.labels(**labels).observe(value)
                return
        except Exception:
            pass
    INMEM.observe(name, labels, value)


def set_gauge(name: str, labels: Optional[Dict[str, str]] = None, value: float = 0.0) -> None:
    labels = labels or {}
    if HAS_PROM:
        try:
            metric = globals().get(name)
            if metric is not None and hasattr(metric, "labels"):
                metric.labels(**labels).set(value)
                return
        except Exception:
            pass
    INMEM.set_gauge(name, labels, value)


def render_prometheus() -> tuple:
    """导出 Prometheus 文本格式; 不可用时返回 in-memory 快照的 JSON."""
    if HAS_PROM:
        return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
    body = json.dumps(INMEM.snapshot(), ensure_ascii=False, indent=2)
    return body.encode("utf-8"), "application/json"


# ============================================================
# 4) FastAPI 中间件
# ============================================================
def install_middleware(app) -> None:
    """挂载请求埋点 + 链路追踪 + 结构化日志到 FastAPI app."""
    try:
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.requests import Request
    except ImportError:
        logger.warning("starlette 不可用, 跳过中间件安装")
        return

    class ObservabilityMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: "Request", call_next):
            rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
            token = REQUEST_ID.set(rid)
            start = time.time()
            status = 500
            try:
                response = await call_next(request)
                status = response.status_code
                return response
            except Exception as e:
                logger.exception("请求异常", extra={"rid": rid})
                raise
            finally:
                dur = time.time() - start
                endpoint = request.url.path
                method = request.method
                REQUEST_ID.reset(token)
                inc_counter("API_REQUESTS_TOTAL", {
                    "endpoint": endpoint, "method": method, "status": str(status),
                })
                observe_histogram("API_REQUEST_DURATION", {
                    "endpoint": endpoint, "method": method,
                }, dur)
                logger.info(
                    f"{method} {endpoint} {status} {dur*1000:.1f}ms rid={rid}",
                    extra={"rid": rid, "latency_ms": round(dur * 1000, 2)},
                )

    app.add_middleware(ObservabilityMiddleware)


# ============================================================
# 5) 结构化 JSON 日志 (用于 ELK / Loki 聚合)
# ============================================================
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        rid = ""
        try:
            rid = REQUEST_ID.get()
        except LookupError:
            pass
        payload = {
            "ts": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "rid": rid,
        }
        for attr in ("latency_ms", "module", "status"):
            v = getattr(record, attr, None)
            if v is not None:
                payload[attr] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_json_logging(level: str = "INFO") -> None:
    """把根 logger 切换为 JSON 格式; 写到 logs/api.jsonl."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "api.jsonl"
    root = logging.getLogger()
    # 避免重复挂 handler
    has_json = any(isinstance(h, logging.FileHandler) and getattr(h, "_json_tag", False)
                   for h in root.handlers)
    if has_json:
        return
    h = logging.FileHandler(log_file, encoding="utf-8")
    h._json_tag = True  # type: ignore
    h.setFormatter(JsonFormatter())
    root.addHandler(h)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.info(f"JSON 日志已挂载: {log_file}")


# ============================================================
# 6) 健康探针 (供 /health 端点用)
# ============================================================
@dataclass
class HealthStatus:
    ok: bool
    db: bool
    ai_router: bool
    last_report_date: Optional[str]
    modules_total: int
    modules_healthy: int
    uptime_seconds: float
    version: str
    detail: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def check_health(router=None) -> HealthStatus:
    """运行时健康检查: DB / AI / 模块状态."""
    from src.storage import db as db_mod
    from src.storage import repository as repo

    db_ok = False
    last_date = None
    try:
        with db_mod.get_conn() as conn:
            row = conn.execute("SELECT MAX(date) AS d FROM daily_metric").fetchone()
            last_date = row["d"] if row else None
        db_ok = True
    except Exception as e:
        logger.debug(f"DB 检查失败: {e}")

    ai_ok = False
    if router is not None:
        try:
            ai_ok = bool(getattr(router, "api_key", None))
        except Exception:
            pass
    else:
        try:
            from src.ai.router import get_default_router
            r = get_default_router()
            ai_ok = bool(getattr(r, "api_key", None))
        except Exception:
            ai_ok = False

    modules = [
        "anomaly", "forecast", "stock_correlation", "macro_indicators",
        "topic_model", "event_study", "rag_history", "volatility",
        "causal", "signal_engine", "forecast_backtest", "narrative",
        # v6
        "qa_assistant", "risk_metrics", "portfolio", "scenario",
    ]
    modules_healthy = 0
    for m in modules:
        try:
            __import__(f"src.analysis.{m}", fromlist=["*"])
            modules_healthy += 1
        except Exception:
            pass

    return HealthStatus(
        ok=db_ok and ai_ok and modules_healthy >= 12,
        db=db_ok,
        ai_router=ai_ok,
        last_report_date=last_date,
        modules_total=len(modules),
        modules_healthy=modules_healthy,
        uptime_seconds=round(time.time() - INMEM.started_at, 2),
        version="v6",
    )
