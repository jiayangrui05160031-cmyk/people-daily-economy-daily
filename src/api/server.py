"""api.server - FastAPI REST 入口 (v6)

把 16 个分析模块 + 多智能体 + SHAP 解释暴露为 HTTP API:

  System
    GET  /                                项目介绍
    GET  /health                          健康检查 (DB / AI / 模块)
    GET  /metrics                         Prometheus 指标
    GET  /modules                         16 模块状态
    GET  /dates                           有数据的日期列表
    GET  /dates/latest                    最新报告日期
  Report
    GET  /report/{date}                   Markdown 报告原文
    GET  /report/{date}/json              报告元数据
    GET  /dashboard/{date}                仪表盘 HTML
    GET  /dashboard/index                 仪表盘历史索引
  Analytics
    GET  /metrics/daily/{date}            量化指标
    GET  /timeseries                      时序 (近 30 天)
    GET  /industries                      行业列表
  v6 Advanced
    GET  /v6/risk                         风险指标 (Sharpe/MaxDD/VaR...)
    GET  /v6/portfolio                    行业组合回测
    GET  /v6/scenario/list                7 个情景定义
    GET  /v6/scenario/run                 运行蒙特卡洛情景
    POST /v6/qa                           智能问答 (RAG + LLM)
  Frontier (v6 新增)
    POST /v6/council                      多智能体顾问团
    GET  /v6/shap                         SHAP 决策解释
  Ops
    POST /v6/run                          触发端到端 (异步)
    WS   /ws/stream                       WebSocket 实时日志

启动:
    python -m src.api.server --port 8000
    uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from fastapi import FastAPI, HTTPException, Query, Body, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, Response, PlainTextResponse, HTMLResponse, FileResponse
    from pydantic import BaseModel, Field
except ImportError as e:
    raise SystemExit(
        "FastAPI 未安装. 请先: pip install fastapi uvicorn[standard] pydantic"
    ) from e

from src.config import REPORT_DIR, DASHBOARD_DIR, TIMESERIES_DB_PATH
from src.storage import db as db_mod
from src.storage import repository as repo
from src.utils.date_utils import parse_date, resolve_target_date, yesterday

# v6 子模块
from src.api.observability import (
    check_health, install_middleware, setup_json_logging,
    render_prometheus, INMEM,
    inc_counter, observe_histogram,
)
from src.api.multi_agent import council as run_council
from src.api.shap_explain import explain_decision as run_shap


# ============================================================
# Pydantic v2 schemas
# ============================================================
class HealthResp(BaseModel):
    ok: bool
    db: bool
    ai_router: bool
    last_report_date: Optional[str]
    modules_total: int
    modules_healthy: int
    uptime_seconds: float
    version: str


class ModuleStatus(BaseModel):
    name: str
    version: str          # v5 / v6
    healthy: bool
    file: str
    note: str = ""


class DateItem(BaseModel):
    date: str
    article_count: int
    sentiment_index: float
    policy_stance_score: float


class DailyMetricsResp(BaseModel):
    date: str
    article_count: int
    total_words: int
    unique_keywords: int
    sentiment_index: float
    policy_stance_score: float
    attention_entropy: float
    attention_top_share: float
    industry_count: int
    policy_count: int
    event_count: int


class TimeSeriesPoint(BaseModel):
    date: str
    sentiment_index: float
    policy_stance_score: float
    industry_count: int
    attention_entropy: float


class IndustryItem(BaseModel):
    name: str
    hit_count: int
    stance: str
    weight: float


class QABody(BaseModel):
    question: str = Field(..., min_length=2, max_length=500)
    target_date: Optional[str] = None
    top_k: int = Field(5, ge=1, le=20)


class CouncilBody(BaseModel):
    question: str = Field(..., min_length=2, max_length=500)
    target_date: Optional[str] = None
    use_llm: bool = True


class ScenarioQuery(BaseModel):
    target_date: Optional[str] = None
    scenario: str = "rate_cut_50bp"
    horizon_days: int = Field(30, ge=5, le=180)
    n_sims: int = Field(500, ge=50, le=5000)


class PortfolioQuery(BaseModel):
    target_date: Optional[str] = None
    industries: Optional[List[str]] = None
    weights: Optional[Dict[str, float]] = None
    rebalance: str = "weekly"
    lookback: int = Field(60, ge=10, le=180)


class RiskQuery(BaseModel):
    target_date: Optional[str] = None
    lookback: int = Field(90, ge=10, le=365)
    rf_rate: float = 0.025


# ============================================================
# 工具
# ============================================================
def _safe(obj: Any) -> Any:
    """递归把 dataclass/dict 转 JSON 可序列化对象."""
    if obj is None:
        return None
    if is_dataclass(obj):
        return _safe(asdict(obj) if hasattr(obj, "as_dict") is False else obj.as_dict())
    if isinstance(obj, dict):
        return {str(k): _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(x) for x in obj]
    if isinstance(obj, (int, float, str, bool)):
        return obj
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    # 其它不可序列化 -> repr
    try:
        return str(obj)
    except Exception:
        return f"<unserializable: {type(obj).__name__}>"


def _to_json(obj: Any) -> Any:
    return _safe(obj)


def _resolve_date(date_str: Optional[str]) -> str:
    if not date_str:
        return resolve_target_date("").isoformat()
    try:
        return parse_date(date_str).isoformat()
    except Exception:
        return resolve_target_date("").isoformat()


# 模块清单 (16 个)
MODULES_V5 = [
    ("anomaly", "src.analysis.anomaly", "1. 异常检测 (Z-score 滚动)"),
    ("forecast", "src.analysis.forecast", "2. 时序预测 (MA + 95% 置信带)"),
    ("stock_correlation", "src.analysis.stock_correlation", "3. 产业-A 股联动"),
    ("macro_indicators", "src.analysis.macro_indicators", "4. 宏观指标 (CPI/PMI/PPI)"),
    ("topic_model", "src.analysis.topic_model", "5. 主题建模 (LDA)"),
    ("event_study", "src.analysis.event_study", "6. 事件研究 (6 类政策)"),
    ("rag_history", "src.analysis.rag_history", "7. RAG 历史回放 (TF-IDF)"),
    ("volatility", "src.analysis.volatility", "8. 波动率 / 恐慌指数"),
    ("causal", "src.analysis.causal", "9. 因果分析 (政策-市场)"),
    ("signal_engine", "src.analysis.signal_engine", "10. 多信号决策引擎"),
    ("forecast_backtest", "src.analysis.forecast_backtest", "11. 预测回测 (walk-forward)"),
    ("narrative", "src.analysis.narrative", "12. LLM 高管简报"),
]
MODULES_V6 = [
    ("qa_assistant", "src.analysis.qa_assistant", "13. RAG 智能问答 (v6)"),
    ("risk_metrics", "src.analysis.risk_metrics", "14. 风险指标 Sharpe/VaR/CVaR (v6)"),
    ("portfolio", "src.analysis.portfolio", "15. 行业组合回测 + alpha/beta (v6)"),
    ("scenario", "src.analysis.scenario", "16. 情景分析 + 蒙特卡洛 (v6)"),
]


# ============================================================
# Lifespan
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_json_logging("INFO")
    from src.utils.logger import get_logger
    log = get_logger("api.server")
    log.info("=" * 60)
    log.info("宏观经济智能分析 API v6 启动")
    log.info("=" * 60)
    yield
    log.info("API shutdown")


# ============================================================
# FastAPI app
# ============================================================
app = FastAPI(
    title="宏观经济智能分析 API",
    description=(
        "**v6** · 16 个分析模块 · 多智能体顾问团 · SHAP 决策解释 · "
        "FastAPI REST · Prometheus 可观测"
    ),
    version="9.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# 立即挂载中间件 (不能放在 lifespan 里)
install_middleware(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 0. 根路径
# ============================================================
@app.get("/", response_class=HTMLResponse, tags=["system"])
async def root() -> str:
    return """<!DOCTYPE html>
<html lang=zh>
<head>
<meta charset=utf-8>
<title>宏观经济智能分析 API v6</title>
<style>
  body { font-family: -apple-system, "Segoe UI", sans-serif; max-width: 920px; margin: 40px auto; padding: 0 20px; color: #1f2937; }
  h1 { border-bottom: 2px solid #4f46e5; padding-bottom: 8px; }
  .card { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; margin: 12px 0; }
  code { background: #eef2ff; padding: 2px 6px; border-radius: 4px; font-size: 90%; }
  a { color: #4f46e5; }
  ul { line-height: 1.8; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }
  .badge-v5 { background: #dbeafe; color: #1e40af; }
  .badge-v6 { background: #fef3c7; color: #92400e; }
</style>
</head>
<body>
<h1>📈 宏观经济智能分析 API v6</h1>

<p>基于 <code>Requests + BeautifulSoup + jieba + scikit-learn + Plotly + minimax LLM</code> 的宏观新闻智能分析平台, 现已暴露为 <strong>FastAPI REST</strong>.</p>

<div class="card">
  <h3>🚀 快速开始</h3>
  <ul>
    <li><a href="/docs">/docs</a> — Swagger UI (推荐)</li>
    <li><a href="/redoc">/redoc</a> — ReDoc 文档</li>
    <li><a href="/health">/health</a> — 健康检查</li>
    <li><a href="/modules">/modules</a> — 16 模块状态</li>
    <li><a href="/metrics">/metrics</a> — Prometheus 指标</li>
  </ul>
</div>

<div class="card">
  <h3>🧠 16 个分析模块</h3>
  <h4><span class="badge badge-v5">v5</span> 12 个传统模块</h4>
  <ul>
    <li>anomaly · forecast · stock_correlation · macro_indicators</li>
    <li>topic_model · event_study · rag_history · volatility</li>
    <li>causal · signal_engine · forecast_backtest · narrative</li>
  </ul>
  <h4><span class="badge badge-v6">v6</span> 4 个新增模块</h4>
  <ul>
    <li>qa_assistant — RAG 智能问答</li>
    <li>risk_metrics — Sharpe/Sortino/MaxDD/VaR/CVaR</li>
    <li>portfolio — 行业组合回测 + alpha/beta/IR</li>
    <li>scenario — 蒙特卡洛 1000 次 + 7 情景</li>
  </ul>
  <h4><span class="badge badge-v6">v6 frontier</span> 前沿特性</h4>
  <ul>
    <li><strong>multi_agent</strong> — 4 智能体顾问团 (鹰/鸽/量化/务实) + 仲裁</li>
    <li><strong>shap_explain</strong> — SHAP 风格决策可解释性</li>
    <li><strong>prometheus</strong> — 业务可观测性</li>
  </ul>
</div>

<p>© 2026 · 人民日报经济新闻每日热点分析 · v6 升级版</p>
</body></html>"""


# ============================================================
# 1. System endpoints
# ============================================================
@app.get("/health", response_model=HealthResp, tags=["system"])
async def health() -> HealthResp:
    h = check_health()
    return HealthResp(**h.as_dict())


@app.get("/metrics", tags=["system"])
async def prometheus() -> Response:
    body, ctype = render_prometheus()
    return Response(content=body, media_type=ctype)


@app.get("/modules", tags=["system"])
async def modules() -> Dict[str, Any]:
    out = []
    for name, path, note in MODULES_V5 + MODULES_V6:
        healthy = False
        try:
            __import__(path, fromlist=["*"])
            healthy = True
        except Exception as e:
            note = f"{note} (加载失败: {e})"
        version = "v6" if name in {m[0] for m in MODULES_V6} else "v5"
        out.append({
            "name": name, "version": version, "healthy": healthy,
            "file": path.replace(".", "/") + ".py", "note": note,
        })
    return {
        "total": len(out),
        "v5": sum(1 for m in out if m["version"] == "v5"),
        "v6": sum(1 for m in out if m["version"] == "v6"),
        "healthy": sum(1 for m in out if m["healthy"]),
        "modules": out,
    }


@app.get("/dates", tags=["system"])
async def list_dates(limit: int = Query(60, ge=1, le=365)) -> List[DateItem]:
    try:
        with db_mod.get_conn() as conn:
            rows = conn.execute(
                "SELECT date, article_count, sentiment_index, policy_stance_score "
                "FROM daily_metric ORDER BY date DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [DateItem(
            date=r["date"], article_count=int(r["article_count"] or 0),
            sentiment_index=float(r["sentiment_index"] or 0),
            policy_stance_score=float(r["policy_stance_score"] or 0),
        ) for r in rows]
    except Exception as e:
        raise HTTPException(503, f"DB 不可用: {e}")


@app.get("/dates/latest", tags=["system"])
async def latest_date() -> Dict[str, str]:
    try:
        with db_mod.get_conn() as conn:
            row = conn.execute(
                "SELECT MAX(date) AS d FROM daily_metric"
            ).fetchone()
        d = row["d"] if row and row["d"] else _resolve_date(None)
        return {"date": d}
    except Exception as e:
        raise HTTPException(503, f"DB 不可用: {e}")


# ============================================================
# 2. Report endpoints
# ============================================================
@app.get("/report/{date}", tags=["report"])
async def get_report(date: str) -> Response:
    d = _resolve_date(date)
    fp = REPORT_DIR / f"{d}.md"
    if not fp.exists():
        raise HTTPException(404, f"报告不存在: {fp}")
    body = fp.read_text(encoding="utf-8")
    inc_counter("API_REPORT_TOTAL", {"date": d})
    return HTMLResponse(
        f"<html><head><meta charset=utf-8><title>{d} 报告</title>"
        f"<style>body{{font-family:monospace;max-width:1080px;margin:24px auto;padding:0 20px;line-height:1.7}}"
        f"h1,h2,h3{{color:#1e40af}}pre,code{{background:#f3f4f6;padding:2px 4px;border-radius:4px}}</style></head>"
        f"<body><a href='/docs'>← API</a>{body}</body></html>"
    )


@app.get("/report/{date}/json", tags=["report"])
async def get_report_meta(date: str) -> Dict[str, Any]:
    d = _resolve_date(date)
    fp = REPORT_DIR / f"{d}.md"
    if not fp.exists():
        raise HTTPException(404, f"报告不存在: {fp}")
    body = fp.read_text(encoding="utf-8")
    sections = [line.strip() for line in body.split("\n") if line.startswith("## ")]
    return {
        "date": d,
        "file": str(fp),
        "size_bytes": fp.stat().st_size,
        "section_count": len(sections),
        "sections": sections[:35],
    }


@app.get("/dashboard/{date}", tags=["report"])
async def get_dashboard(date: str) -> FileResponse:
    d = _resolve_date(date)
    fp = DASHBOARD_DIR / f"{d}.html"
    if not fp.exists():
        raise HTTPException(404, f"仪表盘不存在: {fp}")
    return FileResponse(str(fp), media_type="text/html")


@app.get("/dashboard/index", tags=["report"])
async def get_dashboard_index() -> FileResponse:
    fp = DASHBOARD_DIR / "index.html"
    if not fp.exists():
        raise HTTPException(404, "历史索引尚未生成, 请先运行 main.py")
    return FileResponse(str(fp), media_type="text/html")


# ============================================================
# 3. Analytics endpoints
# ============================================================
@app.get("/metrics/daily/{date}", response_model=DailyMetricsResp, tags=["analytics"])
async def daily_metrics(date: str) -> DailyMetricsResp:
    d = _resolve_date(date)
    try:
        with db_mod.get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM daily_metric WHERE date = ?", (d,),
            ).fetchone()
        if not row:
            raise HTTPException(404, f"该日期无量化指标: {d}")
        return DailyMetricsResp(
            date=d,
            article_count=int(row["article_count"] or 0),
            total_words=int(row["total_words"] or 0),
            unique_keywords=int(row["unique_keywords"] or 0),
            sentiment_index=float(row["sentiment_index"] or 50.0),
            policy_stance_score=float(row["policy_stance_score"] or 0.0),
            attention_entropy=float(row["attention_entropy"] or 0.0),
            attention_top_share=float(row["attention_top_share"] or 0.0),
            industry_count=int(row["industry_count"] or 0),
            policy_count=int(row["policy_count"] or 0),
            event_count=int(row["event_count"] or 0),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(503, f"DB 错误: {e}")


@app.get("/timeseries", tags=["analytics"])
async def timeseries(days: int = Query(30, ge=1, le=365)) -> List[TimeSeriesPoint]:
    try:
        with db_mod.get_conn() as conn:
            rows = conn.execute(
                "SELECT date, sentiment_index, policy_stance_score, industry_count, attention_entropy "
                "FROM daily_metric ORDER BY date DESC LIMIT ?",
                (days,),
            ).fetchall()
        return [TimeSeriesPoint(
            date=r["date"],
            sentiment_index=float(r["sentiment_index"] or 0),
            policy_stance_score=float(r["policy_stance_score"] or 0),
            industry_count=int(r["industry_count"] or 0),
            attention_entropy=float(r["attention_entropy"] or 0),
        ) for r in rows]
    except Exception as e:
        raise HTTPException(503, f"DB 错误: {e}")


@app.get("/industries", tags=["analytics"])
async def industries(date: Optional[str] = Query(None)) -> List[IndustryItem]:
    d = _resolve_date(date)
    try:
        with db_mod.get_conn() as conn:
            rows = conn.execute(
                "SELECT industry, hit_count, stance FROM industry_daily "
                "WHERE date = ? ORDER BY hit_count DESC LIMIT 20",
                (d,),
            ).fetchall()
        out = []
        for r in rows:
            hit = int(r["hit_count"] or 0)
            out.append(IndustryItem(
                name=r["industry"], hit_count=hit, stance=r["stance"] or "中性",
                weight=round(hit / max(1, sum(int(x["hit_count"] or 0) for x in rows)), 4),
            ))
        return out
    except Exception as e:
        raise HTTPException(503, f"DB 错误: {e}")


# ============================================================
# 4. v6 Advanced endpoints
# ============================================================
@app.get("/v6/risk", tags=["v6-advanced"])
async def v6_risk(
    target_date: Optional[str] = Query(None),
    lookback: int = Query(90, ge=10, le=365),
    rf_rate: float = Query(0.025, ge=0.0, le=0.2),
) -> Dict[str, Any]:
    d = _resolve_date(target_date)
    from src.analysis.risk_metrics import compute as risk_compute
    t0 = time.time()
    try:
        rep = risk_compute(d, lookback=lookback, rf_rate=rf_rate)
        if rep is None:
            raise HTTPException(404, f"风险指标数据不足: {d}")
        dur = round((time.time() - t0) * 1000, 1)
        observe_histogram("API_V6_RISK_MS", {"ok": "1"}, dur)
        return _to_json(rep)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"风险模块失败: {e}")


@app.post("/v6/portfolio", tags=["v6-advanced"])
async def v6_portfolio(q: PortfolioQuery) -> Dict[str, Any]:
    d = _resolve_date(q.target_date)
    from src.analysis.portfolio import backtest as portfolio_backtest
    from src.config import INDUSTRY_TO_STOCKS
    if q.industries:
        industries = [i for i in q.industries if i in INDUSTRY_TO_STOCKS]
        if not industries:
            industries = ["新能源", "半导体", "金融", "消费"]
    else:
        industries = ["新能源", "半导体", "金融", "消费"]
    if q.weights:
        weights = {i: q.weights.get(i, 0.0) for i in industries}
        # 归一化
        s = sum(weights.values()) or 1.0
        weights = {i: v / s for i, v in weights.items()}
    else:
        weights = {i: 1.0 / len(industries) for i in industries}
    try:
        rep = portfolio_backtest(d, portfolio=weights, rebalance=q.rebalance, lookback=q.lookback)
        if rep is None:
            raise HTTPException(404, "组合回测数据不足")
        return _to_json(rep)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"组合模块失败: {e}")


@app.get("/v6/scenario/list", tags=["v6-advanced"])
async def v6_scenario_list() -> Dict[str, Any]:
    from src.analysis.scenario import SCENARIOS, list_scenarios
    return {
        "scenarios": list_scenarios() if callable(list_scenarios) else list(SCENARIOS.keys()),
        "definitions": {k: v.get("name", k) for k, v in SCENARIOS.items()},
    }


@app.post("/v6/scenario/run", tags=["v6-advanced"])
async def v6_scenario_run(q: ScenarioQuery) -> Dict[str, Any]:
    from src.analysis.scenario import run as scenario_run
    d = _resolve_date(q.target_date)
    try:
        rep = scenario_run(d, scenario=q.scenario, horizon_days=q.horizon_days, n_sims=q.n_sims)
        if rep is None:
            raise HTTPException(404, "情景分析数据不足")
        return _to_json(rep)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"情景模块失败: {e}")


@app.post("/v6/qa", tags=["v6-advanced"])
async def v6_qa(body: QABody) -> Dict[str, Any]:
    from src.analysis.qa_assistant import ask as qa_ask
    d = _resolve_date(body.target_date)
    router = None
    try:
        from src.ai.router import get_default_router
        router = get_default_router()
    except Exception:
        pass
    try:
        rep = qa_ask(body.question, target_date=d, router=router, top_k=body.top_k)
        if rep is None:
            raise HTTPException(503, "问答模块失败")
        return _to_json(rep)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"问答失败: {e}")


# ============================================================
# 5. Frontier endpoints (v6 前沿特性)
# ============================================================
@app.post("/v6/council", tags=["v6-frontier"])
async def v6_council(body: CouncilBody) -> Dict[str, Any]:
    """多智能体顾问团: 4 个 LLM/人设 + 1 个仲裁者."""
    d = _resolve_date(body.target_date)
    router = None
    if body.use_llm:
        try:
            from src.ai.router import get_default_router
            router = get_default_router()
        except Exception:
            pass
    try:
        rep = run_council(body.question, d, router=router, use_llm=body.use_llm)
        return _to_json(rep)
    except Exception as e:
        raise HTTPException(500, f"顾问团失败: {e}")


@app.get("/v6/shap", tags=["v6-frontier"])
async def v6_shap(target_date: Optional[str] = Query(None)) -> Dict[str, Any]:
    """SHAP 风格特征贡献度: 解释 signal_engine 的 BUY/HOLD/REDUCE/SELL."""
    d = _resolve_date(target_date)
    try:
        rep = run_shap(d)
        if rep is None:
            raise HTTPException(404, "SHAP 解释失败: 数据不足")
        return _to_json(rep)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"SHAP 解释失败: {e}")


# ============================================================
# 6. Operations
# ============================================================
@app.post("/v6/run", tags=["ops"])
async def v6_run(
    target_date: Optional[str] = Body(None),
    skip_scrape: bool = Body(True),
    skip_ai: bool = Body(False),
) -> Dict[str, Any]:
    """触发端到端 pipeline (异步)."""
    d = _resolve_date(target_date)

    async def _run():
        cmd = [
            sys.executable, "-X", "utf8", "-m", "main",
            "--date", d, "--window-days", "14",
        ]
        if skip_scrape:
            cmd.append("--skip-scrape")
        if skip_ai:
            cmd.append("--skip-ai")
        else:
            cmd.append("--no-dashboard")  # 后台跑不阻塞
            cmd.append("--no-kg")
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=str(ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return {
            "returncode": proc.returncode,
            "stdout_tail": (stdout or b"")[-2000:].decode("utf-8", errors="replace"),
            "stderr_tail": (stderr or b"")[-1000:].decode("utf-8", errors="replace"),
        }
    res = await _run()
    return {"date": d, "result": res}


# ============================================================
# 7. WebSocket 实时日志流
# ============================================================
@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket) -> None:
    """WebSocket 实时流: 推送最新分析摘要 (10s 心跳)."""
    await ws.accept()
    try:
        await ws.send_json({"type": "welcome", "ts": datetime.utcnow().isoformat() + "Z",
                            "msg": "已连接宏观经济 API v6 实时流"})
        # 推 3 帧示例
        for i in range(3):
            await asyncio.sleep(2)
            try:
                with db_mod.get_conn() as conn:
                    row = conn.execute(
                        "SELECT date, sentiment_index, policy_stance_score, industry_count "
                        "FROM daily_metric ORDER BY date DESC LIMIT 1"
                    ).fetchone()
            except Exception:
                row = None
            await ws.send_json({
                "type": "tick", "n": i + 1,
                "ts": datetime.utcnow().isoformat() + "Z",
                "metric": dict(row) if row else None,
                "inmem": {
                    "uptime_s": round(time.time() - INMEM.started_at, 1),
                    "counters": len(INMEM.counters),
                },
            })
        # 长轮询等待客户端消息
        while True:
            msg = await ws.receive_text()
            if msg == "ping":
                await ws.send_text("pong")
            elif msg == "metrics":
                await ws.send_json({"type": "metrics", "snapshot": INMEM.snapshot()})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "msg": str(e)})
        except Exception:
            pass


# ============================================================
# 8. 自检
# ============================================================
@app.get("/_selftest", tags=["system"])
async def selftest() -> Dict[str, Any]:
    """API 自检: 健康 + 16 模块 + 几个端点 dry-run."""
    out: Dict[str, Any] = {"ts": datetime.utcnow().isoformat() + "Z"}
    out["health"] = check_health().as_dict()
    try:
        with db_mod.get_conn() as conn:
            n = conn.execute("SELECT COUNT(*) AS c FROM daily_metric").fetchone()["c"]
        out["db_rows"] = n
    except Exception as e:
        out["db_error"] = str(e)
    # 模块
    out["modules"] = []
    for name, path, note in MODULES_V5 + MODULES_V6:
        try:
            __import__(path, fromlist=["*"])
            out["modules"].append({"name": name, "ok": True})
        except Exception as e:
            out["modules"].append({"name": name, "ok": False, "err": str(e)[:80]})
    return out


# ============================================================

# ============================================================
# 7. v6 Frontier Streaming (SSE + WebSocket)
# ============================================================
@app.get("/v6/stream/qa", tags=["v6-frontier"])
async def stream_qa(
    question: str = Query(..., min_length=2, max_length=300),
    target_date: Optional[str] = Query(None),
    top_k: int = Query(3, ge=1, le=10),
) -> StreamingResponse:
    """SSE 实时流式问答 (RAG + LLM token-by-token)."""
    from fastapi.responses import StreamingResponse
    from src.api.streaming import stream_qa_answer
    d = _resolve_date(target_date)
    router = None
    try:
        from src.ai.router import get_default_router
        router = get_default_router()
    except Exception:
        pass
    return StreamingResponse(
        stream_qa_answer(question, d, router=router, top_k=top_k),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/v6/stream/council", tags=["v6-frontier"])
async def stream_council_sse(
    question: str = Query(..., min_length=2, max_length=300),
    target_date: Optional[str] = Query(None),
    use_llm: bool = Query(False),
) -> StreamingResponse:
    """SSE 实时多智能体顾问团 (4 角色流式)."""
    from fastapi.responses import StreamingResponse
    from src.api.streaming import stream_council as _stream_council
    d = _resolve_date(target_date)
    router = None
    if use_llm:
        try:
            from src.ai.router import get_default_router
            router = get_default_router()
        except Exception:
            pass
    return StreamingResponse(
        _stream_council(question, d, router=router, use_llm=use_llm),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/v6/stream/health", tags=["v6-frontier"])
async def stream_health(interval: float = Query(2.0, ge=0.5, le=60.0)) -> StreamingResponse:
    """SSE 实时健康心跳."""
    from fastapi.responses import StreamingResponse
    from src.api.streaming import stream_health_ticks
    return StreamingResponse(
        stream_health_ticks(interval),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# ============================================================
# 8. WebSocket 实时告警
# ============================================================
_ALERT_WS_STARTED = False

@app.websocket("/v6/alerts")
async def ws_alerts(ws: WebSocket) -> None:
    """WebSocket 告警. ?types=anomaly,signal,risk"""
    from src.api.alerts import get_alert_manager
    am = get_alert_manager()
    subscribed = None
    try:
        qp = ws.query_params
        types_param = qp.get("types")
        if types_param:
            subscribed = set(t.strip() for t in types_param.split(",") if t.strip())
    except Exception:
        pass
    await am.connect(ws, subscribed)
    global _ALERT_WS_STARTED
    if not _ALERT_WS_STARTED:
        _ALERT_WS_STARTED = True
        await am.start_background_scanner(interval=60.0)
    try:
        while True:
            msg = await ws.receive_text()
            if msg == "ping":
                await ws.send_text("pong")
            elif msg == "stats":
                await ws.send_json({"type": "stats", "stats": am.stats()})
    except WebSocketDisconnect:
        await am.disconnect(ws)
    except Exception:
        try:
            await am.disconnect(ws)
        except Exception:
            pass

@app.get("/v6/alerts/stats", tags=["v6-frontier"])
async def alerts_stats() -> Dict[str, Any]:
    from src.api.alerts import get_alert_manager
    return get_alert_manager().stats()

# ============================================================
# 9. AI 智能爬虫 (LLM 替代 BeautifulSoup)
# ============================================================
class ExtractBody(BaseModel):
    html: str = Field(..., min_length=20, max_length=200000)
    url: str = Field("", max_length=500)
    hint_title: str = Field("", max_length=200)

@app.post("/v6/extract", tags=["v6-frontier"])
async def ai_extract(body: ExtractBody) -> Dict[str, Any]:
    """AI 智能提取: 主题/情感/关键词一并产出."""
    from src.scraper.ai_extractor import AIExtractor
    ext = AIExtractor()
    r = ext.extract(body.html, url=body.url, hint_title=body.hint_title)
    return _to_json(r)

class RelevanceBody(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    summary: str = Field("", max_length=500)
    url: str = Field("", max_length=500)

@app.post("/v6/relevance", tags=["v6-frontier"])
async def ai_relevance(body: RelevanceBody) -> Dict[str, Any]:
    """AI 相关性评分."""
    from src.scraper.ai_relevance import AIRelevanceFilter
    f = AIRelevanceFilter()
    r = f.filter(body.title, body.summary, body.url)
    return _to_json(r)

class DedupBody(BaseModel):
    art_a: Dict[str, Any]
    art_b: Dict[str, Any]

@app.post("/v6/dedup", tags=["v6-frontier"])
async def ai_dedup(body: DedupBody) -> Dict[str, Any]:
    """AI 语义去重."""
    from src.scraper.ai_dedup import AIDeduper
    d = AIDeduper()
    r = d.is_duplicate(body.art_a, body.art_b)
    return _to_json(r)

# ============================================================
# 10. Embedding 向量 RAG (v9: route to retrieval.VectorRetriever)
# ============================================================
_EMBED_RAG_INSTANCE = None

def _get_embed_rag():
    """v9: returns a VectorRetriever. The legacy `EmbedRAG` class
    (src.analysis.embed_rag) was deleted because it duplicated
    embeddings.EmbeddingStore. server.py's /v6/embed/* endpoints now
    use the new retrieval/ subpackage."""
    global _EMBED_RAG_INSTANCE
    if _EMBED_RAG_INSTANCE is None:
        from src.retrieval import VectorRetriever
        _EMBED_RAG_INSTANCE = VectorRetriever()
    return _EMBED_RAG_INSTANCE

class EmbedAddBody(BaseModel):
    doc_id: str = Field(..., min_length=1, max_length=100)
    text: str = Field(..., min_length=10, max_length=2000)
    metadata: Optional[Dict[str, Any]] = None

@app.post("/v6/embed/add", tags=["v6-frontier"])
async def embed_add(body: EmbedAddBody) -> Dict[str, Any]:
    rag = _get_embed_rag()
    rag.add(body.doc_id, body.text, body.metadata or {})
    return {"ok": True, "stats": rag.backend_info()}

class EmbedQueryBody(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(5, ge=1, le=20)
    min_score: float = Field(0.0, ge=0.0, le=1.0)

@app.post("/v6/embed/query", tags=["v6-frontier"])
async def embed_query(body: EmbedQueryBody) -> Dict[str, Any]:
    rag = _get_embed_rag()
    hits = rag.search(body.text, top_k=body.top_k, min_score=body.min_score)
    return {
        "query": body.text,
        "backend": rag.backend_info().get("backend", "unknown"),
        "hits": [_to_json(h) for h in hits],
    }

@app.get("/v6/embed/stats", tags=["v6-frontier"])
async def embed_stats() -> Dict[str, Any]:
    return _get_embed_rag().backend_info()

# CLI 启动
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="宏观经济 API v6")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    try:
        import uvicorn
    except ImportError:
        raise SystemExit("uvicorn 未安装. 请先: pip install uvicorn[standard]")
    print(f"\n>>> 宏观经济智能分析 API v6 starting on http://{args.host}:{args.port}")
    print(f">>> Swagger UI: http://{args.host}:{args.port}/docs\n")
    uvicorn.run(
        "src.api.server:app",
        host=args.host, port=args.port,
        reload=args.reload, workers=args.workers,
    )


if __name__ == "__main__":
    main()
