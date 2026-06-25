"""dashboard.builder - 交互式 HTML 仪表盘生成器 (v5)

新增: 12 个前沿模块可视化
- 异常 Z-score 柱状
- 预测 vs 实际 折线 (含置信带)
- 产业-A股涨跌对比
- 宏观指标走势 (CPI/PMI/PPI/GDP)
- LDA 主题词云 (Treemap)
- 事件时间线
- RAG 相似度热图
- 波动率/恐慌指数仪表盘
- 因果链路图 (Sankey)
- 多信号决策仪表盘 (Speed Gauge)
- 预测回测准确率
- LLM 高管简报
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.analysis.anomaly import detect as detect_anomaly

from src.analysis.causal import analyze as causal_analyze
from src.analysis.event_study import study as event_study
from src.analysis.forecast import predict_next_day
from src.analysis.forecast_backtest import backtest as forecast_backtest_fn
from src.analysis.macro_indicators import snapshot as macro_snapshot
from src.analysis.narrative import generate as narrative_generate
from src.analysis.rag_history import recall as rag_recall
from src.analysis.signal_engine import synthesize as signal_synthesize
from src.analysis.stock_correlation import correlate as correlate_market
from src.analysis.topic_model import fit_with_evolution as topic_fit
from src.analysis.volatility import compute as volatility_compute
from src.config import DASHBOARD_DIR, IMAGE_DIR

from src.storage import repository as repo
from src.utils.logger import get_logger

logger = get_logger("dashboard.builder")

_TEMPLATE_DIR = Path(__file__).resolve().parent
_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=select_autoescape(enabled_extensions=("j2", "html")),
    trim_blocks=True,
    lstrip_blocks=True,
)


def _safe_call(fn, *args, **kwargs):
    """静默调用, 失败返回 None."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        logger.debug(f"dashboard module {fn.__name__} failed: {e}")
        return None


def _topic_query_text(ai_result) -> str:
    if ai_result is None:
        return ""
    parts = []
    try:
        for k in (ai_result.theme_keywords.keywords or [])[:8]:
            parts.append(getattr(k, "word", ""))
        if getattr(ai_result, "core_insights", None) and ai_result.core_insights.insights:
            parts.append(str(ai_result.core_insights.insights)[:200])
        for ind in (ai_result.industries.industries or [])[:3]:
            parts.append(str(getattr(ind, "name", "")))
    except Exception:
        pass
    return " ".join(x for x in parts if x)


def _ai_payload_for(date: str) -> Optional[Any]:
    """从 ai_report 读取当日 AI 结果."""
    return repo.get_ai_report(date)


def _build_charts(date: str, days: int = 14) -> Dict:
    end = date
    start_dt = datetime.strptime(date, "%Y-%m-%d") - timedelta(days=days)
    start = start_dt.strftime("%Y-%m-%d")

    metrics = repo.list_metrics(start_date=start, end_date=end)
    metric_map = {m.date: m for m in metrics}
    today_dict = _safe_asdict(metric_map.get(date))

    # ===== 基础 8 图数据 =====
    kws = repo.get_keywords(date, theme_only=False)[:15]
    keywords_chart = {
        "labels": [k.keyword for k in kws],
        "values": [k.freq for k in kws],
        "theme_flags": [bool(k.is_theme) for k in kws],
    }

    sentiment_chart = {
        "dates": [m.date for m in metrics],
        "values": [m.sentiment_index for m in metrics],
    }

    policy_chart = {
        "dates": [m.date for m in metrics],
        "values": [m.policy_stance_score for m in metrics],
        "confidences": [m.article_count for m in metrics],
    }

    attention_chart = {
        "dates": [m.date for m in metrics],
        "entropy": [m.attention_entropy for m in metrics],
        "top_share": [m.attention_top_share for m in metrics],
    }

    industries = repo.get_industries(date)[:12]
    industries_chart = {
        "labels": [i.industry for i in industries],
        "hits": [i.hit_count for i in industries],
        "stances": [i.stance or "中性" for i in industries],
    }

    top_kws = [k.keyword for k in kws[:5]]
    keyword_trend = {"keywords": top_kws, "series": []}
    for kw in top_kws:
        series = repo.get_keyword_series(kw, days=days)
        if series:
            keyword_trend["series"].append({
                "name": kw,
                "dates": [d for d, _ in series],
                "values": [v for _, v in series],
            })

    entities = repo.get_entities(date)[:15]
    entities_chart = {
        "labels": [e.entity_value for e in entities],
        "values": [e.freq for e in entities],
        "types": [e.entity_type for e in entities],
    }

    # ===== 前沿 12 模块数据 =====
    ai_report = _ai_payload_for(date)
    industries_hit = [i.industry for i in industries[:6]]
    theme_kws = [{"word": k.keyword, "score": k.tfidf} for k in kws[:8]]

    anomaly_rep = _safe_call(detect_anomaly, date, window_days=30)
    forecast_rep = _safe_call(predict_next_day, date, lookback_days=14)
    market_rep = _safe_call(correlate_market, date, industries_hit[:6], top_n_per_industry=3)
    macro_rep = _safe_call(macro_snapshot, date, theme_keywords=theme_kws, industries=industries_hit)
    arts = []
    try:
        from src.report.archiver import load_raw as _load_raw
        arts = _load_raw(date)
    except FileNotFoundError:
        arts = []
    except Exception:
        arts = []
    topics_rep = _safe_call(topic_fit, arts, date, history_days=3) if arts else None
    events_list = (ai_report.events.events if ai_report and hasattr(ai_report, "events") else []) if ai_report else []
    events_rep = _safe_call(event_study, events_list, target_date=date)
    rag_query = _topic_query_text(ai_report)
    rag_rep = _safe_call(rag_recall, rag_query, date, top_k=5) if rag_query else None
    vol_rep = _safe_call(volatility_compute, date, window_days=14)
    causal_rep = _safe_call(causal_analyze, events_list, target_date=date)
    backtest_rep = _safe_call(forecast_backtest_fn, date, n_days=20, horizons=(1, 3, 7))

    # 信号引擎 (需要其他模块的输出)
    signal_rep = None
    if any([anomaly_rep, forecast_rep, vol_rep, market_rep, macro_rep, events_rep, topics_rep]):
        try:
            signal_rep = signal_synthesize(
                date,
                anomaly=anomaly_rep, forecast=forecast_rep, volatility=vol_rep,
                market=market_rep, macro=macro_rep, events_study=events_rep,
                topics=topics_rep, ai_result=ai_report,
            )
        except Exception as e:
            logger.debug(f"signal_synthesize failed: {e}")

    # 简报 (使用模板兜底避免 LLM 调用)
    narrative_rep = None
    if any([anomaly_rep, forecast_rep, vol_rep, market_rep, signal_rep]):
        advanced_for_nar = {
            "anomaly": anomaly_rep, "forecast": forecast_rep, "volatility": vol_rep,
            "market": market_rep, "macro": macro_rep, "topics": topics_rep,
            "events_study": events_rep, "rag": rag_rep, "signal": signal_rep,
            "backtest": backtest_rep,
        }
        try:
            narrative_rep = narrative_generate(date, advanced_for_nar, ai_result=ai_report, router=None)
        except Exception as e:
            logger.debug(f"narrative_generate failed: {e}")

    return {
        "date": date,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "today": today_dict,
        "metrics": metrics,
        "keywords": keywords_chart,
        "sentiment": sentiment_chart,
        "policy": policy_chart,
        "attention": attention_chart,
        "industries": industries_chart,
        "keyword_trend": keyword_trend,
        "entities": entities_chart,
        # v5 新增
        "anomaly": anomaly_rep,
        "forecast": forecast_rep,
        "market": market_rep,
        "macro": macro_rep,
        "topics": topics_rep,
        "events_study": events_rep,
        "rag": rag_rep,
        "volatility": vol_rep,
        "causal": causal_rep,
        "signal": signal_rep,
        "backtest": backtest_rep,
        "narrative": narrative_rep,
        "kg_image": f"../images/knowledge_graph_{date}.png",
        "wordcloud_image": f"../images/wordcloud_{date}.png",
    }


def _safe_asdict(obj):
    if obj is None:
        return None
    if hasattr(obj, "as_dict"):
        try:
            return obj.as_dict()
        except Exception:
            pass
    try:
        return asdict(obj)
    except Exception:
        return None


def render(date: str, output_filename: Optional[str] = None,
           window_days: int = 14) -> Path:
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    output_filename = output_filename or f"{date}.html"
    output_path = DASHBOARD_DIR / output_filename

    charts = _build_charts(date, days=window_days)
    charts["metrics_json"] = json.dumps([_safe_asdict(m) for m in charts["metrics"]], ensure_ascii=False, default=str)
    # 把所有 advanced 模块的 asdict 序列化给 JS 用
    charts["advanced_json"] = json.dumps({
        k: _safe_asdict(charts.get(k)) for k in [
            "anomaly", "forecast", "market", "macro", "topics",
            "events_study", "rag", "volatility", "causal",
            "signal", "backtest", "narrative",
        ]
    }, ensure_ascii=False, default=str)

    template = _env.get_template("template.html.j2")
    rendered = template.render(**charts)
    output_path.write_text(rendered, encoding="utf-8")
    logger.info(f"dashboard rendered: {output_path} ({len(rendered)} chars)")
    return output_path


def render_index(output_filename: str = "index.html", days: int = 30) -> Path:
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    dates = repo.latest_dates(days)
    metrics = []
    for d in dates:
        m = repo.get_metric(d)
        if m:
            metrics.append(m)

    html = ["""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>宏观经济日报 · 历史索引 (v5)</title>
<style>
body{font-family:'Microsoft YaHei',sans-serif;background:#0b1220;color:#e6edf3;margin:0;padding:24px}
h1{font-size:22px;color:#58a6ff;margin:0 0 16px}
table{width:100%;border-collapse:collapse;background:#0d1117;border-radius:8px;overflow:hidden}
th,td{padding:10px 14px;text-align:left;border-bottom:1px solid #21262d}
th{background:#161b22;color:#8b949e;font-weight:500;font-size:13px}
tr:hover{background:#161b22}
a{color:#58a6ff;text-decoration:none}
.bull{display:inline-block;padding:2px 8px;border-radius:4px;font-size:12px}
.bull-up{background:#1a4d2e;color:#7ee787}
.bull-down{background:#4d1a1a;color:#ff7b72}
.bull-flat{background:#1f2937;color:#8b949e}
.tag{font-size:11px;padding:1px 6px;border-radius:3px;background:#21262d;color:#8b949e;margin-left:4px}
</style></head><body>""",
    "<h1>宏观经济日报 · 历史索引 <span class='tag'>v5 · 12 模块</span></h1>",
    f"<p>最近 {len(metrics)} 天 · 仪表盘 / 数据</p>",
    "<table><tr><th>日期</th><th>文章数</th><th>情绪指数</th><th>政策倾向</th><th>注意力熵</th><th>主题词</th><th>操作</th></tr>"]

    for m in metrics:
        if m.sentiment_index >= 60:
            bull_cls = "bull-up"
        elif m.sentiment_index <= 40:
            bull_cls = "bull-down"
        else:
            bull_cls = "bull-flat"
        sentiment_label = f"{m.sentiment_index:.1f}"
        policy_label = {1.0: "扩张", 0.0: "中性", -1.0: "收紧"}.get(m.policy_stance_score, "中性")
        arrow = "↑" if m.sentiment_index >= 55 else ("↓" if m.sentiment_index <= 45 else "→")
        kws = repo.get_keywords(m.date, theme_only=True)[:3]
        kw_text = ", ".join(k.keyword for k in kws) if kws else "—"
        html.append(
            f"<tr><td><b>{m.date}</b></td>"
            f"<td>{m.article_count}</td>"
            f"<td><span class='bull {bull_cls}'>{sentiment_label} {arrow}</span></td>"
            f"<td>{policy_label}</td>"
            f"<td>{m.attention_entropy:.2f}</td>"
            f"<td>{kw_text}</td>"
            f"<td><a href='{m.date}.html'>仪表盘</a> · "
            f"<a href='../reports/{m.date}.md'>报告</a></td></tr>"
        )

    html.append("</table></body></html>")
    out = DASHBOARD_DIR / output_filename
    out.write_text("\n".join(html), encoding="utf-8")
    logger.info(f"dashboard index rendered: {out}")
    return out


if __name__ == "__main__":
    from datetime import date as _date
    today = _date.today().isoformat()
    p = render(today)
    print("dashboard:", p)
    idx = render_index()
    print("index:", idx)
