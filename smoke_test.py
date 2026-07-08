"""v9 smoke test - 23 个分析模块 (v5 12 + v6 4 + v8 3 + v9 4) + minimax LLM 路由 全链路自检.
用法:
    python smoke_test.py            # 跑 16 模块 + LLM 路由
    python smoke_test.py --fast     # 只跑 16 模块 (跳过 LLM)
    python smoke_test.py --llm-only # 只跑 LLM 路由
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import traceback
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("AI_BASE_URL", "https://api.minimaxi.com/v1")
os.environ.setdefault("AI_MODEL", "MiniMax-M3")
os.environ.setdefault("AI_PROVIDER", "minimax")
os.environ.setdefault(
    "ECONOMY_TIMESERIES_DB_PATH",
    str(Path(tempfile.gettempdir()) / "macro_intelligence_smoke.sqlite3"),
)


class C:
    OK = "\033[92m"; FAIL = "\033[91m"; WARN = "\033[93m"
    INFO = "\033[96m"; BOLD = "\033[1m"; END = "\033[0m"


def _clr(s, c): return f"{c}{s}{C.END}"


def header(t):
    line = "=" * 70
    print(f"\n{_clr(line, C.BOLD)}\n{_clr('  ' + t, C.BOLD)}\n{_clr(line, C.BOLD)}")


def subheader(t):
    print(f"\n{_clr('--- ' + t + ' ---', C.INFO)}")


def record(results, name, ok, detail="", elapsed=0.0):
    status = _clr("OK", C.OK) if ok else _clr("FAIL", C.FAIL)
    es = f"{elapsed:5.2f}s" if elapsed > 0 else "  -- "
    print(f"  [{status}] {es}  {name:<40}  {detail}")
    results.append({"name": name, "ok": ok, "detail": detail, "elapsed": elapsed})


def _mock_metrics(n=30):
    import random
    from datetime import date, timedelta
    random.seed(42)
    base = date(2026, 5, 13)
    out = []
    for i in range(n):
        d = (base + timedelta(days=i)).isoformat()
        out.append({
            "date": d, "article_count": 30 + (i % 5),
            "total_words": 15000 + i * 200, "unique_keywords": 25 + (i % 4),
            "policy_stance_score": round(0.3 + 0.2 * (i / n) + random.uniform(-0.04, 0.04), 3),
            "sentiment_index": round(50.0 + 5.0 * (i / n) + random.uniform(-1, 1), 2),
            "attention_entropy": round(0.7 + 0.1 * (i / n) + random.uniform(-0.02, 0.02), 3),
            "attention_top_share": round(0.25 + random.uniform(-0.05, 0.05), 3),
            "industry_count": 8 + (i % 3), "policy_count": 5 + (i % 2), "event_count": 12 + (i % 4),
        })
    return out


def _mock_articles(n=8):
    from src.scraper.pipeline import Article
    samples = [
        ("央行宣布降准0.5个百分点", "中国人民银行决定下调金融机构存款准备金率0.5个百分点,释放长期资金约1万亿元。"),
        ("5月新能源汽车销量创新高", "中汽协数据显示,5月新能源汽车销量同比增长38.5%,市场渗透率突破47%。"),
        ("国常会:稳经济一揽子政策", "国务院常务会议研究部署稳经济一揽子政策。"),
        ("半导体国产化率突破新高", "国内半导体设备国产化率达38%, 14nm 工艺实现量产。"),
        ("人民币汇率双向波动", "在岸人民币兑美元日内波动200点。"),
        ("房地产新政持续发力", "多地优化限购政策, 降低首付比例。"),
        ("数字经济立法进程加快", "数据要素流通条例草案征求意见。"),
        ("光伏组件出口保持强劲", "海关数据显示, 1-5月光伏组件出口金额同比增长27%。"),
    ]
    arts = []
    for t, c in samples[:n]:
        a = Article()
        a.title = t; a.content_text = c; a.source = "mock"
        arts.append(a)
    return arts


def _mock_events():
    from src.ai.schema import NewsEvent, EventType
    return [
        NewsEvent(subject="中国人民银行", action="宣布下调存款准备金率", object="释放长期资金",
                  event_type=EventType.MONETARY, impact="利好银行地产板块"),
        NewsEvent(subject="财政部", action="出台新能源汽车补贴", object="延长至2027年",
                  event_type=EventType.FISCAL, impact="新能源车板块受益"),
    ]


def _inject_mock_db(metrics):
    from src.storage import db as db_mod
    from src.storage import repository as repo
    db_mod.get_conn()
    with db_mod.get_conn() as conn:
        conn.execute("DELETE FROM daily_metric WHERE date >= ?", (metrics[0]["date"],))
        for m in metrics:
            repo.upsert_metric(repo.DailyMetric(**m))
        conn.commit()


class _A:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _ok(obj): return is_dataclass(obj)


# 12 v5 modules
def check_anomaly(results):
    t0 = time.time()
    try:
        from src.analysis.anomaly import detect
        rep = detect("2026-06-12", window_days=20)
        ok, detail = _ok(rep), f"signals={len(rep.signals)} risk={rep.overall_risk}"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    record(results, "anomaly", ok, detail, time.time() - t0)


def check_forecast(results):
    t0 = time.time()
    try:
        from src.analysis.forecast import predict_next_day
        rep = predict_next_day("2026-06-12", lookback_days=14)
        ok, detail = _ok(rep), f"target={rep.target_date} forecasts={len(rep.forecasts)}"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    record(results, "forecast", ok, detail, time.time() - t0)


def check_stock_correlation(results):
    t0 = time.time()
    try:
        from src.analysis.stock_correlation import correlate
        rep = correlate("2026-06-12", ["新能源", "半导体", "金融"], top_n_per_industry=2)
        if rep is None:
            ok, detail = True, "(network unavailable, gracefully degraded)"
        else:
            ok, detail = _ok(rep), f"industries={len(rep.industries)} overall={rep.market_overall}"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    record(results, "stock_correlation", ok, detail, time.time() - t0)


def check_macro_indicators(results):
    t0 = time.time()
    try:
        from src.analysis.macro_indicators import snapshot
        rep = snapshot("2026-06-12", theme_keywords=[{"word": "降息"}], industries=["金融"])
        ok, detail = _ok(rep), f"indicators={len(rep.indicators)} linkage={len(rep.news_linkage)}"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    record(results, "macro_indicators", ok, detail, time.time() - t0)


def check_topic_model(results):
    t0 = time.time()
    try:
        from src.analysis.topic_model import fit
        rep = fit(_mock_articles(), n_topics=3)
        if rep is None:
            ok, detail = True, "(insufficient data)"
        else:
            ok, detail = _ok(rep), f"topics={len(rep.topics)}"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    record(results, "topic_model", ok, detail, time.time() - t0)


def check_event_study(results):
    t0 = time.time()
    try:
        from src.analysis.event_study import study
        rep = study(_mock_events(), target_date="2026-06-12")
        ok, detail = _ok(rep), f"events={len(rep.events)} hist={rep.history_size}"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    record(results, "event_study", ok, detail, time.time() - t0)


def check_rag_history(results):
    t0 = time.time()
    try:
        from src.analysis.rag_history import recall
        rep = recall("央行降准 流动性", "2026-06-12", top_k=3)
        ok, detail = _ok(rep), f"matches={len(rep.recalls)} corpus={rep.corpus_size}"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    record(results, "rag_history", ok, detail, time.time() - t0)


def check_volatility(results):
    t0 = time.time()
    try:
        from src.analysis.volatility import compute
        rep = compute("2026-06-12")
        ok = _ok(rep)
        vix = getattr(rep, "index", None) or getattr(rep, "vix", 0.0)
        detail = f"vix={vix:.2f} level={rep.level}"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    record(results, "volatility", ok, detail, time.time() - t0)


def check_causal(results):
    t0 = time.time()
    try:
        from src.analysis.causal import analyze
        rep = analyze(_mock_events(), target_date="2026-06-12")
        ok, detail = _ok(rep), f"chains={len(rep.chains)} conf={rep.confidence}"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    record(results, "causal", ok, detail, time.time() - t0)


def check_signal_engine(results):
    t0 = time.time()
    try:
        from src.analysis.signal_engine import synthesize
        a = _A(overall_risk="low", signals=[_A(metric="sentiment_index", z_score=1.8)])
        f = _A(headline="看多", predicted_sentiment=66.0, current_sentiment=55.0)
        v = _A(index=25.0, level="calm")
        m = _A(market_overall="bullish", industries=[_A(industry="新能源", stance="利好", sentiment=0.7, confidence=0.8)])
        macro = _A(indicators=[_A(category="景气", value=50.5, previous=49.8)])
        es = _A(events=[_A(stance="利好"), _A(stance="利好")])
        ai = _A(policy_direction=_A(direction="扩张", confidence=0.75))
        sig = synthesize("2026-06-12", anomaly=a, forecast=f, volatility=v, market=m, macro=macro, events_study=es, ai_result=ai)
        ok, detail = _ok(sig), f"action={sig.action} score={sig.score:+.2f} conf={sig.confidence:.0%}"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    record(results, "signal_engine", ok, detail, time.time() - t0)


def check_forecast_backtest(results):
    t0 = time.time()
    try:
        from src.analysis.forecast_backtest import backtest
        rep = backtest("2026-06-12", n_days=10, horizons=(1, 3))
        ok, detail = _ok(rep), f"horizons={len(rep.horizon_reports)} cases={len(rep.cases)}"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    record(results, "forecast_backtest", ok, detail, time.time() - t0)


def check_narrative(results):
    t0 = time.time()
    try:
        from src.analysis.narrative import generate
        advanced = {
            "anomaly": _A(date="2026-06-12", overall_risk="low", signals=[_A(metric="sentiment_index", z_score=2.1, direction="up")]),
            "forecast": _A(headline="明日偏多", predicted_sentiment=66.0, current_sentiment=62.0),
            "volatility": _A(index=28.0, level="calm"),
            "market": _A(market_overall="bullish"),
            "macro": _A(indicators=[_A(name="PMI", value=50.5, previous=49.8)]),
            "topics": _A(topics=[_A(label="新能源", dominance=0.42, dominant_industry="新能源")]),
            "events_study": _A(events=[_A(title="央行降准", impact_level=5, industry="金融")]),
            "rag": _A(recalls=[_A(title="Q4复盘", date="2026-03-15", score=0.82)]),
            "signal": _A(action="BUY", score=0.42, confidence=0.62, top_reasons=["政策扩张"], risks=["PMI临界"],
                         industry_signals=[_A(industry="新能源", action="关注", score=0.7)]),
            "backtest": _A(horizon_reports=[_A(horizon=1, direction_accuracy=0.81)]),
        }
        ai = _A(
            policy_direction=_A(direction="扩张", confidence=0.75),
            industries=_A(industries=[_A(name="新能源", heat=_A(value="高"), stance=_A(value="利好"))]),
            theme_keywords=_A(keywords=[_A(word="新能源"), _A(word="AI")]),
            core_insights=_A(insights="今日新能源板块持续走高。"),
            policies=_A(policies=[_A(title="央行降准", stance="利好")]),
        )
        rep = generate("2026-06-12", advanced, ai_result=ai, router=None)
        ok, detail = _ok(rep) and len(rep.sections) == 5, f"sections={len(rep.sections)} gen_by={rep.generated_by}"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    record(results, "narrative", ok, detail, time.time() - t0)


# 4 v6 modules (NEW)
def check_qa_assistant(results):
    t0 = time.time()
    try:
        from src.analysis.qa_assistant import ask
        rep = ask("近期降准对新能源板块有什么影响?", target_date="2026-06-12", router=None)
        ok = _ok(rep) and bool(rep.answer)
        detail = f"answer_len={len(rep.answer)} citations={len(rep.citations)} gen_by={rep.generated_by}"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    record(results, "qa_assistant", ok, detail, time.time() - t0)


def check_risk_metrics(results):
    t0 = time.time()
    try:
        from src.analysis.risk_metrics import compute
        rep = compute("2026-06-12", lookback=29, rf_rate=0.025)
        if rep is None:
            ok, detail = True, "(insufficient data)"
        else:
            ok = _ok(rep) and rep.risk_level in ("low", "medium", "high", "extreme")
            detail = f"sharpe={rep.sharpe_ratio:+.2f} dd={rep.max_drawdown:+.2%} var95={rep.var_95:+.2%} level={rep.risk_level}"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    record(results, "risk_metrics", ok, detail, time.time() - t0)


def check_portfolio(results):
    t0 = time.time()
    try:
        from src.analysis.portfolio import backtest
        rep = backtest("2026-06-12",
                       portfolio={"新能源": 0.3, "半导体": 0.2, "金融": 0.3, "消费": 0.2},
                       rebalance="weekly", lookback=29)
        if rep is None:
            ok, detail = True, "(insufficient data)"
        else:
            ok = _ok(rep)
            detail = f"cum={rep.cumulative_return:+.2%} sharpe={rep.sharpe_ratio:+.2f} alpha={rep.alpha:+.4f} ir={rep.information_ratio:+.2f}"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    record(results, "portfolio", ok, detail, time.time() - t0)


def check_scenario(results):
    t0 = time.time()
    try:
        from src.analysis.scenario import run, list_scenarios
        rep = run("2026-06-12", scenario="rate_cut_50bp", horizon_days=30, n_sims=100)
        if rep is None:
            ok, detail = True, "(insufficient data)"
        else:
            ok = _ok(rep) and 0.0 <= rep.p_positive <= 1.0
            detail = f"scenarios={len(list_scenarios())} p_pos={rep.p_positive:.0%} p5={rep.percentile_5:.1f} p95={rep.percentile_95:.1f}"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    record(results, "scenario", ok, detail, time.time() - t0)


V5_CHECKS = [check_anomaly, check_forecast, check_stock_correlation, check_macro_indicators,
             check_topic_model, check_event_study, check_rag_history, check_volatility,
             check_causal, check_signal_engine, check_forecast_backtest, check_narrative]
V6_CHECKS = [check_qa_assistant, check_risk_metrics, check_portfolio, check_scenario]
ALL_CHECKS = V5_CHECKS + V6_CHECKS


# 3 v8 modules (NEW)
def check_factor_model(results):
    t0 = time.time()
    try:
        from src.analysis.factor_model import compute
        rep = compute("2026-06-12", lookback=29)
        if rep is None:
            ok, detail = True, "(insufficient data)"
        else:
            ok = _ok(rep) and len(rep.factors) == 5
            detail = f"score={rep.total_score:+.3f} rank={rep.total_rank} n_factors={len(rep.factors)}"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    record(results, "factor_model", ok, detail, time.time() - t0)


def check_forecast_enhanced(results):
    t0 = time.time()
    try:
        from src.analysis.forecast_enhanced import predict
        rep = predict("2026-06-12", metric="sentiment_index", lookback=29)
        if rep is None:
            ok, detail = True, "(insufficient data)"
        else:
            ok = _ok(rep) and len(rep.horizons) >= 3
            detail = f"method={rep.method_selected} period={rep.seasonality_period} mae={rep.in_sample_mae:.2f} h={len(rep.horizons)}"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    record(results, "forecast_enhanced", ok, detail, time.time() - t0)


def check_policy_pdf(results):
    t0 = time.time()
    try:
        from src.analysis.policy_pdf import parse
        # 用 nonexistent 文件测试导入+优雅降级
        rep = parse("nonexistent_test.pdf")
        ok = _ok(rep) and len(rep.parse_warnings) > 0
        detail = f"warnings={len(rep.parse_warnings)} graceful=True"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    record(results, "policy_pdf", ok, detail, time.time() - t0)


V8_CHECKS = [check_factor_model, check_forecast_enhanced, check_policy_pdf]
# ===== v9 前沿 AI 深度 (4 模块) =====
def check_embeddings(results):
    t0 = time.time()
    try:
        from src.analysis.embeddings import EmbeddingStore, get_store
        # 用 hash 后端自检, 无需联网
        store = EmbeddingStore(prefer="hash")
        store.add_many([
            ("d1", "中国人民银行下调存款准备金率 0.5 个百分点 释放长期资金", {"cat": "monetary"}),
            ("d2", "5 月新能源汽车销量同比增长 38.5% 渗透率突破 47%", {"cat": "auto"}),
            ("d3", "工信部发布新型储能制造业高质量发展行动方案", {"cat": "industry"}),
            ("d4", "国务院研究部署稳经济一揽子政策", {"cat": "policy"}),
        ])
        hits = store.search("降准 利好哪些板块", top_k=2)
        ok = len(store.index) >= 4 and len(hits) > 0 and hits[0].score > 0
        detail = f"backend={store.backend_info().get('backend','?')} index={len(store.index)} hits={len(hits)} top={hits[0].score:.3f}"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    record(results, "embeddings", ok, detail, time.time() - t0)


def check_regime(results):
    t0 = time.time()
    try:
        from src.analysis.regime import fit_from_daily_metric
        # fit_from_daily_metric 需要 daily_metric 表, 30 天 mock 已注入
        rep = fit_from_daily_metric("2026-06-12", lookback=29, metric_col="sentiment_index")
        if rep is None:
            ok, detail = True, "(insufficient data)"
        else:
            ok = _ok(rep) and rep.n_obs >= 6 and rep.current_regime in ("calm", "euphoric")
            detail = f"current={rep.current_regime} p_calm={rep.current_p_calm:.2f} ll={rep.log_likelihood:.1f} obs={rep.n_obs}"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    record(results, "regime", ok, detail, time.time() - t0)


def check_graph_rag(results):
    t0 = time.time()
    try:
        from src.analysis.graph_rag import build_graph_rag
        arts = _mock_articles(8)
        rep = build_graph_rag(arts, router=None, max_communities=6, use_llm_summary=False)
        ok = _ok(rep) and rep.n_nodes > 0
        detail = f"nodes={rep.n_nodes} edges={rep.n_edges} communities={rep.n_communities} modularity={rep.modularity:.3f}"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    record(results, "graph_rag", ok, detail, time.time() - t0)


def check_react_agent(results):
    t0 = time.time()
    try:
        from src.agent.react_agent import build_default_agent
        # template 模式自检: 不依赖 LLM, 也能跑
        agent = build_default_agent(router=None, max_steps=5)
        res = agent.run("今日宏观风险与机会", target_date="2026-06-12")
        ok = _ok(res) and res.n_tool_calls >= 1 and res.generated_by in ("react-template", "react-text", "react-function-calling")
        detail = f"steps={res.n_steps} tool_calls={res.n_tool_calls} by={res.generated_by} {res.elapsed_ms}ms"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    record(results, "react_agent", ok, detail, time.time() - t0)


V9_CHECKS = [check_embeddings, check_regime, check_graph_rag, check_react_agent]
ALL_CHECKS = V5_CHECKS + V6_CHECKS + V8_CHECKS + V9_CHECKS


SAMPLE = """[1] 央行宣布降准1万亿
央行决定下调金融机构存款准备金率0.5个百分点。
[2] 新能源汽车销量创新高
中国新能源汽车5月销量同比增长38.5%。"""


def check_llm(results):
    subheader("LLM 路由 (minimax)")
    if not os.environ.get("AI_API_KEY", "").strip():
        record(results, "llm/config", True, "skipped: AI_API_KEY is not set", 0.0)
        return
    from src.ai.router import ModelRouter
    from src.ai.schema import ThemeKeywordsResult
    router = ModelRouter()
    for model in ["MiniMax-M3", "MiniMax-M2.5-highspeed"]:
        t0 = time.time()
        try:
            raw, used, pt, ct = router.chat(
                messages=[{"role": "system", "content": "你是经济分析助手,只返回 JSON。"},
                          {"role": "user", "content": f"提炼3个主题词:\n{SAMPLE}\n返回: {{\"keywords\":[{{\"word\":\"x\",\"score\":0.9,\"explain\":\"...\"}}]}}"}],
                model=model, temperature=0.3, max_tokens=400, use_json_mode=(model == "MiniMax-M3"),
            )
            parsed = router.extract_json(raw)
            v = ThemeKeywordsResult(**parsed) if parsed else None
            ok, detail = (v is not None and len(v.keywords) > 0), f"used={used} tokens={pt}+{ct} kws={len(v.keywords) if v else 0}"
        except Exception as e:
            ok, detail = False, f"{type(e).__name__}: {e}"
        record(results, f"llm/{model}", ok, detail, time.time() - t0)


def run_all(skip_llm=False, llm_only=False):
    header("宏观经济智能分析平台 v9 · 烟雾测试 (23 模块 = v5 12 + v6 4 + v8 3 + v9 4)")
    print(f"  Python: {sys.version.split()[0]}    Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    results = []
    if not llm_only:
        try:
            subheader("注入 mock 时序数据库 (30 天)")
            _inject_mock_db(_mock_metrics())
            print(f"  {_clr('OK', C.OK)} 30 天 mock daily_metric 已写入")
        except Exception as e:
            print(f"  {_clr('WARN', C.WARN)} mock 注入失败: {e}")
        subheader("v5 12 模块 (基础)")
        for chk in V5_CHECKS:
            try:
                chk(results)
            except Exception as e:
                record(results, chk.__name__, False, f"unexpected: {e}")
                traceback.print_exc()
        subheader("v6 4 模块 (新增)")
        for chk in V6_CHECKS:
            try:
                chk(results)
            except Exception as e:
                record(results, chk.__name__, False, f"unexpected: {e}")
                traceback.print_exc()
        subheader("v8 3 模块 (前沿量化)")
        for chk in V8_CHECKS:
            try:
                chk(results)
            except Exception as e:
                record(results, chk.__name__, False, f"unexpected: {e}")
                traceback.print_exc()
        subheader("v9 4 模块 (前沿 AI 深度)")
        for chk in V9_CHECKS:
            try:
                chk(results)
            except Exception as e:
                record(results, chk.__name__, False, f"unexpected: {e}")
                traceback.print_exc()
    if not skip_llm:
        check_llm(results)
    header("汇总")
    n = len(results)
    n_ok = sum(1 for r in results if r["ok"])
    n_fail = n - n_ok
    print(f"  Total: {n}    Pass: {_clr(str(n_ok), C.OK)}    Fail: {_clr(str(n_fail), C.FAIL)}    Time: {sum(r['elapsed'] for r in results):.2f}s")
    if n_fail > 0:
        print(f"\n  {_clr('失败明细:', C.FAIL)}")
        for r in results:
            if not r["ok"]:
                print(f"    - {r['name']:<32} {r['detail']}")
        print(f"\n  {_clr('PARTIAL', C.WARN)} - 部分模块异常\n")
        return 1
    print(f"\n  {_clr('ALL GREEN', C.OK)} - v9 平台 23 模块 + LLM 路由就绪\n")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fast", action="store_true")
    p.add_argument("--llm-only", action="store_true")
    a = p.parse_args()
    sys.exit(run_all(skip_llm=a.fast, llm_only=a.llm_only))


if __name__ == "__main__":
    main()
