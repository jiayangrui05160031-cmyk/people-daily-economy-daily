"""main.py - 宏观经济智能分析平台 CLI 入口 (v5)

串联: 多源爬取(智能反爬 + 质量过滤 + 语义去重) -> NLP -> AI(10 维+自评+反思)
    -> 时序库 -> 量化指标 -> 跨日对比 -> 趋势涌现 -> 知识图谱 -> 产业-A股
    -> 反思循环 -> 交互式仪表盘 -> Markdown 报告 -> 历史索引。

用法:
    python main.py                                 # 全流程,今天日期
    python main.py --yesterday                     # 抓昨天(兼容旧语义)
    python main.py --date 2026-06-12               # 指定日期
    python main.py --days 7                       # 回溯天数(默认 2)
    python main.py --window-days 14               # 仪表盘/对比窗口(默认 14)
    python main.py --skip-scrape                  # 跳过爬取,用已有 JSON
    python main.py --skip-ai                      # 跳过 AI,纯 NLP 报告
    python main.py --reflect                      # 启用 AI 反思循环(默认关闭)
    python main.py --no-dashboard                 # 关闭仪表盘生成
    python main.py --no-kg                        # 关闭知识图谱生成
    python main.py --no-comparison                # 关闭跨日对比(对比需 2+ 天数据)
    python main.py --enable-xinhua                # 启用新华网源(.env 需配)
    python main.py --enable-pbc                   # 启用央行源(.env 需配)
    python main.py --debug                        # 调试日志
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai.analyzer import analyze_all
from src.ai.reflection import reflect_on_report, should_reflect
from src.ai.router import get_default_router
from src.ai.schema import AnalysisReport
from src.dashboard.builder import render as render_dashboard, render_index as render_dashboard_index
from src.kg.graph import build as build_kg, summarize as summarize_kg
from src.kg.viz import render as render_kg_png
from src.nlp.stats import analyze as nlp_analyze
from src.report.archiver import archive_raw, load_raw
from src.report.renderer import render as render_report
from src.scraper.multi_source import fetch_multi_source_articles
from src.scraper.pipeline import Article
from src.storage import repository as repo
from src.utils.date_utils import resolve_target_date, yesterday
from src.utils.logger import get_logger

logger = get_logger("main")


def banner(text, width=60):
    print()
    print("=" * width)
    print("  " + text)
    print("=" * width)


def ok(msg):
    print("[OK] " + msg)


def warn(msg):
    print("[!] " + msg)


def step1_scrape(target_date, lookback_days=2, enable_xinhua=False, enable_pbc=False, enable_dedup=True, router=None):
    """多源爬取 (智能反爬 + 质量过滤 + 语义去重)。"""
    banner("STEP 1/7 - 多源爬取 (智能反爬 + 质量过滤 + 语义去重)")
    t0 = time.time()
    articles = fetch_multi_source_articles(
        target_date=target_date, lookback_days=lookback_days,
        enable_xinhua=enable_xinhua, enable_gov_pbc=enable_pbc,
        enable_dedup=enable_dedup, router=router,
    )
    src_count = {}
    for a in articles:
        s = a.source or "?"
        src_count[s] = src_count.get(s, 0) + 1
    src_info = ", ".join(f"{k}={v}" for k, v in src_count.items())
    ok(f"抓取完成: {len(articles)} 篇 ({src_info}), 用时 {time.time() - t0:.1f}s")
    return articles


def step2_nlp(articles):
    """NLP 分析 (分词/关键词/词频/产业匹配)。"""
    banner("STEP 2/7 - NLP 分析")
    t0 = time.time()
    stats = nlp_analyze(articles)
    ok(f"NLP 完成: {stats.total_words} 词, {len(stats.keywords)} 关键词, "
       f"{len(stats.industry_hits)} 产业命中, 用时 {time.time() - t0:.1f}s")
    if stats.keywords:
        print("    Top 5 关键词: " + ", ".join(w for w, _ in stats.keywords[:5]))
    if stats.industry_hits:
        print("    命中产业: " + ", ".join(stats.industry_hits.keys()))
    return stats


def step3_ai(articles, nlp_stats, target_date, router=None, reflect=False, date="", prior_results=None):
    """AI 分析 (10 维 + 自评 + 可选反思)。"""
    banner("STEP 3/7 - AI 分析 (10 维任务 + 自评)")
    t0 = time.time()
    result = analyze_all(articles, nlp_stats=nlp_stats, date=date, router=router)
    _nkw = len(result.theme_keywords.keywords) if result.theme_keywords else 0
    _npo = len(result.policies.policies) if result.policies else 0
    _nind = len(result.industries.industries) if result.industries else 0
    _nout = len(result.outlooks.outlooks) if result.outlooks else 0
    _nev = len(result.events.events) if result.events else 0
    _nsen = len(result.sentiment.items) if result.sentiment else 0
    _ncl = len(result.cross_links.links) if result.cross_links else 0
    ok(f"AI 完成: 主题词 {_nkw}, 政策 {_npo}, 产业 {_nind}, 判断 {_nout}, 事件 {_nev}, 情感 {_nsen}, 聚类 {_ncl}, 用时 {time.time() - t0:.1f}s")
    if result.policy_direction:
        print(f"    政策风向: {result.policy_direction.direction or '?'} (置信度 {result.policy_direction.confidence:.2f})")
    if result.self_eval:
        print(f"    AI 自评: 一致性 {result.self_eval.consistency:.2f} / 依据 {result.self_eval.groundedness:.2f} / 完整 {result.self_eval.completeness:.2f}")
    if reflect and router and should_reflect(result.self_eval):
        warn("self_eval 低于阈值, 触发反思循环")
        try:
            result = reflect_on_report(router, result, articles, prior_results or {}, date=target_date)
            ok("反思完成")
        except Exception as e:
            warn(f"反思失败: {e}")
    return result


def step4_quant_and_compare(articles, nlp_stats, ai_result, target_date, no_comparison=False, window_days=14):
    """量化指标 + 跨日对比 + 趋势涌现/衰退 (写到时序库)。"""
    from src.analysis.metrics import compute as compute_metrics
    from src.trend.comparison import compare as compare_trend
    from src.trend.emerging import detect as detect_emerging
    from src.utils.date_utils import parse_date, previous_business_day, same_weekday_last_week
    banner("STEP 4/7 - 量化指标 + 跨日对比 + 趋势 (写入时序库)")
    t0 = time.time()
    metrics = compute_metrics(
        nlp_stats.word_freq, ai_result,
        list(nlp_stats.industry_hits.keys()), 12,
    )
    cmp_res = None
    if not no_comparison:
        try:
            prev_dt = previous_business_day(parse_date(target_date))
            week_dt = same_weekday_last_week(parse_date(target_date))
            cmp_res = compare_trend(target_date, prev_date=prev_dt.isoformat(), week_date=week_dt.isoformat())
        except Exception as e:
            warn(f"跨日对比失败: {e}")
    trend_report = None
    try:
        trend_report = detect_emerging(target_date, recent_days=3, prev_days=7)
    except Exception as e:
        warn(f"趋势检测失败: {e}")
    ok(f"指标计算: 情绪={metrics.sentiment_index:.1f} 政策={metrics.policy_stance_score:.2f} 熵={metrics.attention_entropy:.3f} 广度={metrics.industry_breadth:.3f}")
    if cmp_res:
        ok(f"对比: 文章{cmp_res.article_count_diff.direction}{cmp_res.article_count_diff.pct_diff:.1f}%, 重叠率{cmp_res.keyword_overlap_pct:.1f}%")
    if trend_report:
        ok(f"趋势: 涌现{len(trend_report.emerging)} 衰退{len(trend_report.declining)} 持续{len(trend_report.persistent)}")
    print(f"    [时序] 用时 {time.time() - t0:.1f}s")
    return metrics, cmp_res, trend_report


def step4b_advanced(target_date, nlp_stats, ai_result, articles=None, router=None):
    """前沿分析: 9 个模块。"""
    from src.analysis.anomaly import detect as detect_anomaly
    from src.analysis.forecast import predict_next_day
    from src.analysis.stock_correlation import correlate as correlate_market
    from src.analysis.macro_indicators import snapshot as macro_snapshot
    from src.analysis.topic_model import fit_with_evolution as topic_fit
    from src.analysis.event_study import study as event_study_fn
    from src.analysis.rag_history import recall as rag_recall
    from src.analysis.volatility import compute as volatility_compute
    from src.analysis.causal import analyze as causal_analyze
    banner("STEP 4b/7 - 前沿分析 (12 模块)")
    t0 = time.time()
    industries_hit = list(nlp_stats.industry_hits.keys())
    theme_kws = [{"word": k.word, "score": k.score} for k in ai_result.theme_keywords.keywords]
    out = {}
    try:
        a = detect_anomaly(target_date, window_days=30)
        out["anomaly"] = a
        if a.signals: ok(f"异常: {len(a.signals)} 项, 风险 {a.overall_risk}")
        else: ok(f"异常: {a.summary}")
    except Exception as e: warn(f"异常失败: {e}")
    try:
        f = predict_next_day(target_date, lookback_days=14)
        out["forecast"] = f
        ok(f"预测: {f.headline}")
    except Exception as e: warn(f"预测失败: {e}")
    try:
        m = correlate_market(target_date, industries_hit[:6], top_n_per_industry=3)
        if m: out["market"] = m; ok(f"市场: {m.market_overall}, {len(m.industries)} 产业")
        else: warn("市场: 无数据")
    except Exception as e: warn(f"市场失败: {e}")
    try:
        mc = macro_snapshot(target_date, theme_keywords=theme_kws, industries=industries_hit)
        if mc: out["macro"] = mc; ok(f"宏观: {len(mc.indicators)} 项")
    except Exception as e: warn(f"宏观失败: {e}")
    try:
        if articles:
            tp = topic_fit(articles, target_date, history_days=3)
            if tp: out["topics"] = tp; ok(f"主题: {tp.n_topics} 主题, 熵 {tp.doc_topic_entropy:.3f}")
    except Exception as e: warn(f"主题失败: {e}")
    try:
        es = event_study_fn(ai_result.events.events, target_date=target_date)
        out["events_study"] = es
        ok(f"事件: {len(es.events)} 个, 历史 {es.history_size} 条")
    except Exception as e: warn(f"事件失败: {e}")
    try:
        parts = [k.get("word", "") for k in theme_kws[:8]]
        if getattr(ai_result, "core_insights", None) and ai_result.core_insights.insights:
            parts.append(ai_result.core_insights.insights)
        for ind in ai_result.industries.industries[:3]:
            parts.append(ind.name + ": " + ind.summary)
        query = " ".join(parts)
        rag = rag_recall(query, target_date, top_k=5)
        if rag and rag.recalls: out["rag"] = rag; ok(f"RAG: {len(rag.recalls)} 条, top {rag.top_score:.3f}")
        else: warn(f"RAG: {rag.summary if rag else chr(26032) + chr(26029)}")
    except Exception as e: warn(f"RAG 失败: {e}")
    try:
        vol = volatility_compute(target_date, window_days=14)
        out["volatility"] = vol
        ok(f"波动率: {vol.index:.1f} ({vol.level})")
    except Exception as e: warn(f"波动率失败: {e}")
    try:
        ca = causal_analyze(ai_result.events.events, target_date=target_date)
        out["causal"] = ca
        ok(f"因果: {len(ca.chains)} 链, 置信 {ca.confidence:.2f}")
    except Exception as e: warn(f"因果失败: {e}")
    # ===== v5 新增三大模块 =====
    try:
        from src.analysis.signal_engine import synthesize as signal_synth
        sig = signal_synth(
            target_date,
            anomaly=out.get("anomaly"),
            forecast=out.get("forecast"),
            volatility=out.get("volatility"),
            market=out.get("market"),
            macro=out.get("macro"),
            events_study=out.get("events_study"),
            topics=out.get("topics"),
            ai_result=ai_result,
        )
        out["signal"] = sig
        ok(f"决策: {sig.action} (分 {sig.score:+.2f}, 置信 {sig.confidence:.0%})")
    except Exception as e: warn(f"决策引擎失败: {e}")
    try:
        from src.analysis.forecast_backtest import backtest as forecast_backtest_fn
        bt = forecast_backtest_fn(target_date, n_days=20, horizons=(1, 3, 7))
        out["backtest"] = bt
        if bt.horizon_reports:
            best = max(bt.horizon_reports, key=lambda r: r.direction_accuracy)
            ok(f"回测: horizon={best.horizon}天 方向准确率 {best.direction_accuracy:.0%}")
        else:
            ok(f"回测: {bt.summary}")
    except Exception as e: warn(f"回测失败: {e}")
    try:
        from src.analysis.narrative import generate as narrative_generate
        nar = narrative_generate(target_date, out, ai_result=ai_result, router=router)
        out["narrative"] = nar
        ok(f"简报: {len(nar.sections)} 段 ({nar.generated_by}, {nar.word_count} 字)")
    except Exception as e: warn(f"简报失败: {e}")
    # ===== v6 新增四大模块 =====
    try:
        from src.analysis.risk_metrics import compute as risk_compute
        rm = risk_compute(target_date, lookback=90, rf_rate=0.025)
        if rm:
            out["risk"] = rm
            ok(f"风险: 夏普 {rm.sharpe_ratio:+.2f} / 回撤 {rm.max_drawdown:+.2%} / VaR95 {rm.var_95:+.2%} / 等级 {rm.risk_level}")
        else:
            warn("风险: 数据不足")
    except Exception as e: warn(f"风险失败: {e}")
    try:
        from src.analysis.portfolio import backtest as portfolio_backtest
        from src.config import INDUSTRY_TO_STOCKS
        # 只取有 A 股映射的行业, 避免 KeyError
        available = [ind for ind in industries_hit if ind in INDUSTRY_TO_STOCKS]
        top_inds = (available or ["新能源", "半导体", "金融", "消费"])[:4]
        weights = {ind: 1.0 / len(top_inds) for ind in top_inds}
        pf = portfolio_backtest(target_date, portfolio=weights, rebalance="weekly", lookback=90)
        if pf:
            out["portfolio"] = pf
            ok(f"组合: 累计 {pf.cumulative_return:+.2%} / 夏普 {pf.sharpe_ratio:+.2f} / alpha {pf.alpha:+.4f}")
        else:
            warn("组合: 数据不足")
    except Exception as e: warn(f"组合失败: {e}")
    try:
        from src.analysis.scenario import run as scenario_run
        sc = scenario_run(target_date, scenario="rate_cut_50bp", horizon_days=30, n_sims=500)
        if sc:
            out["scenario"] = sc
            ok(f"情景: [{sc.scenario_name}] P(正向) {sc.p_positive:.0%} / 5~95分位 [{sc.percentile_5:.1f}, {sc.percentile_95:.1f}]")
        else:
            warn("情景: 数据不足")
    except Exception as e: warn(f"情景失败: {e}")
    try:
        from src.analysis.qa_assistant import ask as qa_ask
        qa_q = "今日宏观主要风险与机会是什么?"
        qa = qa_ask(qa_q, target_date=target_date, router=router, top_k=5)
        if qa and qa.answer:
            out["qa"] = qa
            ok(f"问答: '{qa_q}' -> {len(qa.citations)} 引用 ({qa.generated_by})")
    except Exception as e: warn(f"问答失败: {e}")
    # ===== v7 前沿: 多智能体 2 轮辩论 + LLM Judge =====
    try:
        from src.analysis.council_debate import debate
        # 当 router 不可用时 (skip_ai) 走模板模式
        use_llm = router is not None and getattr(router, "api_key", None)
        rep = debate("当前宏观风险与机会", target_date=target_date, router=router,
                     rounds=2, use_llm=use_llm, use_llm_judge=use_llm)
        if rep:
            out["council_debate"] = rep
            final = rep.final
            judge_tag = " LLM" if rep.used_llm_judge else " 模板"
            ok("顾问团辩论: " + str(len(rep.rounds)) + " 轮 / LLM " + str(rep.used_llm_rounds) + " 次" + judge_tag
               + " / 最终 " + final.final_stance
               + " (置信 " + format(final.final_confidence, ".0%") + ")")
    except Exception as e: warn("辩论失败: " + str(e))
    # ===== v7 前沿: SHAP 风格特征贡献度解释 =====
    try:
        from src.api.shap_explain import explain_decision
        industries_hit = list(nlp_stats.industry_hits.keys())
        theme_kws = [{"word": k.word, "score": k.score} for k in ai_result.theme_keywords.keywords]
        sh = explain_decision(target_date, ai_result=ai_result,
                              industries_hit=industries_hit, theme_kws=theme_kws)
        if sh:
            out["shap"] = sh
            ok("SHAP 解释: 决策 " + sh.final_action + " (分 " + format(sh.final_score, "+.2f")
               + "), 主导信号 " + (sh.top_positive or sh.top_negative or "N/A"))
    except Exception as e: warn("SHAP 失败: " + str(e))
    # ===== v8 前沿: 量化多因子 + 时序预测增强 + 政策 PDF 解析 =====
    try:
        from src.analysis.factor_model import compute as factor_compute
        fm = factor_compute(target_date, lookback=90)
        if fm:
            out["factor_model"] = fm
            ok(f"多因子: 评分 {fm.total_score:+.3f} ({fm.total_rank}) 主驱 {fm.factors[0].factor}")
    except Exception as e: warn("多因子失败: " + str(e))
    try:
        from src.analysis.forecast_enhanced import predict as forecast_v8
        # 默认 sentiment_index, 4 个 horizon
        fc_v8 = forecast_v8(target_date, metric="sentiment_index", lookback=90)
        if fc_v8:
            out["forecast_enhanced"] = fc_v8
            ok(f"时序预测 v8: MAE {fc_v8.in_sample_mae:.2f} MAPE {fc_v8.in_sample_mape:.1f}% 周期 {fc_v8.seasonality_period}")
    except Exception as e: warn("时序预测 v8 失败: " + str(e))
    try:
        from src.analysis.policy_pdf import parse as policy_pdf_parse
        # 尝试从 data/raw 找最近的 PDF
        import os
        from pathlib import Path as _P
        pdf_dir = _P("data/policy_pdfs")
        if pdf_dir.exists():
            pdfs = sorted(pdf_dir.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
            if pdfs:
                doc = policy_pdf_parse(str(pdfs[0]), use_llm=False, router=router if router else None)
                out["policy_pdf"] = doc
                ok(f"政策 PDF: {pdfs[0].name} | {doc.title[:30]} | 立场 {doc.overall_stance}")
    except Exception as e: warn("政策 PDF 失败: " + str(e))

    # ===== v9 前沿: Embedding 向量库 + Hamilton 区制 + GraphRAG + ReAct Agent =====
    try:
        from src.analysis.embeddings import get_store, semantic_search
        from src.analysis.regime import fit_from_daily_metric, regime_to_signal
        from src.analysis.graph_rag import build_graph_rag, ask_global
        from src.agent.react_agent import build_default_agent, run_agent

        banner("STEP 4c/7 - v9 前沿 (Embedding RAG + Hamilton 区制 + GraphRAG + ReAct Agent)")
        t0 = time.time()
        v9 = {}

        # 1) Embedding 向量库 + 语义检索
        try:
            if articles:
                # 把今日文章入库 (title + content 前 200 字, 带 date 元数据)
                from src.analysis.embeddings import EmbeddingStore
                # 默认走 hash 后端, 避免本地模型下载/HuggingFace 限速
                store = EmbeddingStore(prefer="hash")
                docs = []
                for a in articles[:30]:
                    title = getattr(a, "title", "") or ""
                    content = getattr(a, "content_text", "") or ""
                    if not content and hasattr(a, "body"):
                        content = a.body or ""
                    snippet = (title + " " + content[:200]).strip()
                    if snippet:
                        docs.append((getattr(a, "url", "") or title,
                                     snippet,
                                     {"date": target_date, "src": getattr(a, "source", "?")}))
                if docs:
                    store.add_many(docs)
                # 检索 1 个代表性问题
                query = "今日宏观主要风险与机会"
                hits = store.search(query, top_k=3)
                v9["embeddings"] = {
                    "backend": store.backend_info().get("backend", "?"),
                    "index_size": len(store.index),
                    "top_hit_score": float(hits[0].score) if hits else 0.0,
                    "query": query,
                    "n_hits": len(hits),
                    "hits": [{"doc_id": h.id, "score": float(h.score),
                              "snippet": (h.text or "")[:80]} for h in hits[:3]],
                }
                ok(f"Embeddings: backend={v9['embeddings']['backend']}, "
                   f"index={v9['embeddings']['index_size']} 篇, "
                   f"top score={v9['embeddings']['top_hit_score']:.3f}")
        except Exception as e: warn(f"Embeddings 失败: {e}")

        # 2) Hamilton 区制检测
        try:
            reg = fit_from_daily_metric(target_date, lookback=60, metric_col="sentiment_index")
            if reg:
                sig = regime_to_signal(reg)
                v9["regime"] = {
                    **reg.as_dict(),
                    "signal_score": sig.get("score", 0.0),
                    "signal_direction": sig.get("direction", "neutral"),
                    "signal_reason": sig.get("reason", ""),
                }
                ok(f"Hamilton 区制: current={reg.current_regime} "
                   f"p_calm={reg.current_p_calm:.2f} (converged={reg.converged}, ll={reg.log_likelihood:.1f})")
            else: warn("Hamilton 区制: 数据不足")
        except Exception as e: warn(f"Hamilton 区制失败: {e}")

        # 3) GraphRAG 知识图谱社区 + 全局问答
        try:
            if articles and router is not None and getattr(router, "api_key", None):
                use_llm = True
            else:
                use_llm = False
            gr = build_graph_rag(articles, router=router,
                                 max_communities=6, use_llm_summary=use_llm)
            if gr.n_communities > 0:
                ga = ask_global("今日宏观核心主题与跨主题关联", articles,
                                router=router, top_communities=2, use_llm=use_llm)
                v9["graph_rag"] = {
                    **gr.as_dict(),
                    "qa_question": ga.get("question", ""),
                    "qa_answer": ga.get("answer", "")[:500],
                    "qa_generated_by": ga.get("generated_by", "empty"),
                }
                ok(f"GraphRAG: {gr.n_nodes} 节点 / {gr.n_edges} 边 / "
                   f"{gr.n_communities} 社区 (modularity={gr.modularity:.3f}, {gr.global_generated_by})")
            else: warn("GraphRAG: 知识图谱为空")
        except Exception as e: warn(f"GraphRAG 失败: {e}")

        # 4) ReAct Agent 自主工具调用
        try:
            use_llm_agent = router is not None and getattr(router, "api_key", None)
            agent = build_default_agent(router=router if use_llm_agent else None, max_steps=5)
            ar = agent.run("综合判断今日宏观风险与机会,给出 1 句话投资建议",
                           target_date=target_date)
            v9["agent"] = {
                "question": ar.question,
                "answer": ar.answer[:600],
                "n_steps": ar.n_steps,
                "n_tool_calls": ar.n_tool_calls,
                "final_action": ar.final_action,
                "generated_by": ar.generated_by,
                "elapsed_ms": ar.elapsed_ms,
                "steps": [s.as_dict() for s in ar.steps[:5]],
            }
            ok(f"ReAct Agent: {ar.n_steps} 步 / {ar.n_tool_calls} 工具调用 "
               f"({ar.generated_by}, {ar.elapsed_ms}ms)")
        except Exception as e: warn(f"ReAct Agent 失败: {e}")

        out["v9"] = v9
        ok(f"v9 前沿完成 (4 子模块), 用时 {time.time() - t0:.1f}s")
    except Exception as e: warn(f"v9 step 整体失败: {e}")

    ok(f"前沿完成, 用时 {time.time() - t0:.1f}s")
    return out


def step5_kg(articles, target_date, no_kg=False, router=None):
    """知识图谱构建 + PNG 渲染。"""
    banner("STEP 5/7 - 知识图谱 (实体识别 + 共现 + PageRank + 可视化)")
    t0 = time.time()
    if no_kg:
        warn("--no-kg 已启用, 跳过")
        return None
    try:
        G = build_kg(articles, router=router, min_node_freq=1)
        if G.number_of_nodes() == 0:
            warn("图谱为空(可能无明显实体), 跳过渲染")
            return None
        from src.config import IMAGE_DIR
        png_name = f"knowledge_graph_{target_date}.png"
        png_path = render_kg_png(G, png_name)
        summary = summarize_kg(G, top_k=10)
        ok(f"KG: {summary['stats']['nodes']} 节点 / {summary['stats']['edges']} 边, "
           f"Top 节点: {[n['id'] for n in summary['nodes'][:3]]}, 用时 {time.time() - t0:.1f}s")
        if png_path:
            return f"../images/{png_name}"
        return None
    except Exception as e:
        warn(f"知识图谱失败: {e}")
        return None


def step6_dashboard(target_date, no_dashboard=False, window_days=14):
    """生成单日仪表盘 + 历史索引。"""
    banner("STEP 6/7 - 交互式仪表盘 (Plotly 离线 HTML + GitHub Dark)")
    t0 = time.time()
    if no_dashboard:
        warn("--no-dashboard 已启用, 跳过")
        return
    try:
        p = render_dashboard(target_date, window_days=window_days)
        idx = render_dashboard_index(days=60)
        ok(f"仪表盘: {p}")
        ok(f"历史索引: {idx}, 用时 {time.time() - t0:.1f}s")
    except Exception as e:
        warn(f"仪表盘失败: {e}")


def step7_report(articles, nlp_stats, ai_result, target_date, kg_image=None, advanced=None):
    """生成 Markdown 报告 (v3 模板: 15 节 + 量化指标卡 + KG + 产业-A股 + AI 自评)。"""
    banner("STEP 7/7 - 报告生成 (Markdown v3)")
    t0 = time.time()
    try:
        archive_path = archive_raw(articles, target_date)
        ok(f"JSON 归档: {archive_path}")
    except Exception as e:
        warn(f"JSON 归档失败: {e}")
    try:
        report_path = render_report(articles, nlp_stats, ai_result, target_date, kg_image=kg_image, advanced=advanced)
        ok(f"报告: {report_path} ({report_path.stat().st_size} bytes)")
        # 同步导出 HTML (浏览器打开即可打印/保存为 PDF)
        try:
            from src.report.html_export import export as html_export
            html_path = html_export(str(report_path))
            if html_path:
                ok(f"HTML 报告: {html_path} ({html_path.stat().st_size} bytes) - 浏览器打开 Ctrl+P 可保存为 PDF")
        except Exception as e: warn(f"HTML 导出失败: {e}")
        ok(f"完成, 用时 {time.time() - t0:.1f}s")
        return report_path
    except Exception as e:
        warn(f"报告失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def _build_nlp_fallback_report(articles, nlp_stats):
    """当 --skip-ai 时, 用 NLP 数据构造满足 schema 校验的 AnalysisReport。"""
    from src.ai.schema import (
        ThemeKeywordsResult, ThemeKeyword, PolicyDirectionResult,
        IndustriesResult, IndustryFocus, PoliciesResult,
        CoreInsightsResult, OutlooksResult, Outlook,
        HeatLevel, Stance, Judgment,
    )
    kws_data = nlp_stats.keywords or nlp_stats.word_freq or [("经济", 1.0)]
    kws_data = list(kws_data)[:10]
    if not kws_data:
        kws_data = [("经济", 1.0)]
    tk_list = []
    for i, item in enumerate(kws_data):
        if isinstance(item, tuple):
            word, score = item[0], float(item[1] if len(item) > 1 else 0.5)
        else:
            word, score = str(item), 0.5
        score = max(0.1, min(0.99, score if score <= 1.0 else score / 100.0))
        tk_list.append(ThemeKeyword(
            word=str(word)[:20],
            score=round(score, 3),
            explain="(NLP 兜底, 非 AI 生成)",
        ))
    if not tk_list:
        tk_list.append(ThemeKeyword(word="经济", score=0.5, explain="(NLP 兜底默认词)"))
    ind_list = []
    for ind_name, hits in sorted(nlp_stats.industry_hits.items(), key=lambda x: -x[1])[:5]:
        ind_list.append(IndustryFocus(
            name=str(ind_name)[:10],
            heat=HeatLevel.HIGH if hits >= 5 else HeatLevel.MEDIUM,
            article_count=max(1, int(hits)),
            summary=f"命中 {int(hits)} 次 (NLP 兜底, 非 AI 生成)",
            stance=Stance.NEUTRAL,
        ))
    if not ind_list:
        ind_list.append(IndustryFocus(
            name="综合", heat=HeatLevel.MEDIUM, article_count=len(articles),
            summary="(NLP 兜底, 未识别到特定产业关键词)",
            stance=Stance.NEUTRAL,
        ))
    insight_text = (
        f"基于 {len(articles)} 篇经济新闻 (共 {nlp_stats.total_words} 词) 的 NLP 分析。"
        f"涉及 {len(nlp_stats.industry_hits)} 个产业, 关键词 {len(nlp_stats.keywords)} 个。"
        "AI 分析未启用, 本报告为纯 NLP 摘要。"
    )
    top_ind_name = ind_list[0].name if ind_list else "经济"
    return AnalysisReport(
        theme_keywords=ThemeKeywordsResult(keywords=tk_list),
        policy_direction=PolicyDirectionResult(
            direction="中性", confidence=0.0, keywords=["(NLP 兜底)"],
            interpretation="(跳过 AI, 纯 NLP 模式无法判定政策方向)",
        ),
        industries=IndustriesResult(industries=ind_list),
        policies=PoliciesResult(policies=[]),
        core_insights=CoreInsightsResult(insights=insight_text),
        outlooks=OutlooksResult(outlooks=[
            Outlook(
                topic="整体趋势观察",
                judgment=Judgment.NEUTRAL,
                rationale=f"(NLP 兜底) 共 {len(articles)} 篇报道, 主导产业为 {top_ind_name}, 关键词集中度待 AI 进一步研判。",
            ),
        ]),
    )


def main():
    parser = argparse.ArgumentParser(
        description="宏观经济智能分析平台 - 每日经济新闻深度分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--date", type=str, default="",
                        help="报告日期 YYYY-MM-DD, 默认今天")
    parser.add_argument("--days", type=int, default=2,
                        help="爬取回溯天数(含当天), 默认 2")
    parser.add_argument("--window-days", type=int, default=14,
                        help="仪表盘/对比窗口天数, 默认 14")
    parser.add_argument("--yesterday", action="store_true",
                        help="抓『昨天』而非『今天』 (兼容旧语义)")
    parser.add_argument("--skip-scrape", action="store_true",
                        help="跳过爬取, 从 data/raw/{date}.json 加载")
    parser.add_argument("--skip-ai", action="store_true",
                        help="跳过 AI, 降级为纯 NLP 报告")
    parser.add_argument("--reflect", action="store_true",
                        help="启用 AI 反思循环 (self_eval 低分重生成)")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="关闭交互式仪表盘生成")
    parser.add_argument("--no-kg", action="store_true",
                        help="关闭知识图谱生成")
    parser.add_argument("--no-comparison", action="store_true",
                        help="关闭跨日对比 (无历史数据时使用)")
    parser.add_argument("--enable-xinhua", action="store_true",
                        help="启用新华网源 (默认关闭, .env 需 ENABLE_XINHUA=1)")
    parser.add_argument("--enable-pbc", action="store_true",
                        help="启用央行源 (默认关闭, .env 需 ENABLE_GOV_PBOC=1)")
    parser.add_argument("--no-dedup", action="store_true",
                        help="关闭语义去重")
    parser.add_argument("--debug", action="store_true", help="调试日志")
    args = parser.parse_args()

    if args.debug:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    if args.date:
        target_date = resolve_target_date(args.date).isoformat()
    elif args.yesterday:
        target_date = yesterday().isoformat()
    else:
        target_date = resolve_target_date("").isoformat()
    banner(f"宏观经济智能分析 · {target_date} · v5")

    router = None
    if not args.skip_ai:
        try:
            router = get_default_router()
        except Exception as e:
            warn(f"router 初始化失败: {e}, AI 步骤可能失败")

    if args.skip_scrape:
        warn(f"--skip-scrape 已启用, 从归档加载 {target_date}")
        try:
            articles = load_raw(target_date)
            ok(f"加载完成: {len(articles)} 篇")
        except FileNotFoundError as e:
            print(f"[X] {e}")
            return 1
    else:
        articles = step1_scrape(
            target_date, lookback_days=args.days,
            enable_xinhua=args.enable_xinhua, enable_pbc=args.enable_pbc,
            enable_dedup=not args.no_dedup, router=router,
        )
        if not articles:
            warn("未抓取到任何文章, 退出")
            return 0

    nlp_stats = step2_nlp(articles)

    if args.skip_ai:
        warn("--skip-ai 已启用, 跳过 AI")
        ai_result = _build_nlp_fallback_report(articles, nlp_stats)
    else:
        ai_result = step3_ai(articles, nlp_stats, target_date, router=router, reflect=args.reflect, date=target_date)

    step4_quant_and_compare(articles, nlp_stats, ai_result, target_date, no_comparison=args.no_comparison, window_days=args.window_days)

    advanced = step4b_advanced(target_date, nlp_stats, ai_result, articles=articles, router=router)

    kg_image = step5_kg(articles, target_date, no_kg=args.no_kg, router=router)

    step6_dashboard(target_date, no_dashboard=args.no_dashboard, window_days=args.window_days)

    report_path = step7_report(articles, nlp_stats, ai_result, target_date, kg_image=kg_image, advanced=advanced)

    banner("全部完成")
    print(f"  日期: {target_date}")
    print(f"  文章: {len(articles)} 篇")
    print(f"  报告: {report_path}")
    print(f"  仪表盘: dashboard/{target_date}.html")
    print(f"  历史索引: dashboard/index.html")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
