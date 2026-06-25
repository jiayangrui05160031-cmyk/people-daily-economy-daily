"""analysis.signal_engine - 多信号融合决策引擎

把 9 个前沿分析模块的输出量化成一组投票 (-1 ~ +1), 加权合成一个综合信号,
输出 actionable 决策建议 (BUY / HOLD / REDUCE / SELL) + 置信度 + 关键理由.

设计要点:
- 每个输入信号通过 `_extract_signal()` 抽取出 (score, weight, reason) 三元组
- 缺失模块 = 中性 (0) 不影响总分
- 聚合分数按 sigmoid 软化, 决策阈值 [-0.3, 0.3] 分四档
- 行业视角: 同步给出每个被关注行业的子信号 (基于 market_correlation + topics)
- 完全 offline: 不调 LLM, 适合作为仪表盘/报告的"决策之心"

典型用法:
    from src.analysis.signal_engine import synthesize
    sig = synthesize(target_date, anomaly, forecast, volatility, market, macro, events_study, topics, policy_direction)
    print(sig.action, sig.score, sig.top_reasons[:3])
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

from src.utils.date_utils import parse_date
from src.utils.logger import get_logger

logger = get_logger("analysis.signal_engine")


# ============================================================
# 信号数据类
# ============================================================
@dataclass
class Signal:
    name: str            # 信号名 (如 "policy_stance")
    score: float         # -1 (强烈看空) ~ +1 (强烈看多)
    weight: float        # 0 ~ 1, 该信号在总合成里的权重
    reason: str          # 人话理由, 1 行
    direction: str = ""  # 看多/看空/中性

    def as_dict(self):
        return asdict(self)


@dataclass
class IndustrySignal:
    industry: str
    score: float
    action: str
    drivers: List[str] = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


@dataclass
class DecisionReport:
    date: str
    score: float                 # 合成分数 (-1 ~ +1)
    confidence: float            # 0 ~ 1
    action: str                  # BUY / HOLD / REDUCE / SELL
    signals: List[Signal]
    industry_signals: List[IndustrySignal] = field(default_factory=list)
    top_reasons: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    summary: str = ""

    def as_dict(self):
        d = asdict(self)
        d["signals"] = [s.as_dict() for s in self.signals]
        d["industry_signals"] = [i.as_dict() for i in self.industry_signals]
        return d


# ============================================================
# 单信号提取
# ============================================================
def _safe_get(obj, *path, default=None):
    """安全取值, 避免 AttributeError."""
    cur = obj
    for p in path:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(p)
        else:
            cur = getattr(cur, p, None)
    return cur if cur is not None else default


def _sig_from_policy(ai_result) -> Optional[Signal]:
    if ai_result is None:
        return None
    d = _safe_get(ai_result, "policy_direction", "direction")
    c = _safe_get(ai_result, "policy_direction", "confidence", default=0.5)
    if not d:
        return None
    table = {"扩张": 1.0, "收紧": -1.0, "中性": 0.0}
    score = table.get(d, 0.0) * float(c)
    return Signal(
        name="policy_stance", score=round(score, 3), weight=0.20,
        reason=f"政策风向={d} (置信 {c:.0%})", direction=d,
    )


def _sig_from_anomaly(anomaly) -> Optional[Signal]:
    if anomaly is None or not _safe_get(anomaly, "signals"):
        return None
    risk = _safe_get(anomaly, "overall_risk", default="low")
    n = len(_safe_get(anomaly, "signals", default=[]) or [])
    table = {"low": 0.2, "medium": -0.2, "high": -0.6, "extreme": -1.0}
    score = table.get(risk, 0.0)
    return Signal(
        name="anomaly_risk", score=round(score, 3), weight=0.15,
        reason=f"异常风险={risk}, 触发 {n} 项", direction="bear" if score < 0 else "bull",
    )


def _sig_from_forecast(forecast) -> Optional[Signal]:
    if forecast is None:
        return None
    headline = _safe_get(forecast, "headline", default="")
    pred = _safe_get(forecast, "predicted_sentiment", default=50.0)
    cur = _safe_get(forecast, "current_sentiment", default=50.0)
    delta = float(pred) - float(cur)
    score = max(-1.0, min(1.0, delta / 20.0))   # ±20 分 => ±1
    return Signal(
        name="forecast", score=round(score, 3), weight=0.15,
        reason=f"明日情绪预测 {pred:.0f} (今日 {cur:.0f}, Δ={delta:+.1f})",
        direction="bull" if score > 0 else ("bear" if score < 0 else "neutral"),
    )


def _sig_from_volatility(vol) -> Optional[Signal]:
    if vol is None:
        return None
    idx = _safe_get(vol, "index", default=0)
    level = _safe_get(vol, "level", default="unknown")
    # 波动率高 = 不确定性 = 偏空
    table = {"extreme_calm": 0.3, "calm": 0.15, "neutral": 0.0, "fear": -0.3, "extreme_fear": -0.8}
    score = table.get(level, 0.0)
    if isinstance(idx, (int, float)) and 0 < idx <= 100 and level == "unknown":
        score = max(-1.0, min(0.3, -((idx - 50) / 50)))
    return Signal(
        name="volatility", score=round(score, 3), weight=0.10,
        reason=f"恐慌指数 {idx:.1f} ({level})",
        direction="bear" if score < 0 else ("bull" if score > 0 else "neutral"),
    )


def _sig_from_market(market) -> Optional[Signal]:
    if market is None:
        return None
    overall = _safe_get(market, "market_overall", default="neutral")
    table = {"bullish": 0.6, "neutral": 0.0, "bearish": -0.6}
    score = table.get(overall, 0.0)
    return Signal(
        name="market_correlation", score=round(score, 3), weight=0.15,
        reason=f"A 股联动={overall}", direction=overall,
    )


def _sig_from_events(events_study) -> Optional[Signal]:
    if events_study is None:
        return None
    events = _safe_get(events_study, "events", default=[]) or []
    if not events:
        return None
    pos = sum(1 for e in events if _safe_get(e, "stance") in ("利好", "positive", "bullish"))
    neg = sum(1 for e in events if _safe_get(e, "stance") in ("利空", "negative", "bearish"))
    total = max(len(events), 1)
    net = (pos - neg) / total
    return Signal(
        name="events_net", score=round(net, 3), weight=0.10,
        reason=f"事件 {len(events)} 个 (利好 {pos} / 利空 {neg})",
        direction="bull" if net > 0 else ("bear" if net < 0 else "neutral"),
    )


def _sig_from_topics(topics) -> Optional[Signal]:
    if topics is None:
        return None
    top = (topics.topics or [])[:3]
    if not top:
        return None
    # 主导主题分散 => 多样化 = 中性偏多; 单一主题过强 => 偏空 (风险集中)
    ent = _safe_get(topics, "doc_topic_entropy", default=0.5)
    score = (ent - 0.5) * 0.6   # 熵 0~1, 中性 0.5
    return Signal(
        name="topic_dispersion", score=round(score, 3), weight=0.05,
        reason=f"主题分布熵={ent:.2f} (Top: {', '.join(t.label for t in top[:3])})",
        direction="bull" if score > 0 else ("bear" if score < 0 else "neutral"),
    )


def _sig_from_macro(macro) -> Optional[Signal]:
    if macro is None:
        return None
    inds = _safe_get(macro, "indicators", default=[]) or []
    if not inds:
        return None
    score_sum = 0.0
    n = 0
    for ind in inds:
        cat = _safe_get(ind, "category", default="")
        value = _safe_get(ind, "value")
        prev = _safe_get(ind, "previous")
        if value is None or prev in (None, 0):
            continue
        try:
            v, p = float(value), float(prev)
        except Exception:
            continue
        delta = (v - p) / max(abs(p), 1e-6)
        # 货币/财政类: 涨 = 看空 (紧缩) ; PMI 类: 涨 = 看多
        if cat in ("货币", "财政"):
            score_sum -= delta * 0.5
        elif cat in ("景气", "PMI", "增长"):
            score_sum += delta * 0.5
        else:
            score_sum += delta * 0.2
        n += 1
    if n == 0:
        return None
    score = max(-1.0, min(1.0, score_sum / n))
    return Signal(
        name="macro_indicators", score=round(score, 3), weight=0.10,
        reason=f"宏观 {n} 项环比净变动",
        direction="bull" if score > 0 else ("bear" if score < 0 else "neutral"),
    )


# ============================================================
# 行业信号
# ============================================================
def _industry_signals(market, topics) -> List[IndustrySignal]:
    out: List[IndustrySignal] = []
    if market is not None:
        for ind in (_safe_get(market, "industries", default=[]) or []):
            name = _safe_get(ind, "industry", default="")
            if not name:
                continue
            stance = _safe_get(ind, "stance", default="中性")
            score = {"利好": 0.7, "中性": 0.0, "利空": -0.7}.get(stance, 0.0)
            drivers = []
            sent = _safe_get(ind, "sentiment", default=0.5)
            if sent:
                drivers.append(f"新闻情感 {float(sent):.2f}")
            conf = _safe_get(ind, "confidence", default=0.5)
            drivers.append(f"AI 置信 {float(conf):.0%}")
            out.append(IndustrySignal(
                industry=name, score=round(score, 3),
                action="关注" if score >= 0.4 else ("观望" if score > -0.4 else "回避"),
                drivers=drivers,
            ))
    if topics is not None and (topics.topics or []):
        # 把 LDA 主题的主导产业也并入
        for t in (topics.topics or [])[:5]:
            if not t.dominant_industry:
                continue
            existing = next((x for x in out if x.industry == t.dominant_industry), None)
            topic_score = (t.dominance - 0.2) * 1.5   # 占比 > 0.2 算显著
            topic_score = max(-1.0, min(1.0, topic_score))
            if existing is None:
                out.append(IndustrySignal(
                    industry=t.dominant_industry, score=round(topic_score, 3),
                    action="关注" if topic_score >= 0.4 else ("观望" if topic_score > -0.4 else "回避"),
                    drivers=[f"LDA 主题占比 {t.dominance:.2f}"],
                ))
            else:
                # 合并: 取平均
                existing.score = round((existing.score + topic_score) / 2, 3)
                existing.drivers.append(f"LDA 主题占比 {t.dominance:.2f}")
    # 排序按 |score| 降序, 截前 8
    out.sort(key=lambda x: abs(x.score), reverse=True)
    return out[:8]


# ============================================================
# 主合成
# ============================================================
def _decide(score: float, conf: float) -> str:
    if score >= 0.3:
        return "BUY" if conf >= 0.55 else "HOLD"
    if score <= -0.3:
        return "SELL" if conf >= 0.55 else "REDUCE"
    return "HOLD"


def _softmax_confidence(signals: List[Signal]) -> float:
    """置信度 = 信号一致性 (低方差) × 平均绝对权重."""
    if not signals:
        return 0.0
    abs_scores = [abs(s.score) * s.weight for s in signals]
    total = sum(abs_scores) or 1.0
    # 一致性: 1 - std/mean (避免除零)
    avg = sum(abs_scores) / len(abs_scores)
    var = sum((x - avg) ** 2 for x in abs_scores) / len(abs_scores)
    std = var ** 0.5
    consistency = max(0.0, 1.0 - std / max(avg, 1e-6))
    magnitude = min(1.0, total / 0.6)   # 总权重 0.6 视为满
    return round(0.6 * consistency + 0.4 * magnitude, 3)


def synthesize(target_date: str,
               anomaly=None, forecast=None, volatility=None,
               market=None, macro=None, events_study=None,
               topics=None, ai_result=None) -> DecisionReport:
    """合成多信号, 输出决策报告."""
    sig_fns = [
        _sig_from_policy,
        _sig_from_anomaly,
        _sig_from_forecast,
        _sig_from_volatility,
        _sig_from_market,
        _sig_from_events,
        _sig_from_topics,
        _sig_from_macro,
    ]
    inputs = {
        "policy": ai_result, "anomaly": anomaly, "forecast": forecast,
        "volatility": volatility, "market": market, "macro": macro,
        "events": events_study, "topics": topics,
    }
    signals: List[Signal] = []
    fn_inputs = {
        _sig_from_policy: ai_result, _sig_from_anomaly: anomaly,
        _sig_from_forecast: forecast, _sig_from_volatility: volatility,
        _sig_from_market: market, _sig_from_events: events_study,
        _sig_from_topics: topics, _sig_from_macro: macro,
    }
    for fn, arg in fn_inputs.items():
        try:
            sig = fn(arg)
            if sig is not None:
                signals.append(sig)
        except Exception as e:
            logger.debug(f"signal {fn.__name__} 解析失败: {e}")

    # 加权合成
    if signals:
        w_sum = sum(s.weight for s in signals) or 1.0
        score = sum(s.score * s.weight for s in signals) / w_sum
    else:
        score = 0.0
    score = max(-1.0, min(1.0, score))
    confidence = _softmax_confidence(signals)
    action = _decide(score, confidence)

    # Top reasons: 按 |score*weight| 排序
    ranked = sorted(signals, key=lambda s: abs(s.score * s.weight), reverse=True)
    top_reasons = [s.reason for s in ranked[:5] if abs(s.score) > 0.05]

    # Risks: 看空方向
    risks = [s.reason for s in ranked if s.score < -0.15][:3]

    # 行业信号
    industry_signals = _industry_signals(market, topics)

    summary = (
        f"综合信号 {score:+.2f} ({action}, 置信 {confidence:.0%}); "
        f"{len(signals)} 个有效信号; "
        + (f"Top: {top_reasons[0]}" if top_reasons else "无突出理由")
    )

    return DecisionReport(
        date=target_date,
        score=round(score, 3),
        confidence=round(confidence, 3),
        action=action,
        signals=signals,
        industry_signals=industry_signals,
        top_reasons=top_reasons,
        risks=risks,
        summary=summary,
    )


if __name__ == "__main__":
    # 自检: 给假数据, 看决策流是否合理
    import json
    from dataclasses import dataclass as _dc, field as _f

    class _A:
        """通用 mock 对象, 接受任意 kwargs."""
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    a = _A(); a.signals = [{"x":1}]; a.overall_risk = "medium"
    f = _A(); f.headline = "看多"; f.predicted_sentiment = 65.0; f.current_sentiment = 55.0
    v = _A(); v.index = 25.0; v.level = "calm"
    m = _A(); m.market_overall = "bullish"; m.industries = [
        _A(**{"industry":"新能源", "stance":"利好", "sentiment":0.7, "confidence":0.8}),
        _A(**{"industry":"房地产", "stance":"利空", "sentiment":0.3, "confidence":0.6}),
    ]
    macro = _A(); macro.indicators = [
        _A(**{"category":"景气", "value":50.5, "previous":49.8}),
        _A(**{"category":"货币", "value":3.1, "previous":3.2}),
    ]
    es = _A(); es.events = [
        _A(stance="利好"), _A(stance="利空"), _A(stance="中性"), _A(stance="利好"),
    ]
    pd = _A(); pd.policy_direction = _A(direction="扩张", confidence=0.75)
    ai = _A(); ai.policy_direction = pd.policy_direction

    rep = synthesize("2026-06-12", anomaly=a, forecast=f, volatility=v,
                     market=m, macro=macro, events_study=es, ai_result=ai)
    print(json.dumps(rep.as_dict(), ensure_ascii=False, indent=2))
    assert rep.action in ("BUY", "HOLD", "REDUCE", "SELL")
    assert -1.0 <= rep.score <= 1.0
    assert 0.0 <= rep.confidence <= 1.0
    print("[OK] signal_engine self-test passed")
