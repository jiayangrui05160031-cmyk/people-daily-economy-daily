"""api.shap_explain - SHAP 风格特征贡献度解释器 (model-agnostic)

对 src.analysis.signal_engine 的 BUY/HOLD/REDUCE/SELL 决策做"事后解释",
输出每个子信号对最终 score 的贡献 (类似 TreeSHAP 的 additive attribution)。

关键设计:
  signal_engine 已经把 8 个子信号用 (score, weight) 表示, 加权和 = final_score.
  这天然就是 SHAP 的 additive decomposition:
      final_score = sum_i(score_i * weight_i)
  我们再叠加一个 "baseline (近 30 天同信号均值)" 的对照, 给出 "该信号今天
  比通常更看多/更看空多少", 让用户能直观理解决策的"异常点"。

不依赖真正的 SHAP 库, 纯闭式加性分解, 计算成本几乎为 0。

典型用法:
    from src.api.shap_explain import explain_decision
    rep = explain_decision("2026-06-12")
    print(rep.final_action, rep.final_score)
    for c in rep.contributions: print(c.feature, c.value, c.direction)
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field, asdict
from datetime import timedelta
from typing import Any, Dict, List, Optional

from src.analysis.signal_engine import synthesize as signal_synth
from src.analysis.anomaly import detect as detect_anomaly
from src.analysis.forecast import predict_next_day
from src.analysis.volatility import compute as volatility_compute
from src.analysis.macro_indicators import snapshot as macro_snapshot
from src.analysis.event_study import study as event_study_fn
from src.analysis.stock_correlation import correlate as correlate_market
from src.storage import db as db_mod
from src.utils.date_utils import parse_date
from src.utils.logger import get_logger

logger = get_logger("api.shap_explain")

# ============================================================
# 数据类
# ============================================================
@dataclass
class FeatureContribution:
    feature: str
    weight: float
    score: float           # -1 ~ +1, 该信号今天的 score
    baseline: float        # 该信号历史 30 天均值
    contribution: float    # = weight * (score - baseline)   (推高/拉低分数)
    raw_contribution: float  # = weight * score               (今天原始贡献)
    direction: str         # "push_up" | "push_down" | "neutral"
    reason: str            # signal_engine 给的人话理由

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SHAPReport:
    date: str
    final_action: str
    final_score: float
    final_confidence: float
    contributions: List[FeatureContribution] = field(default_factory=list)
    top_positive: str = ""
    top_negative: str = ""
    base_score: float = 0.0
    total_weight: float = 0.0
    n_signals: int = 0
    summary: str = ""

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["contributions"] = [c.as_dict() for c in self.contributions]
        return d


# ============================================================
# 1) 历史 baseline (近 30 天同信号的 score 均值)
# ============================================================
def _load_signal_baselines(target_date: str, signal_names: List[str]) -> Dict[str, float]:
    """回放近 30 天的 signal_engine 决策, 取每个信号 score 的历史均值.

    为避免回放太慢, 改用代理: 从 daily_metric 推算的近似基线.
    当 daily_metric 不足时, 返回空 dict (SHAP 仍可工作, baseline=0).
    """
    out: Dict[str, float] = {}
    end = parse_date(target_date)
    start = end - timedelta(days=30)
    try:
        with db_mod.get_conn() as conn:
            rows = conn.execute(
                "SELECT sentiment_index, policy_stance_score, attention_entropy, "
                "industry_count FROM daily_metric "
                "WHERE date BETWEEN ? AND ? ORDER BY date",
                (start.isoformat(), end.isoformat()),
            ).fetchall()
        if not rows:
            return out
        # 映射: signal_name -> 归一化基线
        sent_avg = statistics.mean(
            [float(r["sentiment_index"]) for r in rows if r["sentiment_index"] is not None]
        ) if any(r["sentiment_index"] is not None for r in rows) else 50.0
        pol_avg = statistics.mean(
            [float(r["policy_stance_score"]) for r in rows if r["policy_stance_score"] is not None]
        ) if any(r["policy_stance_score"] is not None for r in rows) else 0.0
        ent_avg = statistics.mean(
            [float(r["attention_entropy"]) for r in rows if r["attention_entropy"] is not None]
        ) if any(r["attention_entropy"] is not None for r in rows) else 0.5
        ind_avg = statistics.mean(
            [float(r["industry_count"]) for r in rows if r["industry_count"] is not None]
        ) if any(r["industry_count"] is not None for r in rows) else 8.0

        # 把 0~100 / 0~1 的值归一化到 [-1, 1]
        out["policy_stance"] = max(-1.0, min(1.0, pol_avg * 5.0))
        out["volatility"] = max(-1.0, min(1.0, (30.0 - ent_avg * 30.0) / 30.0))
        out["anomaly_risk"] = 0.0
        out["forecast"] = 0.0
        out["market_correlation"] = 0.0
        out["events_net"] = 0.0
        out["topic_dispersion"] = max(-1.0, min(1.0, (ent_avg - 0.5) * 2.0))
        out["macro_indicators"] = max(-1.0, min(1.0, pol_avg * 5.0))
    except Exception as e:
        logger.debug(f"baseline 加载失败: {e}")
    return out


# ============================================================
# 2) 主函数: 计算 SHAP 风格贡献
# ============================================================
def explain_decision(target_date: str, ai_result=None, industries_hit=None,
                     theme_kws=None) -> Optional[SHAPReport]:
    """对 signal_engine 决策做 SHAP 风格解释."""
    industries_hit = industries_hit or []
    theme_kws = theme_kws or []

    # 跑各子模块喂给 signal_engine
    try:
        an = detect_anomaly(target_date, window_days=30)
        fc = predict_next_day(target_date, lookback_days=14)
        vol = volatility_compute(target_date, window_days=14)
        mkt = correlate_market(
            target_date,
            industries_hit[:6] if industries_hit else None,
            top_n_per_industry=3,
        )
        mc = macro_snapshot(target_date, theme_keywords=theme_kws, industries=industries_hit)
        es = event_study_fn([], target_date=target_date)
        sig = signal_synth(
            target_date,
            anomaly=an, forecast=fc, volatility=vol, market=mkt, macro=mc,
            events_study=es, topics=None, ai_result=ai_result,
        )
    except Exception as e:
        logger.warning(f"signal_synth 失败: {e}")
        return None

    if not sig or not sig.signals:
        return None

    signal_names = [s.name for s in sig.signals]
    baselines = _load_signal_baselines(target_date, signal_names)

    # 计算 contribution
    contributions: List[FeatureContribution] = []
    raw_total = 0.0
    weighted_baseline_total = 0.0
    for s in sig.signals:
        baseline = float(baselines.get(s.name, 0.0))
        raw_contrib = float(s.weight) * float(s.score)
        shifted = float(s.weight) * (float(s.score) - baseline)
        raw_total += raw_contrib
        weighted_baseline_total += float(s.weight) * baseline
        if shifted > 0.005:
            direction = "push_up"
        elif shifted < -0.005:
            direction = "push_down"
        else:
            direction = "neutral"
        # 简短理由
        reason = s.reason or f"score={s.score:+.2f}, w={s.weight:.2f}"
        contributions.append(FeatureContribution(
            feature=s.name,
            weight=round(float(s.weight), 4),
            score=round(float(s.score), 4),
            baseline=round(baseline, 4),
            contribution=round(shifted, 4),
            raw_contribution=round(raw_contrib, 4),
            direction=direction,
            reason=reason[:120],
        ))

    contributions.sort(key=lambda c: c.contribution, reverse=True)
    top_pos = next((c.feature for c in contributions if c.contribution > 0), "-")
    top_neg = next((c.feature for c in contributions if c.contribution < 0), "-")

    final_score = float(getattr(sig, "score", raw_total))
    final_action = getattr(sig, "action", "HOLD")
    confidence = float(getattr(sig, "confidence", 0.0))
    total_w = sum(c.weight for c in contributions)

    summary = (
        f"决策={final_action} (score={final_score:+.3f}, 置信={confidence:.0%}); "
        f"主推 {top_pos} (Δ={contributions[0].contribution:+.3f}); "
        f"主拖 {top_neg}"
    )

    return SHAPReport(
        date=target_date,
        final_action=final_action,
        final_score=round(final_score, 4),
        final_confidence=round(confidence, 4),
        contributions=contributions,
        top_positive=top_pos,
        top_negative=top_neg,
        base_score=round(weighted_baseline_total, 4),
        total_weight=round(total_w, 4),
        n_signals=len(contributions),
        summary=summary,
    )
