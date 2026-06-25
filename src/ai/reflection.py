"""ai.reflection - AI 反思循环 (ReAct-style)

把 self_eval 的低分结果拿出来重新生成一次,最多 2 轮。
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from src.ai.schema import AnalysisReport, SelfEval
from src.config import AI_REFLECTION_MAX_ROUNDS, AI_REFLECTION_THRESHOLD
from src.utils.logger import get_logger

logger = get_logger("ai.reflection")


def should_reflect(self_eval: Optional[SelfEval], threshold=None) -> bool:
    threshold = threshold or AI_REFLECTION_THRESHOLD
    if not self_eval:
        return False
    return (self_eval.consistency < threshold or
            self_eval.groundedness < threshold or
            self_eval.completeness < threshold)


def reflect_on_report(router, report: AnalysisReport, articles, prior_results: Dict[str, Any],
                      date: str = "", max_rounds=None) -> AnalysisReport:
    """若 self_eval 分数低于阈值,触发二次反思。"""
    max_rounds = max_rounds if max_rounds is not None else AI_REFLECTION_MAX_ROUNDS
    if not router:
        return report
    if not should_reflect(report.self_eval):
        return report

    for rnd in range(max_rounds):
        logger.info(f"[reflection] round {rnd + 1}/{max_rounds}")
        from src.ai.prompts import format_articles_for_llm
        try:
            text = format_articles_for_llm(articles, max_chars=4000)
            prior_str = json.dumps({k: (v.model_dump() if hasattr(v, "model_dump") else v)
                                    for k, v in prior_results.items()},
                                   ensure_ascii=False)[:3000]
            prompt = (
                "你之前的分析结果 self_eval 分数较低,以下是前几轮结果摘要。\n"
                "请基于原文,修订你认为有误的部分,返回完整 JSON。\n\n"
                f"原文:\n{text}\n\n前次结果:\n{prior_str}\n\n"
                "返回 JSON 对象,字段包含 keywords/direction/interpretation/industries/policies/insights/outlooks 等。"
            )
            raw, _, _, _ = router.chat(
                messages=[{"role": "system", "content": "你是中国宏观经济分析专家,输出修订后的 JSON。"},
                          {"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2500,
                use_json_mode=True,
            )
            from src.ai.router import extract_json
            parsed = extract_json(raw or "")
            if parsed:
                if "direction" in parsed and report.policy_direction:
                    d = str(parsed.get("direction", report.policy_direction.direction))
                    if d in ("扩张", "收紧", "中性"):
                        report.policy_direction.direction = d
                if "interpretation" in parsed and report.policy_direction:
                    iv = str(parsed["interpretation"])
                    if 15 <= len(iv) <= 250:
                        report.policy_direction.interpretation = iv
                logger.info(f"[reflection] round {rnd + 1} applied")
                break
        except Exception as e:
            logger.warning(f"[reflection] round {rnd + 1} failed: {e}")

    return report


if __name__ == "__main__":
    print("reflection module self-test (no LLM):")
    se = SelfEval(consistency=0.6, groundedness=0.9, completeness=0.8,
                  comments="一致性需要改进,覆盖度可接受。")
    print("should_reflect:", should_reflect(se, threshold=0.7))
