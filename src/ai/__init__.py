"""ai — AI 推理层 ==============================================
OpenAI 兼容协议的 LLM 调用,6 个 prompt 任务输出 6 维度分析结果。
"""

from .analyzer import analyze_all, AnalysisResult

__all__ = ["analyze_all", "AnalysisResult"]