"""ai - AI 推理层 (v2)

10 维分析: 6 个原任务 + 4 个新任务 (sentiment/events/cross_links/self_eval)
- 强类型: Pydantic v2 AnalysisReport
- 并发: asyncio.Semaphore 控制并发
- 智能路由: 多模型 fallback 链
- 缓存: SQLite WAL
- 追踪: JSONL 链路日志
"""

from .analyzer import analyze_all, analyze_for_date
from .schema import AnalysisReport

__all__ = ["analyze_all", "analyze_for_date", "AnalysisReport"]
