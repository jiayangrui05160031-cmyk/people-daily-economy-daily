"""parallel.py - 异步并发执行器 ===========================================
- asyncio.Semaphore 控制并发数
- 每个 task 独立错误处理(不影响其他)
- 返回值用 Pydantic schema 强约束
- 任务编排支持"主任务 + 依赖任务"(第二轮)
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Generic, List, Optional, Type, TypeVar

from pydantic import BaseModel

from src.ai.router import ModelRouter, get_default_router
from src.utils.logger import get_logger

logger = get_logger("ai.parallel")

T = TypeVar("T", bound=BaseModel)


@dataclass
class TaskResult(Generic[T]):
    name: str
    success: bool
    data: Optional[T] = None
    error: str = ""
    latency_ms: int = 0
    cached: bool = False


async def _run_one(
    name: str,
    runner: Callable[[], Awaitable[BaseModel]],
    semaphore: asyncio.Semaphore,
) -> TaskResult:
    """跑单个 task(受信号量控制)。"""
    async with semaphore:
        t0 = time.time()
        try:
            data = await runner()
            return TaskResult(
                name=name, success=True, data=data,
                latency_ms=int((time.time() - t0) * 1000),
            )
        except Exception as e:
            return TaskResult(
                name=name, success=False, error=str(e),
                latency_ms=int((time.time() - t0) * 1000),
            )


async def run_parallel(
    tasks: Dict[str, Callable[[], Awaitable[BaseModel]]],
    concurrency: int = 4,
) -> Dict[str, TaskResult]:
    """并发跑多个 task,返回 {name: TaskResult}。

    Args:
        tasks: {name: 0-arg async callable 返回 Pydantic model}
        concurrency: 最大并发数
    """
    sem = asyncio.Semaphore(concurrency)
    coros = [_run_one(name, runner, sem) for name, runner in tasks.items()]
    results = await asyncio.gather(*coros, return_exceptions=False)
    return {r.name: r for r in results}


def run_sync(
    tasks: Dict[str, Callable[[], Awaitable[BaseModel]]],
    concurrency: int = 4,
) -> Dict[str, TaskResult]:
    """同步入口(在非 async 上下文里跑)。"""
    return asyncio.run(run_parallel(tasks, concurrency))


# ============================================================
# 高阶编排:两阶段(主任务 + 依赖任务)
# ============================================================
async def run_two_stage(
    stage1: Dict[str, Callable[[], Awaitable[BaseModel]]],
    stage2_factory: Callable[[Dict[str, BaseModel]], Dict[str, Callable[[], Awaitable[BaseModel]]]],
    concurrency: int = 4,
) -> Dict[str, TaskResult]:
    """两阶段并发:第二阶段依赖第一阶段结果(如 self_eval 依赖主分析结果)。"""
    s1 = await run_parallel(stage1, concurrency)
    stage1_data = {name: r.data for name, r in s1.items() if r.success and r.data}
    s2_specs = stage2_factory(stage1_data)
    s2 = await run_parallel(s2_specs, concurrency)
    return {**s1, **s2}


def run_two_stage_sync(
    stage1: Dict[str, Callable[[], Awaitable[BaseModel]]],
    stage2_factory: Callable[[Dict[str, BaseModel]], Dict[str, Callable[[], Awaitable[BaseModel]]]],
    concurrency: int = 4,
) -> Dict[str, TaskResult]:
    return asyncio.run(run_two_stage(stage1, stage2_factory, concurrency))


if __name__ == "__main__":
    import asyncio
    from pydantic import BaseModel, Field

    class Hello(BaseModel):
        msg: str
        n: int

    async def make_hello(name: str, n: int) -> Hello:
        await asyncio.sleep(0.1)
        return Hello(msg=f"hi {name}", n=n)

    tasks = {
        "a": lambda: make_hello("a", 1),
        "b": lambda: make_hello("b", 2),
        "c": lambda: make_hello("c", 3),
    }
    r = run_sync(tasks, concurrency=2)
    assert all(x.success for x in r.values())
    assert r["a"].data.msg == "hi a"
    print(f"parallel 3 tasks: {[(k, v.latency_ms) for k, v in r.items()]}")

    # error handling
    async def fail() -> Hello:
        raise RuntimeError("boom")

    r = run_sync({"x": fail, "y": lambda: make_hello("y", 99)})
    assert not r["x"].success
    assert r["y"].success
    print(f"error handling: x={r['x'].error}, y.ok={r['y'].data.msg}")

    print("All parallel self-tests passed")
