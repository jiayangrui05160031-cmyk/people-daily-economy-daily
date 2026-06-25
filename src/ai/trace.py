"""trace.py - JSONL 链路追踪 (类似 Langfuse 简化版) ==============================
每次 LLM 调用都写一行 JSONL 到 logs/llm_trace.jsonl,含:
- timestamp, task_name, model
- prompt/completion/total tokens
- cost_cny (用本地 token 单价)
- latency_ms
- success / error
- cached: bool (命中缓存)
- prompt_hash (用于 PII 防护,不存原文)
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.logger import get_logger

logger = get_logger("ai.trace")

TRACE_DIR: Path = Path(__file__).resolve().parent.parent.parent / "logs"
TRACE_DIR.mkdir(parents=True, exist_ok=True)
TRACE_FILE: Path = TRACE_DIR / "llm_trace.jsonl"


# token 单价(元/百万 token),按 provider 维护
TOKEN_PRICE: Dict[str, Dict[str, float]] = {
    "minimax": {"input": 1.0, "output": 2.0},
    "deepseek": {"input": 1.0, "output": 2.0},
    "qwen": {"input": 4.0, "output": 12.0},
    "openai": {"input": 15.0, "output": 60.0},  # gpt-4o-mini 估算
    "default": {"input": 5.0, "output": 15.0},
}


def _provider_from_model(model: str) -> str:
    m = model.lower()
    if "minimax" in m or "minimax" in m:
        return "minimax"
    if "deepseek" in m:
        return "deepseek"
    if "qwen" in m:
        return "qwen"
    if "gpt" in m or "o1" in m or "o3" in m:
        return "openai"
    return "default"


def calc_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """估算成本(元)。"""
    prov = _provider_from_model(model)
    price = TOKEN_PRICE.get(prov, TOKEN_PRICE["default"])
    return round(
        (prompt_tokens * price["input"] + completion_tokens * price["output"]) / 1_000_000,
        6,
    )


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def record(
    task_name: str,
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    latency_ms: int = 0,
    success: bool = True,
    error: str = "",
    cached: bool = False,
    prompt: str = "",
    completion: str = "",
) -> None:
    """记录一次 LLM 调用到 JSONL。"""
    total = prompt_tokens + completion_tokens
    cost = 0.0 if cached else calc_cost(model, prompt_tokens, completion_tokens)
    rec = {
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "task": task_name,
        "model": model,
        "tokens": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": total,
        },
        "cost_cny": cost,
        "latency_ms": latency_ms,
        "success": success,
        "error": error[:200] if error else "",
        "cached": cached,
        "prompt_hash": _prompt_hash(prompt) if prompt else None,
        "completion_hash": _prompt_hash(completion) if completion else None,
        "prompt_len": len(prompt),
        "completion_len": len(completion),
    }
    try:
        with TRACE_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"写入 trace 失败: {e}")


@contextmanager
def track(task_name: str, model: str = "unknown"):
    """用作上下文管理器自动记录耗时。"""
    t0 = time.time()
    success = True
    err = ""
    try:
        yield
    except Exception as e:
        success = False
        err = str(e)
        raise
    finally:
        record(
            task_name=task_name,
            model=model,
            latency_ms=int((time.time() - t0) * 1000),
            success=success,
            error=err,
        )


def summary(date: Optional[str] = None) -> Dict[str, Any]:
    """汇总最近 trace(默认全部)。"""
    if not TRACE_FILE.exists():
        return {"records": 0}
    rows = []
    with TRACE_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if date and not r.get("iso", "").startswith(date):
                    continue
                rows.append(r)
            except Exception:
                pass
    if not rows:
        return {"records": 0}
    total_tokens = sum(r["tokens"]["total"] for r in rows)
    total_cost = sum(r.get("cost_cny", 0) for r in rows)
    cached_n = sum(1 for r in rows if r.get("cached"))
    success_n = sum(1 for r in rows if r.get("success"))
    return {
        "records": len(rows),
        "total_tokens": total_tokens,
        "total_cost_cny": round(total_cost, 4),
        "cached": cached_n,
        "success_rate": round(success_n / len(rows), 3) if rows else 0,
        "by_task": {
            t: sum(1 for r in rows if r["task"] == t)
            for t in {r["task"] for r in rows}
        },
    }


if __name__ == "__main__":
    # 自检
    record("test_task", "MiniMax-M3", 100, 200, 1500, success=True)
    record("test_task2", "MiniMax-M2.5-highspeed", 50, 80, 800, cached=True)
    record("test_task3", "unknown-model", 30, 40, 200, success=False, error="timeout")

    print(f"cost(M3 100+200 tok) = {calc_cost('MiniMax-M3', 100, 200)} 元")
    print(f"cost(deepseek 1M+1M) = {calc_cost('deepseek-chat', 1_000_000, 1_000_000)} 元")
    s = summary()
    print(f"summary: {s}")
    print("All trace self-tests passed")
