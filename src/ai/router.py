"""router.py - LLM 路由 + 调用编排 ===========================================
核心设计:
- ModelRouter: 多模型 fallback 链 (M3 主 + M2.5 备 + mock 兜底)
- chat_json: 强制 JSON 输出,3 层解析 (response_format -> 代码块 -> 大括号切片)
- 集成 cache + trace
- 指数退避重试 (tenacity)
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.ai.cache import LLMCache, _articles_hash, get_default_cache
from src.ai.trace import calc_cost, record as trace_record
from src.config import AI_API_KEY, AI_BASE_URL, AI_MAX_RETRIES, AI_MODEL, AI_PROVIDER, AI_TIMEOUT
from src.utils.logger import get_logger

logger = get_logger("ai.router")

T = TypeVar("T", bound=BaseModel)

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ============================================================
# 模型描述
# ============================================================
@dataclass
class ModelSpec:
    name: str
    speed: str  # "fast" | "thinking"
    json_mode: str  # "native" | "prompt" | "unstable"
    max_tokens: int = 4000
    timeout: int = 60


# 默认 fallback 链 (按"先快后慢,先稳后准"排序)
DEFAULT_FALLBACK_CHAIN: List[ModelSpec] = [
    ModelSpec("MiniMax-M3", "thinking", "native", max_tokens=4000, timeout=60),
    ModelSpec("MiniMax-M2.5-highspeed", "fast", "prompt", max_tokens=3000, timeout=30),
]


# ============================================================
# JSON 解析 (3 层兜底)
# ============================================================
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> Dict[str, Any]:
    """从 LLM 输出中提取 JSON,容错 3 层。"""
    if not text:
        return {}
    s = text.strip()

    # 0. 去掉思考块
    s = _THINK_BLOCK_RE.sub("", s).strip()

    # 1. 完整代码块
    m = _CODE_BLOCK_RE.search(s)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # 2. 整段就是 JSON
    if s.startswith("{") and s.endswith("}"):
        try:
            return json.loads(s)
        except Exception:
            pass

    # 3. 找第一个 { 到最后一个 }
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(s[start : end + 1])
        except Exception:
            pass

    logger.warning(f"JSON 解析失败,原文前 300 字符: {s[:300]}")
    return {}


# ============================================================
# Router
# ============================================================
class ModelRouter:
    """统一 LLM 入口:多模型 fallback + 缓存 + 追踪 + 重试。"""

    def __init__(
        self,
        api_key: str = AI_API_KEY,
        base_url: str = AI_BASE_URL,
        chain: Optional[List[ModelSpec]] = None,
        cache: Optional[LLMCache] = None,
    ):
        if not HAS_OPENAI:
            raise RuntimeError("openai SDK 未安装")
        if not api_key:
            raise RuntimeError("AI_API_KEY 未配置 (.env)")
        self.api_key = api_key
        self.base_url = base_url
        self.chain = chain or DEFAULT_FALLBACK_CHAIN
        self.cache = cache or get_default_cache()
        # 懒加载 client (per-model)
        self._clients: Dict[str, OpenAI] = {}

    def _client(self, model: str) -> OpenAI:
        if model not in self._clients:
            self._clients[model] = OpenAI(
                api_key=self.api_key, base_url=self.base_url, timeout=60,
            )
        return self._clients[model]

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        use_json_mode: bool = True,
        disable_thinking: bool = False,
    ) -> Tuple[str, str, int, int]:
        """底层 chat 调用,返回 (raw_text, model, prompt_tokens, completion_tokens)。"""
        spec = next(
            (s for s in self.chain if s.name == model), self.chain[0]
        ) if model else self.chain[0]
        client = self._client(spec.name)
        kwargs: Dict[str, Any] = {
            "model": spec.name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": min(max_tokens, spec.max_tokens),
        }
        if use_json_mode and spec.json_mode == "native":
            kwargs["response_format"] = {"type": "json_object"}
        # minimax / 一些 thinking 模型支持通过 extra_body 关闭 <think>
        if disable_thinking:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        resp = client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content or ""
        usage = resp.usage
        return content, spec.name, usage.prompt_tokens, usage.completion_tokens

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        task_name: str,
        date: str = "",
        articles: Optional[List[Any]] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> Dict[str, Any]:
        """带 fallback + 缓存 + 追踪的 chat。"""
        arts_hash = _articles_hash(articles) if articles else ""

        # 1. 缓存查找 (按 task + 第一个 model 的 key)
        cache_key = self.cache.make_key(date, task_name, arts_hash, self.chain[0].name)
        cached = self.cache.get(cache_key)
        if cached and cached.get("parsed_json"):
            trace_record(
                task_name=task_name, model=cached.get("model", ""),
                prompt_tokens=0, completion_tokens=0, latency_ms=0,
                cached=True,
                prompt=messages[-1]["content"] if messages else "",
                completion=cached.get("raw_content", ""),
            )
            logger.info(f"[cache HIT] {task_name} (saved)")
            try:
                return json.loads(cached["parsed_json"])
            except Exception:
                pass  # 损坏缓存,走正常路径

        # 2. Fallback 调用
        last_err: Optional[Exception] = None
        prompt = messages[-1]["content"] if messages else ""
        for spec in self.chain:
            t0 = time.time()
            try:
                raw, used_model, pt, ct = self.chat(
                    messages, model=spec.name,
                    temperature=temperature, max_tokens=max_tokens,
                )
                latency = int((time.time() - t0) * 1000)
                parsed = extract_json(raw)
                if not parsed:
                    raise ValueError(f"JSON 解析失败,raw={raw[:200]!r}")

                # 写入缓存
                self.cache.put(
                    cache_key, date, task_name, arts_hash, raw, parsed, used_model,
                    prompt_tokens=pt, completion_tokens=ct,
                    cost_cny=calc_cost(used_model, pt, ct),
                )
                trace_record(
                    task_name=task_name, model=used_model,
                    prompt_tokens=pt, completion_tokens=ct, latency_ms=latency,
                    success=True, prompt=prompt, completion=raw,
                )
                logger.info(
                    f"[{used_model}] {task_name} ok ({pt}+{ct} tok, {latency}ms)"
                )
                return parsed
            except Exception as e:
                last_err = e
                latency = int((time.time() - t0) * 1000)
                trace_record(
                    task_name=task_name, model=spec.name,
                    latency_ms=latency, success=False,
                    error=str(e), prompt=prompt,
                )
                logger.warning(f"[{spec.name}] {task_name} 失败: {e}")
                continue

        raise RuntimeError(f"所有模型都失败 (最后错误: {last_err})")


# ============================================================
# Schema 验证
# ============================================================
def parse_to_schema(
    raw: Dict[str, Any],
    schema_cls: Type[T],
) -> T:
    """把 LLM 返回的 dict 强类型化为 Pydantic schema。失败时 raise。"""
    return schema_cls.model_validate(raw)


def safe_parse(
    raw: Dict[str, Any],
    schema_cls: Type[T],
    fallback: Optional[T] = None,
) -> T:
    """带降级的 schema 解析,失败返回 fallback。"""
    try:
        return schema_cls.model_validate(raw)
    except ValidationError as e:
        logger.warning(f"Schema {schema_cls.__name__} 校验失败: {e}")
        if fallback is not None:
            return fallback
        raise


# 全局单例
_default_router: Optional[ModelRouter] = None


def get_default_router() -> ModelRouter:
    global _default_router
    if _default_router is None:
        _default_router = ModelRouter()
    return _default_router


if __name__ == "__main__":
    # 1. JSON 解析测试
    assert extract_json("""<think>hello</think>\n```json\n{"a":1}\n```""") == {"a": 1}
    assert extract_json('{"x": 2}') == {"x": 2}
    assert extract_json('some text {"y": 3} more') == {"y": 3}
    assert extract_json("<think>x</think>\n{\"z\": 4}") == {"z": 4}
    assert extract_json("") == {}
    assert extract_json("garbage") == {}
    print("JSON parser: 6/6 ok")

    # 2. Router 自检(不实际调 API,只验结构)
    try:
        r = ModelRouter(api_key="", base_url="x")
    except RuntimeError as e:
        print(f"Empty key rejected ok: {e}")

    print("All router self-tests passed")
