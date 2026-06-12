"""client — OpenAI 兼容协议 LLM 客户端 ==============================================
支持 DeepSeek / Qwen / OpenAI / 自定义 OpenAI-兼容服务。

通过 .env 配置:
  AI_PROVIDER=deepseek|openai|qwen|custom
  AI_API_KEY=...
  AI_BASE_URL=https://api.deepseek.com/v1
  AI_MODEL=deepseek-chat
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from src.config import (
    AI_API_KEY,
    AI_BASE_URL,
    AI_MAX_RETRIES,
    AI_MODEL,
    AI_PROVIDER,
    AI_TIMEOUT,
)
from src.utils.logger import get_logger

logger = get_logger("ai.client")

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logger.warning("openai SDK 未安装,LLM 功能不可用")


class LLMClient:
    """统一的 LLM 客户端封装,支持 JSON 输出与重试。"""

    def __init__(
        self,
        provider: str = AI_PROVIDER,
        api_key: str = AI_API_KEY,
        base_url: str = AI_BASE_URL,
        model: str = AI_MODEL,
        timeout: int = AI_TIMEOUT,
        max_retries: int = AI_MAX_RETRIES,
    ) -> None:
        if not HAS_OPENAI:
            raise RuntimeError("openai SDK 未安装,请运行: pip install openai")
        if not api_key:
            raise RuntimeError(
                "AI_API_KEY 未配置。请复制 .env.example 为 .env 并填入 API Key。"
            )

        self.provider = provider
        self.model = model
        self.max_retries = max_retries

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        logger.info(f"LLM 客户端已初始化: provider={provider}, model={model}, base_url={base_url}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2000,
        response_format: Optional[Dict[str, str]] = None,
    ) -> str:
        """发送对话请求,返回纯文本响应(已重试)。"""
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if response_format:
                    kwargs["response_format"] = response_format

                resp = self.client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content or ""
                return content
            except Exception as e:
                last_err = e
                wait = 2 ** attempt
                logger.warning(
                    f"LLM 调用第 {attempt}/{self.max_retries} 次失败: {e},等待 {wait}s"
                )
                time.sleep(wait)

        raise RuntimeError(f"LLM 调用失败 {self.max_retries} 次: {last_err}")

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> Dict[str, Any]:
        """发送请求并解析为 JSON dict。失败时返回空 dict。"""
        try:
            # 优先尝试 response_format=json_object(DeepSeek/Qwen/OpenAI 都支持)
            content = self.chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception:
            # 回退:不指定格式,要求模型自己输出 JSON,再手动解析
            content = self.chat(messages, temperature=temperature, max_tokens=max_tokens)

        return self._parse_json(content)

    @staticmethod
    def _parse_json(content: str) -> Dict[str, Any]:
        """从 LLM 输出中提取 JSON,容错处理 markdown 包裹。"""
        if not content:
            return {}
        text = content.strip()

        # 去掉 markdown ```json ... ``` 包裹
        if text.startswith("```"):
            lines = text.split("\n")
            # 去掉首尾 ```
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # 尝试找第一个 { 到最后一个 } 的子串
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass
            logger.warning(f"JSON 解析失败,原始内容前 200 字符: {text[:200]}")
            return {}


# 简易单例(供 analyzer 调用)
_default_client: Optional[LLMClient] = None


def get_default_client() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client