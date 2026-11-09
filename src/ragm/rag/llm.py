"""OpenAI 兼容的 LLM 客户端（Ollama / DeepSeek 均可）。"""
from __future__ import annotations

import json
from collections.abc import Iterator

import httpx

from ..observability.tracing import traceable


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, base_url: str, model: str, api_key: str = "ollama", temperature: float = 0.0, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout

    def _client(self) -> httpx.Client:
        local = "127.0.0.1" in self.base_url or "localhost" in self.base_url
        return httpx.Client(timeout=self.timeout, trust_env=not local)

    def _payload(self, messages: list[dict], max_tokens: int, stream: bool, json_mode: bool) -> dict:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    @traceable(run_type="llm", name="llm_chat")
    def chat(self, messages: list[dict], max_tokens: int = 1024, json_mode: bool = False) -> dict:
        with self._client() as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=self._payload(messages, max_tokens, False, json_mode),
            )
            if resp.status_code >= 400:
                raise LLMError(f"LLM {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
        usage = data.get("usage") or {}
        return {
            "text": data["choices"][0]["message"]["content"],
            "tokens_in": int(usage.get("prompt_tokens") or 0),
            "tokens_out": int(usage.get("completion_tokens") or 0),
        }

    @traceable(run_type="llm", name="llm_chat_tools")
    def chat_tools(self, messages: list[dict], tools: list[dict], max_tokens: int = 1024) -> dict:
        """带工具调用的 chat：返回 text / tool_calls / usage。"""
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "tools": tools,
            "tool_choice": "auto",
        }
        with self._client() as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=body,
            )
            if resp.status_code >= 400:
                raise LLMError(f"LLM {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
        message = data["choices"][0]["message"]
        usage = data.get("usage") or {}
        return {
            "text": message.get("content") or "",
            "tool_calls": message.get("tool_calls") or [],
            "tokens_in": int(usage.get("prompt_tokens") or 0),
            "tokens_out": int(usage.get("completion_tokens") or 0),
        }

    def stream_chat(self, messages: list[dict], max_tokens: int = 1024) -> Iterator[str]:
        with self._client() as client, client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=self._payload(messages, max_tokens, True, False),
        ) as resp:
            if resp.status_code >= 400:
                raise LLMError(f"LLM {resp.status_code}: {resp.read()[:300]}")
            for line in resp.iter_lines():
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    delta = json.loads(chunk)["choices"][0]["delta"].get("content")
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if delta:
                    yield delta