"""MCP stdio 客户端：每次调用起一个子进程（无长连接、无状态残留）。

`MCPClientManager` 的 async 方法供 FastAPI 路由直接 await；`run_call` / `list_tools`
是同步包装，内部在新线程里 `asyncio.run`，避免与调用方已有事件循环冲突。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

MODULE = "ragm.mcp.server"
SOURCE_ROOT = Path(__file__).resolve().parents[2]


class MCPToolError(RuntimeError):
    """MCP 工具返回 is_error；`unknown_tool` 用于 API 映射 404。"""

    def __init__(self, message: str, *, unknown_tool: bool = False) -> None:
        super().__init__(message)
        self.unknown_tool = unknown_tool


def _content_texts(result: Any) -> list[str]:
    texts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            texts.append(text)
    return texts


def extract_tool_result(result: Any) -> Any:
    """把 CallToolResult 归一为 JSON 可序列化的结果；错误抛 MCPToolError。"""
    if getattr(result, "is_error", False):
        message = "\n".join(_content_texts(result)) or "MCP tool failed"
        raise MCPToolError(message, unknown_tool=message.startswith("Unknown tool:"))
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return structured
    texts = _content_texts(result)
    if not texts:
        return None
    if len(texts) == 1:
        try:
            return json.loads(texts[0])
        except (TypeError, ValueError):
            return texts[0]
    return texts


class MCPClientManager:
    """把 stdio MCP Server 作为一次性子进程调用。

    子进程环境显式设置 PYTHONPATH=<当前 worktree 的 src> 与 MCP_SANDBOX_ROOT，
    保证测试（共享主 venv）与生产（本 worktree）都导入正确的源码。
    """

    def __init__(
        self,
        sandbox_root: str | Path,
        *,
        python_executable: str | None = None,
        module: str = MODULE,
        timeout_s: float = 30.0,
        env: dict[str, str] | None = None,
    ) -> None:
        raw = str(sandbox_root or "").strip()
        if not raw:
            raise ValueError("MCP sandbox root is required")
        self.sandbox_root = Path(raw).expanduser().resolve()
        self.python_executable = python_executable or sys.executable
        self.module = module
        self.timeout_s = float(timeout_s)
        self._env_overrides = {str(k): str(v) for k, v in (env or {}).items()}
        self._source_root = SOURCE_ROOT

    def _server_parameters(self) -> Any:
        from mcp import StdioServerParameters

        env = {**os.environ, **self._env_overrides}
        source = str(self._source_root)
        existing = env.get("PYTHONPATH", "")
        if existing:
            parts = existing.split(os.pathsep)
            env["PYTHONPATH"] = existing if source in parts else source + os.pathsep + existing
        else:
            env["PYTHONPATH"] = source
        env["MCP_SANDBOX_ROOT"] = str(self.sandbox_root)
        return StdioServerParameters(command=self.python_executable, args=["-m", self.module], env=env)

    async def list_tools_async(self) -> list[dict[str, Any]]:
        from mcp import Client

        async with Client(self._server_parameters(), read_timeout_seconds=self.timeout_s) as client:
            listing = await client.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": dict(tool.input_schema or {}),
            }
            for tool in listing.tools
        ]

    async def call_tool_async(self, name: str, args: dict[str, Any] | None = None) -> Any:
        from mcp import Client

        async with Client(self._server_parameters(), read_timeout_seconds=self.timeout_s) as client:
            result = await client.call_tool(name, dict(args or {}))
        return extract_tool_result(result)

    def list_tools(self) -> list[dict[str, Any]]:
        return self._run_sync(self.list_tools_async)

    def run_call(self, name: str, args: dict[str, Any] | None = None) -> Any:
        return self._run_sync(lambda: self.call_tool_async(name, args))

    def _run_sync(self, factory: Any) -> Any:
        box: dict[str, Any] = {}

        def target() -> None:
            try:
                box["value"] = asyncio.run(factory())
            except BaseException as exc:  # noqa: BLE001 - 原样跨线程回抛
                box["error"] = exc

        thread = threading.Thread(target=target, name="ragm-mcp-call", daemon=True)
        thread.start()
        thread.join(self.timeout_s + 5.0)
        if thread.is_alive():
            raise TimeoutError(f"MCP call timed out after {self.timeout_s:.1f}s")
        if "error" in box:
            raise box["error"]
        return box.get("value")
