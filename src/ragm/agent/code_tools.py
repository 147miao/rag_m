"""Agent 代码工具：search_code / get_symbol / impact_analysis / repo_overview。

本模块不修改 agent/tools.py（主 agent 负责把 CODE_TOOL_SCHEMAS 与 CodeToolExecutor
接入 AgentGraph）；execute() 统一返回 {"ok": bool, "tool": name, ...}，异常转为 ok=False。
"""
from __future__ import annotations

import time
from typing import Any

from ..code.service import CodeService

CODE_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": (
                "在已索引的本地代码库中检索代码片段（SQLite FTS5），返回 path、qualname、"
                "起止行号与代码文本；适合定位实现、查找符号定义。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "查询词，优先英文标识符 / 路径 / 错误信息"},
                    "k": {"type": "integer", "description": "返回条数，默认 8，最大 50"},
                    "repo_id": {"type": "string", "description": "限定仓库 ID；省略则跨仓库检索"},
                    "path_prefix": {"type": "string", "description": "路径前缀过滤，如 src/ragm/"},
                    "kind": {"type": "string", "description": "符号类型过滤，如 function / class / method"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_symbol",
            "description": "按限定名精确获取符号定义（含完整代码与行号）；同名时可用 path 限定文件。",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_id": {"type": "string", "description": "仓库 ID"},
                    "qualname": {"type": "string", "description": "限定名，如 ragm.rag.graph.build_graph"},
                    "path": {"type": "string", "description": "可选：限定文件路径"},
                },
                "required": ["repo_id", "qualname"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "impact_analysis",
            "description": (
                "返回直接反向 import 指定文件的文件列表（文件级、启发式，不承诺精确调用链或传递闭包）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_id": {"type": "string", "description": "仓库 ID"},
                    "path": {"type": "string", "description": "仓库内相对路径，如 src/ragm/rag/graph.py"},
                },
                "required": ["repo_id", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "repo_overview",
            "description": "返回仓库概览：文件数、符号数、符号类型分布与热门路径。",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_id": {"type": "string", "description": "仓库 ID"},
                },
                "required": ["repo_id"],
            },
        },
    },
]


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class CodeToolExecutor:
    """执行代码工具；依赖 CodeService，便于单测注入 fake。"""

    def __init__(self, service: CodeService) -> None:
        self.service = service

    def search_code(
        self,
        query: str,
        k: int = 8,
        repo_id: str | None = None,
        path_prefix: str | None = None,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.service.search_code(
            query,
            k=k,
            repo_id=repo_id,
            path_prefix=path_prefix,
            kind=kind,
        )

    def get_symbol(self, repo_id: str, qualname: str, path: str | None = None) -> dict[str, Any] | None:
        return self.service.get_symbol(repo_id, qualname, path=path)

    def impact_analysis(self, repo_id: str, path: str) -> dict[str, Any]:
        return self.service.impact(repo_id, path)

    def repo_overview(self, repo_id: str) -> dict[str, Any]:
        return self.service.overview(repo_id)

    def execute(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """执行工具。未知工具 / 服务异常均返回 ok=False，不向上抛。"""
        started = time.perf_counter()
        tool = str(name or "")
        payload = args if isinstance(args, dict) else {}
        try:
            if tool == "search_code":
                query = str(payload.get("query") or "")
                hits = self.search_code(
                    query,
                    k=_to_int(payload.get("k"), 8),
                    repo_id=payload.get("repo_id"),
                    path_prefix=payload.get("path_prefix"),
                    kind=payload.get("kind"),
                )
                out: dict[str, Any] = {
                    "ok": True,
                    "tool": tool,
                    "query": query,
                    "count": len(hits),
                    "hits": hits,
                }
            elif tool == "get_symbol":
                symbol = self.get_symbol(
                    str(payload.get("repo_id") or ""),
                    str(payload.get("qualname") or ""),
                    str(payload.get("path") or "") or None,
                )
                if symbol is None:
                    out = {
                        "ok": False,
                        "tool": tool,
                        "error": f"symbol not found: {payload.get('qualname')}",
                    }
                else:
                    out = {"ok": True, "tool": tool, "symbol": symbol}
            elif tool == "impact_analysis":
                impact = self.impact_analysis(
                    str(payload.get("repo_id") or ""),
                    str(payload.get("path") or ""),
                )
                out = {"ok": True, "tool": tool, **impact}
            elif tool == "repo_overview":
                overview = self.repo_overview(str(payload.get("repo_id") or ""))
                out = {"ok": True, "tool": tool, "overview": overview}
            else:
                out = {"ok": False, "tool": tool, "error": f"unknown tool: {tool}"}
        except Exception as exc:  # noqa: BLE001 - 工具错误应回给 LLM，而不是中断 agent
            out = {"ok": False, "tool": tool, "error": f"{type(exc).__name__}: {exc}"}
        out["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return out

    def run(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """兼容 ToolExecutor.run 的调用习惯。"""
        return self.execute(name, args)
