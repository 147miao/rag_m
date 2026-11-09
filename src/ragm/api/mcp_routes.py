"""MCP API：列出工具、调用工具、敏感工具二次确认。

- `GET /api/mcp/tools`：工具元数据，`requires_confirmation` 由 SENSITIVE_TOOLS 标注；
- `POST /api/mcp/call`：只读工具直接执行；敏感工具只签发 token（202 pending），不执行；
- `POST /api/mcp/confirm`：原子消费 token 后执行，过期/已用/未知分别 410/409/404。

MCP 默认关闭（MCP_ENABLED=false），未装配 manager 时三个端点均返回 503。
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ..config import get_settings
from ..mcp.client import MCPToolError
from ..mcp.confirmation import SENSITIVE_TOOLS, ConfirmationError, ConfirmationStore
from .schemas import McpCallRequest, McpCallResponse, McpConfirmRequest, McpToolInfo

logger = logging.getLogger("ragm.api.mcp")

router = APIRouter(prefix="/api", tags=["mcp"])


def _manager(request: Request) -> Any:
    manager = getattr(request.app.state, "mcp", None)
    if manager is None:
        raise HTTPException(
            status_code=503,
            detail="MCP is disabled or unavailable (set MCP_ENABLED=true and MCP_SANDBOX_ROOT)",
        )
    return manager


def _confirmations(request: Request) -> ConfirmationStore:
    store = getattr(request.app.state, "mcp_confirmations", None)
    if store is None:
        store = ConfirmationStore(ttl_s=get_settings().mcp_confirm_ttl_s)
        request.app.state.mcp_confirmations = store
    return store


async def _execute(manager: Any, tool: str, args: dict[str, Any]) -> Any:
    try:
        return await manager.call_tool_async(tool, args)
    except MCPToolError as exc:
        raise HTTPException(
            status_code=404 if exc.unknown_tool else 400,
            detail=str(exc),
        ) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=f"MCP call timed out: {exc}") from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("MCP tool %s failed", tool)
        raise HTTPException(status_code=502, detail=f"MCP call failed: {exc}") from exc


@router.get("/mcp/tools", response_model=list[McpToolInfo], summary="列出 MCP 工具")
async def list_mcp_tools(request: Request) -> list[McpToolInfo]:
    manager = _manager(request)
    try:
        tools = await manager.list_tools_async()
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("failed to list MCP tools")
        raise HTTPException(status_code=502, detail=f"failed to list MCP tools: {exc}") from exc
    return [
        McpToolInfo(
            name=str(tool.get("name", "")),
            description=str(tool.get("description", "")),
            requires_confirmation=str(tool.get("name", "")) in SENSITIVE_TOOLS,
            input_schema=tool.get("input_schema") or {},
        )
        for tool in tools
    ]


@router.post(
    "/mcp/call",
    response_model=McpCallResponse,
    summary="调用 MCP 工具（敏感工具返回 pending + token）",
    responses={202: {"model": McpCallResponse, "description": "等待确认"}},
)
async def call_mcp_tool(req: McpCallRequest, request: Request) -> Any:
    manager = _manager(request)
    if req.tool in SENSITIVE_TOOLS:
        token = _confirmations(request).issue(req.tool, req.args)
        payload = McpCallResponse(ok=False, pending=True, token=token)
        return JSONResponse(status_code=202, content=payload.model_dump())
    result = await _execute(manager, req.tool, req.args)
    return McpCallResponse(ok=True, result=result)


@router.post("/mcp/confirm", response_model=McpCallResponse, summary="确认并执行待确认的 MCP 调用")
async def confirm_mcp_call(req: McpConfirmRequest, request: Request) -> McpCallResponse:
    manager = _manager(request)
    try:
        call = _confirmations(request).consume_call(req.token)
    except ConfirmationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    result = await _execute(manager, call.tool, call.args)
    return McpCallResponse(ok=True, result=result)


__all__ = ["router"]
