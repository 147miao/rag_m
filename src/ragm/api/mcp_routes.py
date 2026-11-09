"""MCP API 契约骨架（v1.1 W2，owner: feat/mcp）。

prep 阶段只落路径/请求响应模型，全部返回 501；敏感工具确认流程由 feature 分支实现。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .schemas import McpCallRequest, McpCallResponse, McpConfirmRequest, McpToolInfo

router = APIRouter(prefix="/api", tags=["mcp"])

_NOT_IMPLEMENTED = "MCP API not implemented yet (prep/v1.1 skeleton, owner: feat/mcp)"


def _todo() -> HTTPException:
    return HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)


@router.get("/mcp/tools", response_model=list[McpToolInfo], summary="列出 MCP 工具")
async def list_mcp_tools() -> list[McpToolInfo]:
    raise _todo()


@router.post("/mcp/call", response_model=McpCallResponse, summary="调用 MCP 工具（敏感工具返回 pending + token）")
async def call_mcp_tool(req: McpCallRequest) -> McpCallResponse:
    raise _todo()


@router.post("/mcp/confirm", response_model=McpCallResponse, summary="确认并执行待确认的 MCP 调用")
async def confirm_mcp_call(req: McpConfirmRequest) -> McpCallResponse:
    raise _todo()
