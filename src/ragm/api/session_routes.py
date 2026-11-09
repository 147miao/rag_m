"""会话 API 契约骨架（v1.1 W1，owner: feat/session）。

prep 阶段只落路径/请求响应模型，全部返回 501；业务实现由 feature 分支替换。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .schemas import SessionCreateRequest, SessionDetail, SessionInfo

router = APIRouter(prefix="/api", tags=["sessions"])

_NOT_IMPLEMENTED = "session API not implemented yet (prep/v1.1 skeleton, owner: feat/session)"


def _todo() -> HTTPException:
    return HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)


@router.post("/sessions", response_model=SessionInfo, status_code=201, summary="创建会话")
async def create_session(req: SessionCreateRequest) -> SessionInfo:
    raise _todo()


@router.get("/sessions/{session_id}", response_model=SessionDetail, summary="会话详情（含消息）")
async def get_session(session_id: str) -> SessionDetail:
    raise _todo()


@router.delete("/sessions/{session_id}", status_code=204, summary="删除会话")
async def delete_session(session_id: str) -> None:
    raise _todo()
