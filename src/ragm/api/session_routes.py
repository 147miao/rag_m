"""会话 API（v1.1 W1）：创建 / 详情 / 删除，存储走 SessionStore。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..config import get_settings
from ..conversation.store import SessionStore
from .schemas import SessionCreateRequest, SessionDetail, SessionInfo, SessionMessage

router = APIRouter(prefix="/api", tags=["sessions"])


def session_store(request: Request) -> SessionStore:
    """按应用实例懒加载会话存储；测试可通过 SESSION_DB_PATH 指向 tmp_path。"""
    store = getattr(request.app.state, "session_store", None)
    if store is None:
        store = SessionStore(get_settings().session_store_path)
        request.app.state.session_store = store
    return store


def _not_found(session_id: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"session not found: {session_id}")


def _detail_payload(record: dict) -> SessionDetail:
    return SessionDetail(
        id=record["id"],
        title=record["title"],
        created_at=record["created_at"],
        updated_at=record["updated_at"],
        message_count=record["message_count"],
        messages=[
            SessionMessage(role=message["role"], content=message["content"], created_at=message["created_at"])
            for message in record["messages"]
        ],
    )


@router.post("/sessions", response_model=SessionInfo, status_code=201, summary="创建会话")
def create_session(req: SessionCreateRequest, request: Request) -> SessionInfo:
    record = session_store(request).create_session(req.title)
    return SessionInfo(**record)


@router.get("/sessions/{session_id}", response_model=SessionDetail, summary="会话详情（含消息）")
def get_session(session_id: str, request: Request) -> SessionDetail:
    record = session_store(request).get_session(session_id)
    if record is None:
        raise _not_found(session_id)
    return _detail_payload(record)


@router.delete("/sessions/{session_id}", status_code=204, summary="删除会话")
def delete_session(session_id: str, request: Request) -> None:
    if not session_store(request).delete_session(session_id):
        raise _not_found(session_id)
