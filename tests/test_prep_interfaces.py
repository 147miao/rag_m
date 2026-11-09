"""prep/v1.1 接口层契约测试：未实现的骨架路由返回 501，新增 schemas 可构造。"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from ragm.api.main import create_app
from ragm.api.schemas import (
    Answer,
    IngestRequest,
    JobInfo,
    JobListResponse,
    McpCallRequest,
    McpCallResponse,
    McpConfirmRequest,
    McpToolInfo,
    QueryRequest,
    QueryResponse,
    SessionCreateRequest,
    SessionDetail,
    SessionInfo,
    SessionMessage,
    Stats,
)
from ragm.config import Settings, get_settings


@pytest.fixture
async def client(monkeypatch):
    # 测试只验证路由契约，固定规则分类器，避免加载 LoRA / 外部服务
    monkeypatch.setenv("CLASSIFIER_BACKEND", "rules")
    monkeypatch.setenv("RAG_BACKEND", "demo")
    get_settings.cache_clear()
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    get_settings.cache_clear()


# 会话接口已由 feat/session 实现（见 tests/test_conversation.py），此处只保留其余骨架路由。
ROUTE_CASES = [
    ("POST", "/api/ingest", {"src": "docs/a.md"}),
    ("GET", "/api/jobs/j-test", None),
    ("GET", "/api/jobs", None),
    ("POST", "/api/jobs/j-test/retry", None),
]


@pytest.mark.parametrize(("method", "path", "json_body"), ROUTE_CASES)
async def test_prep_routes_return_501(client: AsyncClient, method: str, path: str, json_body) -> None:
    response = await client.request(method, path, json=json_body)
    assert response.status_code == 501
    body = response.json()
    assert body["error"]["code"] == "http_501"
    assert "not implemented" in body["error"]["message"]


# MCP 路由已由 feat/mcp 实现：默认 MCP_ENABLED=false 时返回 503，不再是 501 骨架；
# 启用/调用/确认流程见 tests/test_mcp.py。
MCP_ROUTE_CASES = [
    ("GET", "/api/mcp/tools", None),
    ("POST", "/api/mcp/call", {"tool": "read_file", "args": {"path": "a.txt"}}),
    ("POST", "/api/mcp/confirm", {"token": "tok"}),
]


@pytest.mark.parametrize(("method", "path", "json_body"), MCP_ROUTE_CASES)
async def test_mcp_routes_disabled_return_503(client: AsyncClient, method: str, path: str, json_body) -> None:
    response = await client.request(method, path, json=json_body)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "http_503"


async def test_prep_routes_registered(client: AsyncClient) -> None:
    schema = (await client.get("/openapi.json")).json()
    expected = {
        "/api/sessions",
        "/api/sessions/{session_id}",
        "/api/mcp/tools",
        "/api/mcp/call",
        "/api/mcp/confirm",
        "/api/ingest",
        "/api/jobs",
        "/api/jobs/{job_id}",
        "/api/jobs/{job_id}/retry",
    }
    assert expected <= set(schema["paths"])


def test_session_schemas_constructible() -> None:
    now = datetime.now(UTC)
    create = SessionCreateRequest()
    info = SessionInfo(id="s1", title="新会话", created_at=now, updated_at=now, message_count=1)
    detail = SessionDetail(
        **info.model_dump(),
        messages=[SessionMessage(role="user", content="鸣人的师父是谁？", created_at=now)],
    )
    assert create.title is None
    assert detail.messages[0].role == "user"
    assert detail.model_dump()["message_count"] == 1


def test_mcp_schemas_constructible() -> None:
    tool = McpToolInfo(
        name="write_note",
        description="写笔记",
        requires_confirmation=True,
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
    )
    call = McpCallRequest(tool="write_note", args={"path": "notes/a.md"})
    pending = McpCallResponse(ok=True, pending=True, token="tok")
    done = McpCallResponse(ok=True, result={"written": True})
    confirm = McpConfirmRequest(token="tok")
    assert tool.requires_confirmation is True
    assert call.args["path"] == "notes/a.md"
    assert pending.token == "tok"
    assert done.result == {"written": True}
    assert confirm.token == "tok"


def test_job_and_ingest_schemas_constructible() -> None:
    now = datetime.now(UTC)
    queued = JobInfo(id="j1", status="queued", attempts=0, max_attempts=3, created_at=now, updated_at=now)
    failed = JobInfo(
        id="j2", status="failed", attempts=3, max_attempts=3, error="boom", created_at=now, updated_at=now
    )
    listing = JobListResponse(jobs=[queued, failed])
    ingest = IngestRequest(src="docs/a.md", collection="rag_chunks", recreate=False)
    assert [job.status for job in listing.jobs] == ["queued", "failed"]
    assert listing.jobs[1].error == "boom"
    assert ingest.recreate is False


def test_query_schemas_v11_fields() -> None:
    now = datetime.now(UTC)
    request = QueryRequest(question="他的师父是谁？", session_id="s1")
    assert request.session_id == "s1"
    assert QueryRequest(question="独立问题").session_id is None

    response = QueryResponse(
        id="q1",
        question="他的师父是谁？",
        route="simple",
        confidence=0.9,
        reason="test",
        answer=Answer(),
        citations=[1],
        evidence=[],
        trace=[],
        stats=Stats(latency_ms=1, tokens_in=2, tokens_out=3, cost_cny=0.0, citation_check="ok"),
        generated_at=now,
        session_id="s1",
        rewritten_question="鸣人的师父是谁？",
        subquestions=["鸣人的师父是谁？", "他有什么作品？"],
        provider="opencode-go",
    )
    assert response.session_id == "s1"
    assert response.rewritten_question == "鸣人的师父是谁？"
    assert len(response.subquestions) == 2
    assert response.provider == "opencode-go"


def test_settings_v11_defaults(monkeypatch) -> None:
    for key in (
        "DATA_ROOT",
        "SESSION_DB_PATH",
        "SESSION_MAX_TURNS",
        "SESSION_MAX_CHARS",
        "SESSION_REWRITE_ENABLED",
        "MCP_ENABLED",
        "MCP_SANDBOX_ROOT",
        "MCP_CONFIRM_TTL_S",
        "LLM_PROVIDERS",
        "LLM_CHAT_TIMEOUT",
        "LLM_FIRST_TOKEN_TIMEOUT",
        "JOBS_DB_PATH",
        "JOBS_WORKER",
        "JOBS_MAX_ATTEMPTS",
    ):
        monkeypatch.delenv(key, raising=False)

    settings = Settings(_env_file=None, data_root=Path("/tmp/ragm-prep"))
    assert settings.session_db_path == ""
    assert settings.session_store_path == Path("/tmp/ragm-prep/data/sessions.sqlite")
    assert settings.session_max_turns == 6
    assert settings.session_max_chars == 4000
    assert settings.session_rewrite_enabled is True
    assert settings.mcp_enabled is False
    assert settings.mcp_sandbox_root == ""
    assert settings.mcp_confirm_ttl_s == 60
    assert settings.llm_chat_timeout == 60.0
    assert settings.llm_first_token_timeout == 15.0
    assert settings.jobs_db_path == ""
    assert settings.jobs_store_path == Path("/tmp/ragm-prep/data/jobs.sqlite")
    assert settings.jobs_worker is True
    assert settings.jobs_max_attempts == 3

    providers = settings.resolved_llm_providers
    assert [item["name"] for item in providers] == ["opencode-go", "ollama"]
    assert providers[0]["model"] == "deepseek-v4-flash"
    assert providers[1]["model"] == "gemma4:31b-cloud"


def test_settings_explicit_paths_and_provider_json(monkeypatch) -> None:
    for key in ("SESSION_DB_PATH", "JOBS_DB_PATH", "LLM_PROVIDERS"):
        monkeypatch.delenv(key, raising=False)

    settings = Settings(
        _env_file=None,
        session_db_path="/tmp/s.sqlite",
        jobs_db_path="/tmp/j.sqlite",
        llm_providers='[{"name":"fake","base_url":"http://x/v1","model":"m"}]',
    )
    assert settings.session_store_path == Path("/tmp/s.sqlite")
    assert settings.jobs_store_path == Path("/tmp/j.sqlite")
    assert settings.resolved_llm_providers == [
        {"name": "fake", "base_url": "http://x/v1", "model": "m"},
    ]
