"""MCP 工具 + 敏感操作确认测试（不依赖网络 / 外部服务）。

覆盖验收 A4/A5/A6：
- in-memory `Client(MCPServer)` 验证 list_tools 与 read_file；
- 沙箱越界（相对路径 / 绝对路径 / symlink）拒绝；
- write_note 确认前不落盘，确认后写入；
- token 过期 410 / 已用 409 / 未知 404；
- 可选 stdio 子进程冒烟（真实 MCPClientManager）。
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from mcp import Client

from ragm.api.main import create_app
from ragm.config import get_settings
from ragm.mcp.client import MCPClientManager, extract_tool_result
from ragm.mcp.confirmation import ConfirmationStore, Expired, Unknown, Used
from ragm.mcp.server import create_server


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_SANDBOX_ROOT", str(tmp_path))
    (tmp_path / "hello.txt").write_text("hello mcp", encoding="utf-8")
    return tmp_path


@pytest.fixture
def server(sandbox):
    return create_server()


class InMemoryMCPManager:
    """测试用 manager：同一份 `Client(MCPServer)` 语义，但不拉子进程。"""

    def __init__(self, server: Any) -> None:
        self.server = server

    async def list_tools_async(self) -> list[dict[str, Any]]:
        async with Client(self.server) as client:
            listing = await client.list_tools()
        return [
            {"name": tool.name, "description": tool.description or "", "input_schema": tool.input_schema}
            for tool in listing.tools
        ]

    async def call_tool_async(self, name: str, args: dict[str, Any] | None = None) -> Any:
        async with Client(self.server) as client:
            result = await client.call_tool(name, dict(args or {}))
        return extract_tool_result(result)


@pytest.fixture
async def mcp_api(sandbox, monkeypatch):
    monkeypatch.setenv("CLASSIFIER_BACKEND", "rules")
    monkeypatch.setenv("RAG_BACKEND", "demo")
    monkeypatch.setenv("MCP_ENABLED", "true")
    get_settings.cache_clear()
    app = create_app()
    app.state.mcp = InMemoryMCPManager(create_server())
    app.state.mcp_confirmations = ConfirmationStore(ttl_s=60, secret=b"test-secret")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, app, sandbox
    get_settings.cache_clear()


async def _call(server: Any, name: str, args: dict[str, Any]) -> Any:
    async with Client(server) as client:
        return await client.call_tool(name, args)


def _payload(result: Any) -> dict[str, Any]:
    assert result.is_error is False
    return json.loads(result.content[0].text)


# ---------- server: A4 ----------


async def test_server_lists_three_tools(server) -> None:
    async with Client(server) as client:
        listing = await client.list_tools()
    tools = {tool.name: tool for tool in listing.tools}
    assert set(tools) == {"list_dir", "read_file", "write_note"}
    assert "read" in tools["read_file"].description.lower()
    assert tools["read_file"].input_schema["properties"]["max_chars"]["type"] == "integer"
    assert "path" in tools["write_note"].input_schema["required"]


async def test_read_file_inside_sandbox(server) -> None:
    result = await _call(server, "read_file", {"path": "hello.txt"})
    payload = _payload(result)
    assert payload["text"] == "hello mcp"
    assert payload["truncated"] is False
    assert payload["path"] == "hello.txt"


async def test_read_file_max_chars_truncates(server) -> None:
    payload = _payload(await _call(server, "read_file", {"path": "hello.txt", "max_chars": 5}))
    assert payload["text"] == "hello"
    assert payload["truncated"] is True


async def test_list_dir_returns_entries(server) -> None:
    payload = _payload(await _call(server, "list_dir", {"path": "."}))
    entries = {item["name"]: item for item in payload["entries"]}
    assert entries["hello.txt"]["type"] == "file"
    assert entries["hello.txt"]["size"] == len("hello mcp")


async def test_write_note_at_server_layer_creates_parents(server, sandbox) -> None:
    payload = _payload(await _call(server, "write_note", {"path": "notes/a.md", "text": "note"}))
    assert payload["written"] is True
    assert (sandbox / "notes" / "a.md").read_text(encoding="utf-8") == "note"


async def test_sandbox_escape_rejected(server, sandbox) -> None:
    outside = sandbox.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    for raw in ("../outside.txt", str(outside)):
        result = await _call(server, "read_file", {"path": raw})
        assert result.is_error is True
        assert "sandbox" in result.content[0].text.lower()


async def test_symlink_escape_rejected(server, sandbox) -> None:
    outside = sandbox.parent / "outside-link-target.txt"
    outside.write_text("secret", encoding="utf-8")
    (sandbox / "link.txt").symlink_to(outside)
    result = await _call(server, "read_file", {"path": "link.txt"})
    assert result.is_error is True
    assert "sandbox" in result.content[0].text.lower()


async def test_missing_sandbox_root_rejected(monkeypatch) -> None:
    monkeypatch.delenv("MCP_SANDBOX_ROOT", raising=False)
    result = await _call(create_server(), "read_file", {"path": "any.txt"})
    assert result.is_error is True
    assert "MCP_SANDBOX_ROOT" in result.content[0].text


# ---------- confirmation: A6 ----------


def test_confirmation_issue_and_consume_returns_args() -> None:
    store = ConfirmationStore(ttl_s=60, secret=b"secret")
    args = {"path": "notes/a.md", "text": "hello"}
    token = store.issue("write_note", args)
    other_token = store.issue("write_note", {"path": "other.md"})
    assert token and other_token and token != other_token
    assert store.consume(token) == args
    assert store.pending_count() == 1
    store.consume(other_token)
    assert store.pending_count() == 0


def test_confirmation_replay_raises_used() -> None:
    store = ConfirmationStore(ttl_s=60, secret=b"secret")
    token = store.issue("write_note", {"path": "a.md"})
    store.consume(token)
    with pytest.raises(Used) as exc:
        store.consume(token)
    assert exc.value.status_code == 409


def test_confirmation_unknown_token_raises_unknown() -> None:
    store = ConfirmationStore(ttl_s=60, secret=b"secret")
    for token in ("bogus", "0" * 64, ""):
        with pytest.raises(Unknown) as exc:
            store.consume(token)
        assert exc.value.status_code == 404


def test_confirmation_expired_token_raises_expired() -> None:
    clock = [1000.0]
    store = ConfirmationStore(ttl_s=60, secret=b"secret", now_fn=lambda: clock[0])
    token = store.issue("write_note", {"path": "a.md"})
    clock[0] += 61
    with pytest.raises(Expired) as exc:
        store.consume(token)
    assert exc.value.status_code == 410


def test_confirmation_token_bound_to_tool_and_args() -> None:
    store = ConfirmationStore(ttl_s=60, secret=b"secret")
    token = store.issue("write_note", {"path": "a.md", "text": "x"})
    tampered = token[:-1] + ("0" if token[-1] != "0" else "1")
    with pytest.raises(Unknown):
        store.consume(tampered)

    other = ConfirmationStore(ttl_s=60, secret=b"different-secret")
    other.issue("write_note", {"path": "a.md"})
    with pytest.raises(Unknown):
        other.consume(token)


def test_confirmation_atomic_single_use() -> None:
    store = ConfirmationStore(ttl_s=60, secret=b"secret")
    token = store.issue("write_note", {"path": "a.md"})
    wins: list[dict[str, Any]] = []
    used: list[BaseException] = []
    barrier = threading.Barrier(8)

    def worker() -> None:
        barrier.wait()
        try:
            wins.append(store.consume(token))
        except Used as exc:
            used.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert wins == [{"path": "a.md"}]
    assert len(used) == 7


# ---------- API: A5 / A6 ----------


async def test_api_lists_tools_with_confirmation_flags(mcp_api) -> None:
    client, _, _ = mcp_api
    response = await client.get("/api/mcp/tools")
    assert response.status_code == 200
    tools = {tool["name"]: tool for tool in response.json()}
    assert set(tools) == {"list_dir", "read_file", "write_note"}
    assert tools["write_note"]["requires_confirmation"] is True
    assert tools["read_file"]["requires_confirmation"] is False
    assert tools["read_file"]["input_schema"]["properties"]["path"]["type"] == "string"


async def test_api_read_only_tool_executes_directly(mcp_api) -> None:
    client, _, _ = mcp_api
    response = await client.post("/api/mcp/call", json={"tool": "read_file", "args": {"path": "hello.txt"}})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["pending"] is None
    assert body["result"]["text"] == "hello mcp"


async def test_api_write_note_pending_then_confirm_writes(mcp_api) -> None:
    client, _, sandbox = mcp_api
    target = sandbox / "notes" / "todo.md"
    pending = await client.post(
        "/api/mcp/call",
        json={"tool": "write_note", "args": {"path": "notes/todo.md", "text": "hello note"}},
    )
    assert pending.status_code == 202
    body = pending.json()
    assert body["ok"] is False
    assert body["pending"] is True
    assert body["result"] is None
    assert body["token"]
    assert not target.exists(), "write_note must not touch disk before confirmation"

    confirmed = await client.post("/api/mcp/confirm", json={"token": body["token"]})
    assert confirmed.status_code == 200
    result = confirmed.json()
    assert result["ok"] is True
    assert result["result"]["written"] is True
    assert target.read_text(encoding="utf-8") == "hello note"

    replay = await client.post("/api/mcp/confirm", json={"token": body["token"]})
    assert replay.status_code == 409


async def test_api_confirm_unknown_token_404(mcp_api) -> None:
    client, _, _ = mcp_api
    response = await client.post("/api/mcp/confirm", json={"token": "unknown-token"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "http_404"


async def test_api_confirm_expired_token_410(mcp_api) -> None:
    client, app, sandbox = mcp_api
    token = app.state.mcp_confirmations.issue("write_note", {"path": "late.md", "text": "x"}, ttl=-1)
    response = await client.post("/api/mcp/confirm", json={"token": token})
    assert response.status_code == 410
    assert not (sandbox / "late.md").exists()


async def test_api_sandbox_escape_returns_400(mcp_api) -> None:
    client, _, sandbox = mcp_api
    outside = sandbox.parent / "api-outside.txt"
    outside.write_text("secret", encoding="utf-8")
    response = await client.post("/api/mcp/call", json={"tool": "read_file", "args": {"path": "../api-outside.txt"}})
    assert response.status_code == 400
    assert "sandbox" in response.json()["error"]["message"].lower()


async def test_api_mcp_disabled_returns_503(monkeypatch) -> None:
    monkeypatch.setenv("CLASSIFIER_BACKEND", "rules")
    monkeypatch.setenv("RAG_BACKEND", "demo")
    monkeypatch.delenv("MCP_ENABLED", raising=False)
    get_settings.cache_clear()
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/mcp/tools")
        assert response.status_code == 503
        response = await client.post("/api/mcp/call", json={"tool": "read_file", "args": {"path": "a.txt"}})
        assert response.status_code == 503
        response = await client.post("/api/mcp/confirm", json={"token": "tok"})
        assert response.status_code == 503
    get_settings.cache_clear()


# ---------- optional stdio smoke ----------


@pytest.mark.skipif(os.environ.get("RAGM_MCP_SKIP_STDIO") == "1", reason="stdio smoke disabled by env")
def test_stdio_manager_roundtrip(sandbox) -> None:
    """真实 stdio 子进程：PYTHONPATH / sandbox 环境由 manager 注入。"""
    manager = MCPClientManager(sandbox, timeout_s=20)
    assert {tool["name"] for tool in manager.list_tools()} == {"list_dir", "read_file", "write_note"}
    result = manager.run_call("read_file", {"path": "hello.txt"})
    assert result["text"] == "hello mcp"
