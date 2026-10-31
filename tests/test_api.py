"""API 契约测试：不依赖 Milvus / Ollama（健康检查允许 degraded）。"""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from ragm.api.main import create_app


@pytest.fixture
async def client():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_health(client: AsyncClient) -> None:
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in {"ok", "degraded"}
    assert {"milvus", "ollama", "classifier"} <= set(body["checks"])
    assert "version" in body


async def test_scenarios(client: AsyncClient) -> None:
    resp = await client.get("/api/scenarios")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 4
    assert {s["route"] for s in data} == {"simple", "multihop", "agentic"}
    assert all({"id", "question", "title"} <= set(s) for s in data)


async def test_query_simple(client: AsyncClient) -> None:
    resp = await client.post("/api/query", json={"question": "Milvus 的 BM25 Function 是什么？"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["route"] == "simple"
    assert body["served_by"] == "demo"
    assert body["evidence"] and body["trace"]
    # Stats 用 camelCase 别名序列化，前端无需改动
    assert {"latencyMs", "tokensIn", "tokensOut", "costCny", "citationCheck"} <= set(body["stats"])
    assert body["generated_at"]


async def test_query_route_override(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/query",
        json={"question": "随便问一个多跳问题", "route_override": "multihop"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["route"] == "multihop"
    assert body["graph"]["nodes"]


async def test_query_refusal_case(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/query",
        json={"question": "OpenAI GPT-5 2026 定价是多少？", "route_override": "agentic"},
    )
    assert resp.status_code == 200
    body = resp.json()
    # 手动指定 agentic 时命中第一个 agentic 案例（非拒答），拒答由守卫触发
    assert body["route"] == "agentic"


async def test_query_validation(client: AsyncClient) -> None:
    resp = await client.post("/api/query", json={"question": ""})
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "validation_error"


async def test_route_endpoint(client: AsyncClient) -> None:
    resp = await client.post("/api/route", json={"question": "对比 bge-m3 与 Qwen3 的优劣并给建议"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] in {"simple", "multihop", "reasoning"}
    assert body["route"] in {"simple", "multihop", "agentic"}
    assert 0 <= body["confidence"] <= 1


async def test_query_auto_uses_classifier(client: AsyncClient) -> None:
    resp = await client.post("/api/query", json={"question": "Milvus 的 HNSW 默认 M 是多少？"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["classifier"] is not None
    assert body["route"] == body["classifier"]["route"]


async def test_feedback(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/feedback",
        json={"query_id": "q_test", "rating": "up", "comment": "引用很准"},
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


async def test_stream_sse(client: AsyncClient) -> None:
    resp = await client.post("/api/query/stream", json={"question": "Milvus 的 BM25 Function 是什么？"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    text = (await resp.aread()).decode()
    assert "event: stage" in text
    assert "event: token" in text
    assert "event: done" in text
