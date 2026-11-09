"""v1.1 W1：会话存储 / 查询改写 / 子问题多查询检索 / 会话 API。"""
from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from ragm.api import routes
from ragm.api.main import create_app
from ragm.api.schemas import QueryRequest
from ragm.classifier.base import RouteDecision
from ragm.config import Settings, get_settings
from ragm.conversation.rewrite import Rewriter, SessionContext
from ragm.conversation.store import SessionStore
from ragm.rag.graph import RagGraph
from ragm.rag.langgraph_service import LangGraphRagService
from ragm.rag.llm_router import build_llm
from ragm.rag.service import DemoRagService

HISTORY = [
    {"role": "user", "content": "鸣人的父亲是波风水门"},
    {"role": "assistant", "content": "是的"},
]
REWRITE_PAYLOAD = json.dumps(
    {
        "rewritten": "鸣人的师父是谁？",
        "subquestions": ["鸣人的师父是谁？", "自来也教过鸣人什么？"],
    },
    ensure_ascii=False,
)


# ---------- fakes ----------
class FakeLLM:
    def __init__(self, text: str = "", error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages: list[dict], max_tokens: int = 1024, json_mode: bool = False) -> dict:
        self.calls.append({"messages": messages, "max_tokens": max_tokens, "json_mode": json_mode})
        if self.error is not None:
            raise self.error
        return {"text": self.text, "tokens_in": 12, "tokens_out": 8}

    def stream_chat(self, messages: list[dict], max_tokens: int = 1024) -> Iterator[str]:
        self.calls.append({"messages": messages, "max_tokens": max_tokens, "json_mode": False, "stream": True})
        if self.error is not None:
            raise self.error
        yield self.text


def _hit(chunk_id: str, score: float) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": f"doc-{chunk_id}",
        "title": chunk_id,
        "source": "test",
        "lang": "zh",
        "chunk_index": 0,
        "text": f"text-{chunk_id}",
        "score": score,
        "dense": score,
        "sparse": score,
    }


class FakeSearch:
    def __init__(self, hits_by_query: dict[str, list[dict]]) -> None:
        self.hits_by_query = hits_by_query
        self.calls: list[str] = []

    def search_with_scores(self, query: str, k: int = 8, pool: int = 40) -> dict:
        self.calls.append(query)
        hits = [dict(hit) for hit in self.hits_by_query.get(query, [])][:k]
        return {"query": query, "mode": "hybrid", "latency_ms": 0.0, "hits": hits}

    def chunks_by_titles(self, titles: list[str], limit: int = 8) -> list[dict]:
        return []


class SessionAwareDemo(DemoRagService):
    """模拟 LangGraphRagService：声明 supports_session 并记录收到的上下文。"""

    supports_session = True

    def __init__(self) -> None:
        super().__init__(delay_scale=0.0)
        self.sessions: list[SessionContext | None] = []

    async def query(self, req, decision=None, *, session=None):
        self.sessions.append(session)
        return await super().query(req, decision=decision)


# ---------- store ----------
def test_session_store_crud_and_restart(tmp_path) -> None:
    db_path = tmp_path / "sessions.sqlite"
    store = SessionStore(db_path)
    session = store.create_session("鸣人问答")
    assert session["message_count"] == 0
    assert store.get_session(session["id"])["messages"] == []

    store.append_message(session["id"], "user", "鸣人的父亲是谁？")
    store.append_message(session["id"], "assistant", "波风水门", meta={"route": "simple"})

    detail = store.get_session(session["id"])
    assert detail["message_count"] == 2
    assert [message["role"] for message in detail["messages"]] == ["user", "assistant"]
    assert detail["messages"][1]["meta"] == {"route": "simple"}

    restarted = SessionStore(db_path)
    assert restarted.get_session(session["id"])["message_count"] == 2
    assert restarted.list_sessions()[0]["id"] == session["id"]
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    with restarted._connect() as conn:
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 3000
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    assert restarted.delete_session(session["id"]) is True
    assert restarted.get_session(session["id"]) is None
    assert restarted.delete_session("s_missing") is False
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0


def test_history_limits_turns_and_chars(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.sqlite")
    session_id = store.create_session()["id"]
    for index in range(3):
        store.append_message(session_id, "user", f"问题{index}" + "x" * 100)
        store.append_message(session_id, "assistant", f"回答{index}" + "y" * 100)

    last_turn = store.get_history(session_id, max_turns=1, max_chars=10_000)
    assert [message["role"] for message in last_turn] == ["user", "assistant"]
    assert last_turn[0]["content"].startswith("问题2")

    trimmed = store.get_history(session_id, max_turns=3, max_chars=250)
    assert len(trimmed) == 2
    assert trimmed[-1]["content"].startswith("回答2")
    assert store.get_history(session_id, max_turns=0) == []
    assert store.get_history("s_missing") == []


def test_append_to_missing_session_raises(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.sqlite")
    with pytest.raises(KeyError):
        store.append_message("s_missing", "user", "hi")


# ---------- rewriter ----------
def test_rewriter_resolves_references_and_splits_subquestions() -> None:
    fenced = '```json\n' + REWRITE_PAYLOAD + '\n```'
    llm = FakeLLM(fenced)
    result = Rewriter(llm).rewrite("他的师父是谁？", HISTORY)
    assert "鸣人" in result["rewritten"]
    assert result["subquestions"] == ["鸣人的师父是谁？", "自来也教过鸣人什么？"]
    assert len(llm.calls) == 1
    assert llm.calls[0]["json_mode"] is True
    prompt = llm.calls[0]["messages"][-1]["content"]
    assert "波风水门" in prompt
    assert "他的师父是谁？" in prompt
    system_prompt = llm.calls[0]["messages"][0]["content"]
    assert "上一轮**用户问题**" in system_prompt
    assert "示例" in system_prompt


def test_rewriter_pronoun_guard_uses_previous_user_subject() -> None:
    wrong = json.dumps(
        {"rewritten": "波风水门的师父是谁？", "subquestions": ["波风水门的师父是谁？"]},
        ensure_ascii=False,
    )
    result = Rewriter(FakeLLM(wrong)).rewrite("他的师父是谁？", HISTORY)
    assert result["rewritten"].startswith("鸣人")
    assert "波风水门" in result["rewritten"]


def test_rewriter_falls_back_to_llm_client_without_state_llm(monkeypatch) -> None:
    class StubLLM:
        def __init__(self, *args) -> None:
            self.args = args

    monkeypatch.setattr(routes, "LLMClient", StubLLM)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    rewriter = routes._rewriter(request)
    assert isinstance(rewriter.llm, StubLLM)
    assert len(rewriter.llm.args) == 4
    assert routes._rewriter(request) is rewriter


def test_rewriter_caps_and_dedupes_subquestions() -> None:
    payload = json.dumps(
        {"rewritten": "问题", "subquestions": ["问题", "a", "a", "b", "c", "d"]},
        ensure_ascii=False,
    )
    result = Rewriter(FakeLLM(payload)).rewrite("问题", HISTORY)
    assert result["subquestions"] == ["问题", "a", "b"]


@pytest.mark.parametrize(
    "llm",
    [FakeLLM("not json at all"), FakeLLM(error=RuntimeError("boom"))],
    ids=["bad-json", "llm-error"],
)
def test_rewriter_falls_back_on_failures(llm: FakeLLM) -> None:
    result = Rewriter(llm).rewrite("他的师父是谁？", HISTORY)
    assert result == {"rewritten": "他的师父是谁？", "subquestions": []}


@pytest.mark.skipif(os.environ.get("RAGM_LLM_E2E") != "1", reason="真实 LLM 集成测试，设 RAGM_LLM_E2E=1 开启")
def test_rewriter_real_llm_resolves_pronoun_to_user_subject() -> None:
    history = [
        {"role": "user", "content": "鸣人的父亲是波风水门"},
        {"role": "assistant", "content": "是的，鸣人的父亲是波风水门。"},
    ]
    llm = build_llm(Settings(_env_file=None))
    if not llm.providers:
        pytest.skip("no usable LLM providers (Ollama gemma4 not configured)")
    result = Rewriter(llm).rewrite("他的师父是谁？", history)
    assert "鸣人" in result["rewritten"]
    assert "波风水门的师父" not in result["rewritten"]


def test_rewriter_skips_llm_without_history() -> None:
    llm = FakeLLM(REWRITE_PAYLOAD)
    assert Rewriter(llm).rewrite("他的师父是谁？", []) == {
        "rewritten": "他的师父是谁？",
        "subquestions": [],
    }
    assert llm.calls == []


# ---------- multi-query retrieval ----------
def test_multi_query_retrieval_calls_each_subquestion_and_dedupes() -> None:
    subquestions = ["鸣人的师父是谁？", "自来也教过鸣人什么？"]
    search = FakeSearch(
        {
            subquestions[0]: [_hit("c1", 0.9), _hit("c2", 0.5)],
            subquestions[1]: [_hit("c2", 0.8), _hit("c3", 0.7)],
        }
    )
    graph = RagGraph(None, search, FakeLLM("答案 [1]"), top_k=8)
    state = graph._retrieve_node(
        {"question": "对比鸣人的师父与自来也", "route": "simple", "subquestions": subquestions}
    )
    assert search.calls == subquestions
    assert [hit["chunk_id"] for hit in state["hits"]] == ["c2", "c1", "c3"]
    by_id = {hit["chunk_id"]: hit for hit in state["hits"]}
    assert by_id["c2"]["score"] == 0.8
    assert by_id["c2"]["multi_query"] == 2
    assert state["retrieve_detail"].startswith("多查询 2")


def test_invoke_passes_subquestions_to_retrieve() -> None:
    subquestions = ["q1", "q2"]
    search = FakeSearch({subquestions[0]: [_hit("a", 0.9)], subquestions[1]: [_hit("b", 0.8)]})
    graph = RagGraph(None, search, FakeLLM("答案 [1]"), top_k=8)
    decision = RouteDecision(label="simple", confidence=0.9, backend="test", latency_ms=0.0)
    state = graph.invoke("对比 q1 和 q2", "simple", decision, subquestions=subquestions)
    assert search.calls == subquestions
    assert state["subquestions"] == subquestions
    assert {hit["chunk_id"] for hit in state["hits"]} == {"a", "b"}


def test_retrieve_without_subquestions_keeps_single_query() -> None:
    search = FakeSearch({"怎么问": [_hit("only", 0.9)]})
    graph = RagGraph(None, search, FakeLLM("答案 [1]"), top_k=8)
    state = graph._retrieve_node({"question": "怎么问", "route": "simple"})
    assert search.calls == ["怎么问"]
    assert "多查询" not in state["retrieve_detail"]


async def test_langgraph_service_uses_rewritten_question_and_subquestions() -> None:
    subquestions = ("鸣人的师父是谁？", "自来也教过鸣人什么？")
    search = FakeSearch({subquestions[0]: [_hit("c1", 0.9)], subquestions[1]: [_hit("c2", 0.8)]})
    service = LangGraphRagService(None, search, FakeLLM("答案 [1]"), Settings(_env_file=None))
    context = SessionContext(
        session_id="s1",
        question="他的师父是谁？",
        rewritten_question=subquestions[0],
        subquestions=subquestions,
    )
    response = await service.query(
        QueryRequest(question="他的师父是谁？", route_override="simple"), session=context
    )
    assert search.calls == list(subquestions)
    assert response.question == "他的师父是谁？"


async def test_langgraph_service_backfills_provider() -> None:
    search = FakeSearch({"q": [_hit("c1", 0.9)]})
    llm = FakeLLM("答案 [1]")
    service = LangGraphRagService(None, search, llm, Settings(_env_file=None))
    request = QueryRequest(question="q", route_override="simple")
    # fake llm 没有 last_provider 属性 → None；有则回填为字符串。
    assert (await service.query(request)).provider is None
    llm.last_provider = "fake-primary"
    assert (await service.query(request)).provider == "fake-primary"


def _langgraph_service(provider: str) -> LangGraphRagService:
    llm = FakeLLM("答案 [1]")
    llm.last_provider = provider
    return LangGraphRagService(None, FakeSearch({"q": [_hit("c1", 0.9)]}), llm, Settings(_env_file=None))


async def test_query_response_backfills_provider(session_env) -> None:
    client, app, _, _ = session_env
    app.state.service = _langgraph_service("fake-primary")
    resp = await client.post("/api/query", json={"question": "q", "route_override": "simple"})
    assert resp.status_code == 200
    assert resp.json()["provider"] == "fake-primary"


async def test_query_stream_done_backfills_provider(session_env) -> None:
    client, app, _, _ = session_env
    app.state.service = _langgraph_service("fake-primary")
    resp = await client.post("/api/query/stream", json={"question": "q", "route_override": "simple"})
    assert resp.status_code == 200
    text = (await resp.aread()).decode()
    done_block = [block for block in text.split("\n\n") if block.startswith("event: done")][-1]
    payload = json.loads(done_block.split("data:", 1)[1].strip())
    assert payload["response"]["provider"] == "fake-primary"


# ---------- session API ----------
@pytest.fixture
async def session_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CLASSIFIER_BACKEND", "rules")
    monkeypatch.setenv("RAG_BACKEND", "demo")
    monkeypatch.setenv("SESSION_DB_PATH", str(tmp_path / "sessions.sqlite"))
    get_settings.cache_clear()
    app = create_app()
    app.state.service = DemoRagService(delay_scale=0.0)
    llm = FakeLLM(REWRITE_PAYLOAD)
    app.state.rewriter = Rewriter(llm)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, app, llm, tmp_path / "sessions.sqlite"
    get_settings.cache_clear()


async def test_session_api_crud(session_env) -> None:
    client, _, _, _ = session_env
    created = await client.post("/api/sessions", json={"title": "鸣人会话"})
    assert created.status_code == 201
    body = created.json()
    assert body["title"] == "鸣人会话"
    assert body["message_count"] == 0
    session_id = body["id"]

    detail = await client.get(f"/api/sessions/{session_id}")
    assert detail.status_code == 200
    assert detail.json()["messages"] == []
    assert (await client.get("/api/sessions/s_missing")).status_code == 404

    assert (await client.delete(f"/api/sessions/{session_id}")).status_code == 204
    assert (await client.get(f"/api/sessions/{session_id}")).status_code == 404
    assert (await client.delete(f"/api/sessions/{session_id}")).status_code == 404


async def test_query_with_session_rewrites_and_persists(session_env) -> None:
    client, _, llm, db_path = session_env
    session_id = (await client.post("/api/sessions", json={})).json()["id"]
    store = SessionStore(db_path)
    store.append_message(session_id, "user", "鸣人的父亲是波风水门")
    store.append_message(session_id, "assistant", "是的")

    resp = await client.post(
        "/api/query",
        json={"question": "他的师父是谁？", "session_id": session_id, "route_override": "simple"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["question"] == "他的师父是谁？"
    assert body["session_id"] == session_id
    assert "鸣人" in body["rewritten_question"]
    assert len(body["subquestions"]) == 2
    assert llm.calls[0]["json_mode"] is True

    detail = (await client.get(f"/api/sessions/{session_id}")).json()
    assert detail["message_count"] == 4
    assert [message["role"] for message in detail["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert detail["messages"][2]["content"] == "他的师父是谁？"


async def test_rewriter_prefers_app_state_llm(session_env) -> None:
    client, app, _, db_path = session_env
    del app.state.rewriter
    fake = FakeLLM(REWRITE_PAYLOAD)
    app.state.llm = fake
    session_id = (await client.post("/api/sessions", json={})).json()["id"]
    SessionStore(db_path).append_message(session_id, "user", "鸣人的父亲是波风水门")

    resp = await client.post(
        "/api/query",
        json={"question": "他的师父是谁？", "session_id": session_id, "route_override": "simple"},
    )
    assert resp.status_code == 200
    assert len(fake.calls) == 1
    assert fake.calls[0]["json_mode"] is True
    assert "鸣人" in resp.json()["rewritten_question"]


async def test_query_without_history_skips_rewriter(session_env) -> None:
    client, _, llm, _ = session_env
    session_id = (await client.post("/api/sessions", json={})).json()["id"]
    resp = await client.post(
        "/api/query",
        json={"question": "Milvus 是什么？", "session_id": session_id, "route_override": "simple"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert llm.calls == []
    assert body["rewritten_question"] == "Milvus 是什么？"
    assert body["subquestions"] == []
    assert (await client.get(f"/api/sessions/{session_id}")).json()["message_count"] == 2


async def test_session_aware_service_receives_rewritten_context(session_env) -> None:
    client, app, _, db_path = session_env
    service = SessionAwareDemo()
    app.state.service = service
    session_id = (await client.post("/api/sessions", json={})).json()["id"]
    SessionStore(db_path).append_message(session_id, "user", "鸣人的父亲是波风水门")

    resp = await client.post(
        "/api/query",
        json={"question": "他的师父是谁？", "session_id": session_id, "route_override": "simple"},
    )
    assert resp.status_code == 200
    assert len(service.sessions) == 1
    context = service.sessions[0]
    assert context is not None
    assert "鸣人" in context.rewritten_question
    assert len(context.subquestions) == 2
    assert context.question == "他的师父是谁？"


async def test_query_unknown_session_returns_404(session_env) -> None:
    client, _, _, _ = session_env
    resp = await client.post("/api/query", json={"question": "他的师父是谁？", "session_id": "s_missing"})
    assert resp.status_code == 404
    assert "session not found" in resp.json()["error"]["message"]


async def test_query_without_session_keeps_legacy_response(session_env) -> None:
    client, _, llm, _ = session_env
    resp = await client.post("/api/query", json={"question": "Milvus 是什么？", "route_override": "simple"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] is None
    assert body["subquestions"] == []
    assert body["provider"] is None
    assert llm.calls == []


async def test_rewrite_disabled_by_config(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CLASSIFIER_BACKEND", "rules")
    monkeypatch.setenv("RAG_BACKEND", "demo")
    monkeypatch.setenv("SESSION_DB_PATH", str(tmp_path / "sessions.sqlite"))
    monkeypatch.setenv("SESSION_REWRITE_ENABLED", "false")
    get_settings.cache_clear()
    app = create_app()
    app.state.service = DemoRagService(delay_scale=0.0)
    llm = FakeLLM(REWRITE_PAYLOAD)
    app.state.rewriter = Rewriter(llm)
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            session_id = (await client.post("/api/sessions", json={})).json()["id"]
            SessionStore(tmp_path / "sessions.sqlite").append_message(
                session_id, "user", "鸣人的父亲是波风水门"
            )
            resp = await client.post(
                "/api/query",
                json={"question": "他的师父是谁？", "session_id": session_id, "route_override": "simple"},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert llm.calls == []
            assert body["rewritten_question"] == "他的师父是谁？"
            assert body["subquestions"] == []
    finally:
        get_settings.cache_clear()


async def test_query_stream_with_session_fills_fields_and_persists(session_env) -> None:
    client, _, _, db_path = session_env
    session_id = (await client.post("/api/sessions", json={})).json()["id"]
    SessionStore(db_path).append_message(session_id, "user", "鸣人的父亲是波风水门")

    resp = await client.post(
        "/api/query/stream",
        json={"question": "他的师父是谁？", "session_id": session_id, "route_override": "simple"},
    )
    assert resp.status_code == 200
    text = (await resp.aread()).decode()
    done_block = [block for block in text.split("\n\n") if block.startswith("event: done")][-1]
    payload = json.loads(done_block.split("data:", 1)[1].strip())
    assert payload["response"]["session_id"] == session_id
    assert "鸣人" in payload["response"]["rewritten_question"]

    detail = (await client.get(f"/api/sessions/{session_id}")).json()
    assert detail["message_count"] == 3
