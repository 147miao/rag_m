"""代码 API / 服务 / agent 工具测试：全部使用内存 fake，不依赖网络 / Milvus / GPU。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from ragm.agent.code_tools import CODE_TOOL_SCHEMAS, CodeToolExecutor
from ragm.api import code_routes
from ragm.api.code_routes import router as code_router
from ragm.code.service import CodeService, verify_code_citations

# ---------- fake W1 契约实现 ----------

class FakeStore:
    def __init__(self) -> None:
        self.repos: dict[str, dict[str, Any]] = {}
        self.hits: list[dict[str, Any]] = []
        self.symbols: list[dict[str, Any]] = []
        self.imports: dict[tuple[str, str], list[str]] = {}
        self.overviews: dict[str, dict[str, Any]] = {}
        self.search_calls: list[dict[str, Any]] = []

    def add_repo(self, root: str, name: str) -> dict[str, Any]:
        repo_id = f"repo-{len(self.repos) + 1}"
        repo = {
            "repo_id": repo_id,
            "root": root,
            "name": name,
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
            "file_count": 3,
            "symbol_count": 5,
            "status": "ready",
        }
        self.repos[repo_id] = repo
        return dict(repo)

    def list_repos(self) -> list[dict[str, Any]]:
        return [dict(repo) for repo in self.repos.values()]

    def get_repo(self, repo_id: str) -> dict[str, Any] | None:
        repo = self.repos.get(repo_id)
        return dict(repo) if repo else None

    def delete_repo(self, repo_id: str) -> bool:
        return self.repos.pop(repo_id, None) is not None

    def search(
        self,
        query: str,
        k: int = 8,
        repo_id: str | None = None,
        path_prefix: str | None = None,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        self.search_calls.append(
            {"query": query, "k": k, "repo_id": repo_id, "path_prefix": path_prefix, "kind": kind}
        )
        results = []
        for hit in self.hits:
            if repo_id and hit.get("repo_id") != repo_id:
                continue
            if path_prefix and not str(hit.get("path", "")).startswith(path_prefix):
                continue
            if kind and hit.get("kind") != kind:
                continue
            results.append(dict(hit))
        return results[:k]

    def get_symbol(self, repo_id: str, qualname: str, path: str | None = None) -> dict[str, Any] | None:
        for symbol in self.symbols:
            if symbol.get("repo_id") != repo_id or symbol.get("qualname") != qualname:
                continue
            if path is not None and symbol.get("path") != path:
                continue
            return dict(symbol)
        return None

    def reverse_imports(self, repo_id: str, path: str) -> list[str]:
        return list(self.imports.get((repo_id, path), []))

    def overview(self, repo_id: str) -> dict[str, Any]:
        return dict(self.overviews.get(repo_id, {}))


class FakeIndexer:
    def __init__(self, store: FakeStore, skipped: int = 0) -> None:
        self.store = store
        self.skipped = skipped
        self.calls: list[dict[str, Any]] = []

    def index(self, root: str, name: str, reindex: bool = False) -> dict[str, Any]:
        self.calls.append({"root": root, "name": name, "reindex": reindex})
        repo = next((item for item in self.store.repos.values() if item["root"] == root), None)
        if repo is None:
            repo = self.store.add_repo(root, name)
        return {
            "repo_id": repo["repo_id"],
            "stats": {
                "file_count": repo["file_count"],
                "symbol_count": repo["symbol_count"],
                "skipped": self.skipped,
                "reindexed": reindex,
            },
        }


class FakeLLM:
    provider = "fake"

    def __init__(self, text: str = "", error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.messages: list[dict[str, Any]] | None = None
        self.max_tokens: int | None = None

    def chat(self, messages: list[dict[str, Any]], max_tokens: int = 1024, json_mode: bool = False) -> dict[str, Any]:
        self.messages = messages
        self.max_tokens = max_tokens
        if self.error is not None:
            raise self.error
        return {"text": self.text, "tokens_in": 1, "tokens_out": 1}


# ---------- fixtures ----------

@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def indexer(store: FakeStore) -> FakeIndexer:
    return FakeIndexer(store)


@pytest.fixture
def repo_id(store: FakeStore, tmp_path: Path) -> str:
    return str(store.add_repo(str(tmp_path), "demo")["repo_id"])


@pytest.fixture
def make_app(store: FakeStore, indexer: FakeIndexer, tmp_path: Path):
    def _make(llm: FakeLLM | None = None, service: CodeService | None = None) -> FastAPI:
        app = FastAPI()
        app.include_router(code_router)
        app.state.code_service = service if service is not None else CodeService(store, llm=llm)
        app.state.code_indexer = indexer
        app.state.code_allowed_roots = [str(tmp_path)]
        return app

    return _make


def client_for(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def make_hit(repo_id: str, path: str = "src/a.py", qualname: str = "foo") -> dict[str, Any]:
    return {
        "repo_id": repo_id,
        "chunk_id": f"{path}:{qualname}",
        "path": path,
        "qualname": qualname,
        "kind": "function",
        "start_line": 2,
        "end_line": 5,
        "score": 0.9,
        "text": "def foo():\n    return 1",
    }


def write_lines(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"line {i}" for i in range(1, count + 1)) + "\n", encoding="utf-8")


# ---------- search / symbol / overview / impact ----------

async def test_search_endpoint_filters_and_normalizes(make_app, store, repo_id) -> None:
    store.hits = [
        make_hit(repo_id, "src/a.py", "foo"),
        make_hit(repo_id, "src/b.py", "bar"),
        make_hit(repo_id, "tests/test_a.py", "test_foo"),
    ]
    async with client_for(make_app()) as client:
        resp = await client.get(
            "/api/code/search",
            params={"q": "foo", "repo_id": repo_id, "k": 2, "path_prefix": "src/", "kind": "function"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    assert body[0]["path"] == "src/a.py"
    assert body[0]["start_line"] == 2 and body[0]["end_line"] == 5
    assert body[0]["repo_id"] == repo_id
    assert store.search_calls[-1] == {
        "query": "foo",
        "k": 2,
        "repo_id": repo_id,
        "path_prefix": "src/",
        "kind": "function",
    }


async def test_search_errors(make_app) -> None:
    async with client_for(make_app()) as client:
        missing_repo = await client.get("/api/code/search", params={"q": "foo", "repo_id": "nope"})
        blank_query = await client.get("/api/code/search", params={"q": "   "})
        bad_k = await client.get("/api/code/search", params={"q": "foo", "k": 0})
        no_q = await client.get("/api/code/search")
    assert missing_repo.status_code == 404
    assert blank_query.status_code == 400
    assert bad_k.status_code == 422
    assert no_q.status_code == 422


async def test_symbol_endpoint_and_404(make_app, store, repo_id) -> None:
    store.symbols = [
        {
            "repo_id": repo_id,
            "path": "src/a.py",
            "qualname": "foo",
            "kind": "function",
            "start_line": 2,
            "end_line": 5,
            "docstring": "do foo",
            "code": "def foo():\n    return 1",
        }
    ]
    async with client_for(make_app()) as client:
        resp = await client.get("/api/code/symbol", params={"repo_id": repo_id, "qualname": "foo"})
        missing = await client.get("/api/code/symbol", params={"repo_id": repo_id, "qualname": "nope"})
        missing_repo = await client.get(
            "/api/code/symbol", params={"repo_id": "nope", "qualname": "foo"}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"].startswith("def foo")
    assert body["docstring"] == "do foo"
    assert missing.status_code == 404
    assert missing_repo.status_code == 404


async def test_overview_and_impact(make_app, store, repo_id) -> None:
    store.overviews[repo_id] = {
        "file_count": 7,
        "symbol_count": 11,
        "kinds": {"function": 8, "class": 3},
        "top_paths": ["src/a.py", "src/b.py"],
        "note": "ok",
    }
    store.imports[(repo_id, "src/a.py")] = ["src/b.py", "src/c.py"]
    async with client_for(make_app()) as client:
        overview = await client.get("/api/code/overview", params={"repo_id": repo_id})
        impact = await client.get("/api/code/impact", params={"repo_id": repo_id, "path": "src/a.py"})
        missing_overview = await client.get("/api/code/overview", params={"repo_id": "nope"})
        missing_impact = await client.get(
            "/api/code/impact", params={"repo_id": "nope", "path": "src/a.py"}
        )
    assert overview.status_code == 200
    data = overview.json()
    assert data["repo_id"] == repo_id
    assert data["name"] == "demo"
    assert data["file_count"] == 7 and data["symbol_count"] == 11
    assert data["kinds"] == {"function": 8, "class": 3}
    assert data["top_paths"] == ["src/a.py", "src/b.py"]
    assert impact.status_code == 200
    payload = impact.json()
    assert payload["imports"] == ["src/b.py", "src/c.py"]
    assert payload["heuristic"] is True
    assert "启发式" in payload["note"]
    assert missing_overview.status_code == 404
    assert missing_impact.status_code == 404


# ---------- repos ----------

async def test_repo_lifecycle(make_app, indexer, tmp_path) -> None:
    async with client_for(make_app()) as client:
        created = await client.post(
            "/api/code/repos", json={"root": str(tmp_path), "name": "demo"}
        )
        listed = await client.get("/api/code/repos")
        detail = await client.get(f"/api/code/repos/{created.json()['repo_id']}")
        reindexed = await client.post(f"/api/code/repos/{created.json()['repo_id']}/reindex")
        deleted = await client.delete(f"/api/code/repos/{created.json()['repo_id']}")
        gone = await client.get(f"/api/code/repos/{created.json()['repo_id']}")
        deleted_again = await client.delete(f"/api/code/repos/{created.json()['repo_id']}")
    assert created.status_code == 200
    created_body = created.json()
    assert created_body["repo_id"]
    assert created_body["stats"]["skipped"] == 0
    assert listed.status_code == 200
    assert listed.json()["total"] >= 1
    assert detail.status_code == 200 and detail.json()["root"] == str(tmp_path)
    assert reindexed.status_code == 200
    assert reindexed.json()["stats"]["reindexed"] is True
    assert indexer.calls[-1]["reindex"] is True
    assert deleted.status_code == 200 and deleted.json()["deleted"] is True
    assert gone.status_code == 404
    assert deleted_again.status_code == 404


async def test_repo_create_validation_and_syntax_skips(make_app, indexer, tmp_path) -> None:
    indexer.skipped = 2
    outside = tmp_path.parent / f"outside_{tmp_path.name}"
    outside.mkdir(exist_ok=True)
    try:
        async with client_for(make_app()) as client:
            ok = await client.post("/api/code/repos", json={"root": str(tmp_path), "name": "demo"})
            missing = await client.post(
                "/api/code/repos", json={"root": str(tmp_path / "missing"), "name": "x"}
            )
            forbidden = await client.post(
                "/api/code/repos", json={"root": str(outside), "name": "x"}
            )
            blank_name = await client.post(
                "/api/code/repos", json={"root": str(tmp_path), "name": "   "}
            )
        assert ok.status_code == 200
        assert ok.json()["stats"]["skipped"] == 2
        assert missing.status_code == 404
        assert forbidden.status_code == 403
        assert blank_name.status_code == 422
    finally:
        outside.rmdir()


async def test_repo_root_symlink_escape_forbidden(make_app, tmp_path) -> None:
    outside = tmp_path.parent / f"outside_link_{tmp_path.name}"
    outside.mkdir(exist_ok=True)
    link = tmp_path / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        outside.rmdir()
        pytest.skip("symlink not supported")
    try:
        async with client_for(make_app()) as client:
            resp = await client.post("/api/code/repos", json={"root": str(link), "name": "x"})
        assert resp.status_code == 403
    finally:
        outside.rmdir()


async def test_allowed_roots_from_settings_or_explicit_empty(
    make_app, tmp_path, monkeypatch
) -> None:
    app = make_app()
    app.state.code_allowed_roots = None

    class _Settings:
        def __init__(self) -> None:
            self.code_allowed_roots = [str(tmp_path)]

    monkeypatch.setattr(code_routes, "get_settings", lambda: _Settings())
    async with client_for(app) as client:
        allowed = await client.post(
            "/api/code/repos", json={"root": str(tmp_path), "name": "demo"}
        )

    denied_app = make_app()
    denied_app.state.code_allowed_roots = []
    async with client_for(denied_app) as client:
        denied = await client.post(
            "/api/code/repos", json={"root": str(tmp_path), "name": "demo"}
        )
    assert allowed.status_code == 200
    assert denied.status_code == 403


# ---------- ask ----------

async def test_ask_builds_citations_and_prompt(make_app, store, repo_id, tmp_path) -> None:
    write_lines(tmp_path / "src/a.py", 10)
    write_lines(tmp_path / "src/b.py", 10)
    store.hits = [make_hit(repo_id, "src/a.py", "foo"), make_hit(repo_id, "src/b.py", "bar")]
    llm = FakeLLM("结论：foo 负责处理请求 [1]\n\n实现见 [2] 中的 bar。")
    async with client_for(make_app(llm=llm)) as client:
        resp = await client.post(
            "/api/code/ask", json={"question": "foo 做什么？", "repo_id": repo_id}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "foo 负责处理请求" in body["answer"]["lead"]
    assert "bar" in body["answer"]["body"]
    assert [item["n"] for item in body["citations"]] == [1, 2]
    assert body["citations"][0] == {
        "n": 1,
        "path": "src/a.py",
        "start_line": 2,
        "end_line": 5,
        "qualname": "foo",
    }
    assert body["provider"] == "fake"
    assert body["refusal"] is None
    assert llm.messages is not None
    system = llm.messages[0]["content"]
    user = llm.messages[1]["content"]
    assert "只基于上下文" in system
    assert "引用" in system
    assert "[1] src/a.py:2-5" in user
    assert "[2] src/b.py:2-5" in user
    assert "foo 做什么" in user


async def test_ask_refuses_without_hits(make_app, store, repo_id) -> None:
    llm = FakeLLM("不应被调用 [1]")
    store.hits = []
    async with client_for(make_app(llm=llm)) as client:
        resp = await client.post("/api/code/ask", json={"question": "anything", "repo_id": repo_id})
    assert resp.status_code == 200
    body = resp.json()
    assert body["citations"] == []
    assert body["refusal"] is not None
    assert "未检索到" in body["refusal"]["reason"]
    assert body["provider"] == "fake"
    assert llm.messages is None


async def test_ask_refuses_when_citation_invalid(make_app, store, repo_id) -> None:
    store.hits = [make_hit(repo_id, "src/missing.py", "foo")]
    service = CodeService(
        store,
        llm=FakeLLM("答案见 [1]"),
        file_exists=lambda _path: False,
        read_lines=lambda _path: 100,
    )
    async with client_for(make_app(service=service)) as client:
        missing_file = await client.post(
            "/api/code/ask", json={"question": "foo?", "repo_id": repo_id}
        )
    assert missing_file.status_code == 200
    assert missing_file.json()["citations"] == []
    assert missing_file.json()["refusal"] is not None

    contexts = [
        {
            "n": 1,
            "path": "src/a.py",
            "abs_path": str(Path("/repo") / "src/a.py"),
            "start_line": 2,
            "end_line": 9,
            "qualname": "foo",
        }
    ]
    citations, invalid, check = verify_code_citations(
        "见 [1]",
        contexts,
        file_exists=lambda _path: True,
        read_lines=lambda _path: 5,
    )
    assert citations == []
    assert invalid == [1]
    assert check == "0/1 通过"


async def test_verify_code_citations_unit() -> None:
    contexts = [
        {
            "n": 1,
            "path": "src/a.py",
            "abs_path": "/repo/src/a.py",
            "start_line": 2,
            "end_line": 5,
            "qualname": "foo",
        }
    ]
    citations, invalid, check = verify_code_citations(
        "见 [1] 和 [2]。",
        contexts,
        file_exists=lambda _path: True,
        read_lines=lambda _path: 5,
    )
    assert citations == [
        {"n": 1, "path": "src/a.py", "start_line": 2, "end_line": 5, "qualname": "foo"}
    ]
    assert invalid == [2]
    assert check == "1/2 通过"

    bad_range = [{**contexts[0], "start_line": 6, "end_line": 5}]
    citations, invalid, _ = verify_code_citations(
        "[1]", bad_range, file_exists=lambda _path: True, read_lines=lambda _path: 10
    )
    assert citations == [] and invalid == [1]

    citations, invalid, check = verify_code_citations(
        [1, 1], contexts, file_exists=lambda _path: True, read_lines=lambda _path: 10
    )
    assert len(citations) == 1 and invalid == [] and check == "1/1 通过"


async def test_ask_refuses_on_out_of_range_citation(make_app, store, repo_id) -> None:
    store.hits = [make_hit(repo_id)]
    llm = FakeLLM("只看到 [2]")
    async with client_for(make_app(llm=llm)) as client:
        resp = await client.post("/api/code/ask", json={"question": "foo?", "repo_id": repo_id})
    assert resp.status_code == 200
    body = resp.json()
    assert body["refusal"] is not None
    assert body["citations"] == []
    assert "引用校验" in body["refusal"]["reason"]


async def test_ask_refuses_on_llm_error(make_app, store, repo_id) -> None:
    store.hits = [make_hit(repo_id)]
    llm = FakeLLM(error=RuntimeError("boom"))
    async with client_for(make_app(llm=llm)) as client:
        resp = await client.post("/api/code/ask", json={"question": "foo?", "repo_id": repo_id})
    assert resp.status_code == 200
    body = resp.json()
    assert body["refusal"] is not None
    assert "LLM 调用失败" in body["refusal"]["reason"]


async def test_ask_provider_getter(make_app, store, repo_id, tmp_path) -> None:
    write_lines(tmp_path / "src/a.py", 10)
    store.hits = [make_hit(repo_id)]
    service = CodeService(
        store,
        llm=FakeLLM("结论 [1]"),
        provider_getter=lambda: "injected-provider",
    )
    async with client_for(make_app(service=service)) as client:
        resp = await client.post("/api/code/ask", json={"question": "foo?", "repo_id": repo_id})
    assert resp.status_code == 200
    assert resp.json()["provider"] == "injected-provider"


async def test_ask_without_llm_returns_503(make_app, store, repo_id) -> None:
    async with client_for(make_app(service=CodeService(store))) as client:
        resp = await client.post("/api/code/ask", json={"question": "foo?", "repo_id": repo_id})
    assert resp.status_code == 503


def test_ask_accepts_llm_as_third_positional(store, repo_id, tmp_path) -> None:
    write_lines(tmp_path / "src/a.py", 10)
    store.hits = [make_hit(repo_id)]
    out = CodeService(store).ask("foo?", repo_id, FakeLLM("结论 [1]"))
    assert out["citations"][0]["n"] == 1
    assert out["refusal"] is None


async def test_ask_errors(make_app) -> None:
    async with client_for(make_app(llm=FakeLLM(""))) as client:
        missing_repo = await client.post(
            "/api/code/ask", json={"question": "foo", "repo_id": "nope"}
        )
        blank = await client.post("/api/code/ask", json={"question": "   "})
    assert missing_repo.status_code == 404
    assert blank.status_code == 422


async def test_unassembled_returns_503() -> None:
    app = FastAPI()
    app.include_router(code_router)
    async with client_for(app) as client:
        search = await client.get("/api/code/search", params={"q": "foo"})
        repos = await client.get("/api/code/repos")
        create = await client.post("/api/code/repos", json={"root": "/tmp", "name": "x"})
        ask = await client.post("/api/code/ask", json={"question": "foo"})
    assert search.status_code == 503
    assert repos.status_code == 503
    assert create.status_code == 503
    assert ask.status_code == 503


# ---------- agent tools ----------

def test_agent_tools_execute(store, repo_id, monkeypatch) -> None:
    store.hits = [make_hit(repo_id)]
    store.symbols = [
        {
            "repo_id": repo_id,
            "path": "src/a.py",
            "qualname": "foo",
            "kind": "function",
            "start_line": 2,
            "end_line": 5,
            "docstring": "",
            "code": "def foo(): ...",
        }
    ]
    store.imports[(repo_id, "src/a.py")] = ["src/b.py"]
    store.overviews[repo_id] = {"file_count": 3, "symbol_count": 5}
    executor = CodeToolExecutor(CodeService(store, llm=FakeLLM("")))

    searched = executor.execute("search_code", {"query": "foo", "repo_id": repo_id, "k": 1})
    assert searched["ok"] is True
    assert searched["count"] == 1
    assert searched["hits"][0]["path"] == "src/a.py"
    assert searched["latency_ms"] >= 0

    symbol = executor.execute("get_symbol", {"repo_id": repo_id, "qualname": "foo"})
    assert symbol["ok"] is True
    assert symbol["symbol"]["code"].startswith("def foo")
    assert executor.execute("get_symbol", {"repo_id": repo_id, "qualname": "nope"})["ok"] is False

    impact = executor.execute("impact_analysis", {"repo_id": repo_id, "path": "src/a.py"})
    assert impact["ok"] is True
    assert impact["imports"] == ["src/b.py"]
    assert impact["heuristic"] is True

    overview = executor.execute("repo_overview", {"repo_id": repo_id})
    assert overview["ok"] is True
    assert overview["overview"]["file_count"] == 3

    unknown = executor.execute("missing_tool", {})
    assert unknown["ok"] is False
    assert "unknown tool" in unknown["error"]

    def boom(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        raise RuntimeError("boom")

    monkeypatch.setattr(store, "search", boom)
    failed = executor.execute("search_code", {"query": "foo"})
    assert failed["ok"] is False
    assert "RuntimeError" in failed["error"]

    assert executor.run("repo_overview", {"repo_id": repo_id})["ok"] is True


def test_agent_tool_schemas() -> None:
    names = [schema["function"]["name"] for schema in CODE_TOOL_SCHEMAS]
    assert names == ["search_code", "get_symbol", "impact_analysis", "repo_overview"]
    for schema in CODE_TOOL_SCHEMAS:
        assert schema["type"] == "function"
        assert schema["function"]["parameters"]["type"] == "object"
        assert "required" in schema["function"]["parameters"]
