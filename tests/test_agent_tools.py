"""Agentic 工具链与 LangSmith 追踪的单元测试（不依赖网络/GPU）。"""
from __future__ import annotations

import os

from ragm.agent.tools import TOOL_SCHEMAS, ToolExecutor, build_tool_schemas, tool_result_text
from ragm.config import Settings
from ragm.observability import tracing


def test_tool_schemas() -> None:
    names = [t["function"]["name"] for t in TOOL_SCHEMAS]
    assert names == ["vector_search", "kg_query", "web_search"]
    for t in TOOL_SCHEMAS:
        assert t["function"]["parameters"]["type"] == "object"


def test_build_tool_schemas_default_and_with_code() -> None:
    default = build_tool_schemas()
    assert [t["function"]["name"] for t in default] == ["vector_search", "kg_query", "web_search"]
    assert default == TOOL_SCHEMAS

    with_code = build_tool_schemas(include_code=True)
    assert [t["function"]["name"] for t in with_code] == [
        "vector_search",
        "kg_query",
        "web_search",
        "search_code",
        "get_symbol",
        "impact_analysis",
        "repo_overview",
    ]
    assert len(with_code) == 7


class FakeCodeExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def execute(self, name: str, args: dict) -> dict:
        self.calls.append((name, args))
        return {"ok": True, "tool": name, "echo": args, "latency_ms": 0.5}


def test_tool_executor_dispatches_code_tools() -> None:
    code_executor = FakeCodeExecutor()
    executor = ToolExecutor(search=None, kg=None, settings=Settings(), code_executor=code_executor)
    out = executor.run("search_code", {"query": "ToolExecutor"})
    assert out["ok"] is True
    assert out["tool"] == "search_code"
    assert out["echo"] == {"query": "ToolExecutor"}
    assert code_executor.calls == [("search_code", {"query": "ToolExecutor"})]

    plain = ToolExecutor(search=None, kg=None, settings=Settings())
    assert plain.run("search_code", {"query": "x"})["ok"] is False


def test_tool_result_text_truncates() -> None:
    text = tool_result_text({"hits": ["x" * 5000]}, limit=100)
    assert len(text) <= 100


def test_unknown_tool_returns_error() -> None:
    executor = ToolExecutor(search=None, kg=None, settings=Settings())
    out = executor.run("nope", {})
    assert out["ok"] is False
    assert "unknown tool" in out["error"]


def test_kg_query_without_graph() -> None:
    executor = ToolExecutor(search=None, kg=None, settings=Settings())
    out = executor.kg_query("Milvus")
    assert out["ok"] is False


class FakeKG:
    def __init__(self) -> None:
        self.names = ["Milvus", "Milvus Cloud"]
        self.calls: list[dict] = []

    def expand_hops(self, names: list[str], hops: int = 2, limit: int = 40):
        self.calls.append({"names": names, "hops": hops, "limit": limit})
        return [{"a": "Milvus", "b": "BM25", "qtype": "co_evidence"}], ["BM25"]


def test_kg_query_honors_hops() -> None:
    kg = FakeKG()
    executor = ToolExecutor(search=None, kg=kg, settings=Settings())
    out = executor.kg_query("Milvus", hops=2)
    assert out["ok"] is True
    assert set(kg.calls[0]["names"]) == {"Milvus", "Milvus Cloud"}
    assert kg.calls[0]["hops"] == 2
    assert kg.calls[0]["limit"] == 30
    assert out["paths"][0]["b"] == "BM25"

    executor.run("kg_query", {"entity": "Milvus", "hops": 1})
    assert kg.calls[-1]["hops"] == 1


def test_tracing_disabled_by_default() -> None:
    assert tracing.configure_tracing(Settings(langsmith_tracing=False, langsmith_api_key=None)) is False
    assert tracing.tracing_enabled() is False


def test_tracing_enabled_sets_env(monkeypatch) -> None:
    for key in ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT"):
        monkeypatch.delenv(key, raising=False)
    enabled = tracing.configure_tracing(
        Settings(langsmith_tracing=True, langsmith_api_key="lsv2_test_dummy", langsmith_project="rag-m-test")
    )
    assert enabled is True
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGSMITH_PROJECT"] == "rag-m-test"
    # 关闭，避免影响其他测试
    tracing.configure_tracing(Settings(langsmith_tracing=False, langsmith_api_key=None))
    assert tracing.tracing_enabled() is False


def test_traceable_identity_when_disabled() -> None:
    tracing.configure_tracing(Settings(langsmith_tracing=False, langsmith_api_key=None))

    @tracing.traceable(run_type="chain", name="unit")
    def add(a: int, b: int) -> int:
        return a + b

    assert add(1, 2) == 3
