"""Agentic 工具链与 LangSmith 追踪的单元测试（不依赖网络/GPU）。"""
from __future__ import annotations

import os

from ragm.agent.tools import TOOL_SCHEMAS, ToolExecutor, tool_result_text
from ragm.config import Settings
from ragm.observability import tracing


def test_tool_schemas() -> None:
    names = [t["function"]["name"] for t in TOOL_SCHEMAS]
    assert names == ["vector_search", "kg_query", "web_search"]
    for t in TOOL_SCHEMAS:
        assert t["function"]["parameters"]["type"] == "object"


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
