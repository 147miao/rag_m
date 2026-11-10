"""Agentic 预算与重试单测（fake LLM/executor，不依赖 GPU）。"""
from __future__ import annotations

from ragm.agent.agent_graph import AgentGraph
from ragm.agent.tools import TOOL_SCHEMAS


class FakeLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.schemas = None

    def chat_tools(self, messages, schemas, max_tokens=600):
        self.calls += 1
        self.schemas = schemas
        return {
            "text": "",
            "tool_calls": [{"id": f"c{self.calls}", "function": {"name": "vector_search", "arguments": '{"query": "q"}'}}],
            "tokens_in": 1,
            "tokens_out": 1,
        }

    def chat(self, messages, max_tokens):
        return {"text": "答案 [1]", "tokens_in": 1, "tokens_out": 1}


class FakeExecutor:
    def __init__(self, fail_times: int = 0) -> None:
        self.fail_times = fail_times
        self.runs = 0

    def run(self, name, args):
        self.runs += 1
        if self.runs <= self.fail_times:
            return {"ok": False, "note": "boom", "latency_ms": 1}
        return {"ok": True, "hits": [{"chunk_id": "c1", "title": "t", "text": "x"}], "latency_ms": 1}

    def vector_search(self, query, k):
        return {"ok": True, "hits": [{"chunk_id": "c1", "title": "t", "text": "x"}]}


def test_budget_stops_tool_calls() -> None:
    graph = AgentGraph(FakeLLM(), FakeExecutor(), max_iterations=10, max_calls=2, total_tokens=100000)
    out = graph.invoke("测试问题")
    assert len(out["tool_names"]) == 2


def test_retry_recovers() -> None:
    ex = FakeExecutor(fail_times=1)
    graph = AgentGraph(FakeLLM(), ex, max_iterations=2, max_calls=1, total_tokens=100000, retry=1)
    out = graph.invoke("测试问题")
    assert out["tool_log"][0]["status"] == "ok"
    assert out["tool_log"][0]["attempts"] == 2


def test_agent_graph_uses_custom_tool_schemas() -> None:
    llm = FakeLLM()
    custom = [{"type": "function", "function": {"name": "custom_tool", "parameters": {"type": "object"}}}]
    graph = AgentGraph(llm, FakeExecutor(), tool_schemas=custom)
    graph.invoke("测试问题")
    assert llm.schemas is custom


def test_agent_graph_defaults_to_doc_tool_schemas() -> None:
    llm = FakeLLM()
    graph = AgentGraph(llm, FakeExecutor())
    graph.invoke("测试问题")
    assert llm.schemas is TOOL_SCHEMAS